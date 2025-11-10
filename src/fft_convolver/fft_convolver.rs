use crate::fft_convolver::traits::{ComplexOps, FftBackend};
use crate::Convolution;

#[derive(Default, Clone)]
pub struct GenericFFTConvolver<F: FftBackend, C: ComplexOps<Complex = F::Complex>>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    ir_len: usize,
    block_size: usize,
    _seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    _fft_complex_size: usize,
    segments: Vec<Vec<F::Complex>>,
    segments_ir: Vec<Vec<F::Complex>>,
    fft_buffer: Vec<f32>,
    fft: F,
    pre_multiplied: Vec<F::Complex>,
    conv: Vec<F::Complex>,
    overlap: Vec<f32>,
    current: usize,
    input_buffer: Vec<f32>,
    input_buffer_fill: usize,
    _complex_ops: std::marker::PhantomData<C>,
    temp_buffer: Option<Vec<F::Complex>>,
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> Convolution for GenericFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    fn init(impulse_response: &[f32], block_size: usize, max_response_length: usize) -> Self {
        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);
        let ir_len = padded_ir.len();

        let block_size = block_size.next_power_of_two();
        let seg_size = 2 * block_size;
        let seg_count = (ir_len as f64 / block_size as f64).ceil() as usize;
        let active_seg_count = seg_count;
        let fft_complex_size = C::complex_size(seg_size);

        // FFT
        let mut fft = F::default();
        fft.init(seg_size);
        let fft_buffer = vec![0.; seg_size];

        // prepare segments
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            segments.push(vec![F::Complex::default(); fft_complex_size]);
        }

        let mut segments_ir = Vec::with_capacity(seg_count);

        // prepare ir
        let mut ir_buffer = vec![0.0; seg_size];
        for i in 0..seg_count {
            let mut segment = vec![F::Complex::default(); fft_complex_size];
            let remaining = ir_len - (i * block_size);
            let size_copy = if remaining >= block_size {
                block_size
            } else {
                remaining
            };
            C::copy_and_pad(&mut ir_buffer, &padded_ir[i * block_size..], size_copy);
            fft.forward(&mut ir_buffer, &mut segment).unwrap();
            segments_ir.push(segment);
        }

        // prepare convolution buffers
        let pre_multiplied = vec![F::Complex::default(); fft_complex_size];
        let conv = vec![F::Complex::default(); fft_complex_size];
        let overlap = vec![0.; block_size];

        // prepare input buffer
        let input_buffer = vec![0.; block_size];
        let input_buffer_fill = 0;

        // reset current position
        let current = 0;

        // For IPP backend we need a temp buffer
        let temp_buffer = Some(vec![F::Complex::default(); fft_complex_size]);

        Self {
            ir_len,
            block_size,
            _seg_size: seg_size,
            seg_count,
            active_seg_count,
            _fft_complex_size: fft_complex_size,
            segments,
            segments_ir,
            fft_buffer,
            fft,
            pre_multiplied,
            conv,
            overlap,
            current,
            input_buffer,
            input_buffer_fill,
            _complex_ops: std::marker::PhantomData,
            temp_buffer,
        }
    }

    fn update(&mut self, response: &[f32]) {
        let new_ir_len = response.len();

        if new_ir_len > self.ir_len {
            panic!("New impulse response is longer than initialized length");
        }

        if self.ir_len == 0 {
            return;
        }

        C::zero_real(&mut self.fft_buffer);
        C::zero_complex(&mut self.conv);
        C::zero_complex(&mut self.pre_multiplied);
        C::zero_real(&mut self.overlap);

        self.active_seg_count = ((new_ir_len as f64 / self.block_size as f64).ceil()) as usize;

        // Prepare IR
        for i in 0..self.active_seg_count {
            let segment = &mut self.segments_ir[i];
            let remaining = new_ir_len - (i * self.block_size);
            let size_copy = if remaining >= self.block_size {
                self.block_size
            } else {
                remaining
            };
            C::copy_and_pad(
                &mut self.fft_buffer,
                &response[i * self.block_size..],
                size_copy,
            );
            self.fft.forward(&mut self.fft_buffer, segment).unwrap();
        }

        // Clear remaining segments
        for i in self.active_seg_count..self.seg_count {
            C::zero_complex(&mut self.segments_ir[i]);
        }
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if self.active_seg_count == 0 {
            C::zero_real(output);
            return;
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - self.input_buffer_fill,
            );

            // Copy input to input buffer
            let input_buffer_pos = self.input_buffer_fill;
            C::copy_and_pad(
                &mut self.input_buffer[input_buffer_pos..],
                &input[processed..processed + processing],
                processing,
            );

            // Forward FFT
            C::copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            if let Err(_) = self
                .fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current])
            {
                C::zero_real(output);
                return;
            }

            // complex multiplication
            if input_buffer_was_empty {
                C::zero_complex(&mut self.pre_multiplied);

                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    C::complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                        self.temp_buffer.as_mut().map(|v| v.as_mut_slice()),
                    );
                }
            }

            C::copy_complex(&mut self.conv, &self.pre_multiplied);

            C::complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
                self.temp_buffer.as_mut().map(|v| v.as_mut_slice()),
            );

            // Backward FFT
            if let Err(_) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                C::zero_real(output);
                return;
            }

            // Add overlap
            C::sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                &self.overlap[input_buffer_pos..input_buffer_pos + processing],
            );

            // Input buffer full => Next block
            self.input_buffer_fill += processing;
            if self.input_buffer_fill == self.block_size {
                // Input buffer is empty again now
                C::zero_real(&mut self.input_buffer);
                self.input_buffer_fill = 0;

                // Save the overlap
                C::copy_and_pad(
                    &mut self.overlap,
                    &self.fft_buffer[self.block_size..],
                    self.block_size,
                );

                // Update the current segment
                self.current = if self.current > 0 {
                    self.current - 1
                } else {
                    self.active_seg_count - 1
                };
            }
            processed += processing;
        }
    }
}
