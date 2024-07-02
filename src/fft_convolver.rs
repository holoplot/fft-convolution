use realfft::{ComplexToReal, FftError, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

use crate::{Convolution, Sample};

#[derive(Clone)]
pub struct Fft {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
}

impl Default for Fft {
    fn default() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            fft_forward: planner.plan_fft_forward(0),
            fft_inverse: planner.plan_fft_inverse(0),
        }
    }
}

impl std::fmt::Debug for Fft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl Fft {
    pub fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::<f32>::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
    }

    pub fn forward(&self, input: &mut [f32], output: &mut [Complex<f32>]) -> Result<(), FftError> {
        self.fft_forward.process(input, output)?;
        Ok(())
    }

    pub fn inverse(&self, input: &mut [Complex<f32>], output: &mut [f32]) -> Result<(), FftError> {
        self.fft_inverse.process(input, output)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| *bin /= len as f32);

        Ok(())
    }
}

pub fn complex_size(size: usize) -> usize {
    (size / 2) + 1
}

pub fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
    assert!(dst.len() >= src_size);
    dst[0..src_size].clone_from_slice(&src[0..src_size]);
    dst[src_size..].iter_mut().for_each(|value| *value = 0.);
}

pub fn complex_multiply_accumulate(
    result: &mut [Complex<f32>],
    a: &[Complex<f32>],
    b: &[Complex<f32>],
) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 4 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0].re += a[i + 0].re * b[i + 0].re - a[i + 0].im * b[i + 0].im;
        result[i + 1].re += a[i + 1].re * b[i + 1].re - a[i + 1].im * b[i + 1].im;
        result[i + 2].re += a[i + 2].re * b[i + 2].re - a[i + 2].im * b[i + 2].im;
        result[i + 3].re += a[i + 3].re * b[i + 3].re - a[i + 3].im * b[i + 3].im;
        result[i + 0].im += a[i + 0].re * b[i + 0].im + a[i + 0].im * b[i + 0].re;
        result[i + 1].im += a[i + 1].re * b[i + 1].im + a[i + 1].im * b[i + 1].re;
        result[i + 2].im += a[i + 2].re * b[i + 2].im + a[i + 2].im * b[i + 2].re;
        result[i + 3].im += a[i + 3].re * b[i + 3].im + a[i + 3].im * b[i + 3].re;
    }
    for i in end4..len {
        result[i].re += a[i].re * b[i].re - a[i].im * b[i].im;
        result[i].im += a[i].re * b[i].im + a[i].im * b[i].re;
    }
}

pub fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(result.len(), a.len());
    assert_eq!(result.len(), b.len());
    let len = result.len();
    let end4 = 3 * (len / 4);
    for i in (0..end4).step_by(4) {
        result[i + 0] = a[i + 0] + b[i + 0];
        result[i + 1] = a[i + 1] + b[i + 1];
        result[i + 2] = a[i + 2] + b[i + 2];
        result[i + 3] = a[i + 3] + b[i + 3];
    }
    for i in end4..len {
        result[i] = a[i] + b[i];
    }
}
#[derive(Default, Clone)]
pub struct FFTConvolver {
    ir_len: usize,
    block_size: usize,
    _seg_size: usize,
    seg_count: usize,
    active_seg_count: usize,
    _fft_complex_size: usize,
    segments: Vec<Vec<Complex<f32>>>,
    segments_ir: Vec<Vec<Complex<f32>>>,
    fft_buffer: Vec<f32>,
    fft: Fft,
    pre_multiplied: Vec<Complex<f32>>,
    conv: Vec<Complex<f32>>,
    overlap: Vec<f32>,
    current: usize,
    input_buffer: Vec<f32>,
    input_buffer_fill: usize,
}

impl FFTConvolver {
    fn update_segment(&mut self, response: &[Sample], index: usize) {
        let segment = &mut self.segments_ir[index];
        let remaining = response.len() - index * self.block_size;
        let copy_size = std::cmp::min(self.block_size, remaining);
        copy_and_pad(
            &mut self.fft_buffer,
            &response[index * self.block_size..],
            copy_size,
        );
        self.fft.forward(&mut self.fft_buffer, segment).unwrap();
    }
}

impl Convolution for FFTConvolver {
    fn init(impulse_response: &[Sample], block_size: usize, max_response_length: usize) -> Self {
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
        let fft_complex_size = complex_size(seg_size);

        // FFT
        let mut fft = Fft::default();
        fft.init(seg_size);
        let mut fft_buffer = vec![0.; seg_size];

        // prepare segments
        let segments = vec![vec![Complex::new(0., 0.); fft_complex_size]; seg_count];
        let mut segments_ir = Vec::new();

        // prepare ir
        for i in 0..seg_count {
            let mut segment = vec![Complex::new(0., 0.); fft_complex_size];
            let remaining = ir_len - (i * block_size);
            let size_copy = if remaining >= block_size {
                block_size
            } else {
                remaining
            };
            copy_and_pad(&mut fft_buffer, &padded_ir[i * block_size..], size_copy);
            fft.forward(&mut fft_buffer, &mut segment).unwrap();
            segments_ir.push(segment);
        }

        // prepare convolution buffers
        let pre_multiplied = vec![Complex::new(0., 0.); fft_complex_size];
        let conv = vec![Complex::new(0., 0.); fft_complex_size];
        let overlap = vec![0.; block_size];

        // prepare input buffer
        let input_buffer = vec![0.; block_size];
        let input_buffer_fill = 0;

        // reset current position
        let current = 0;

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
        }
    }

    fn update(&mut self, response: &[Sample]) {
        let new_ir_len = response.len();

        if new_ir_len > self.ir_len {
            panic!("New impulse response is longer than initialized length");
        }

        if self.ir_len == 0 {
            return;
        }

        self.fft_buffer.fill(0.);
        self.conv.fill(Complex::new(0., 0.));
        self.pre_multiplied.fill(Complex::new(0., 0.));
        self.overlap.fill(0.);

        self.active_seg_count = ((new_ir_len as f64 / self.block_size as f64).ceil()) as usize;

        // Prepare IR
        for i in 0..self.active_seg_count {
            self.update_segment(response, i);
        }

        // Clear remaining segments
        for i in self.active_seg_count..self.seg_count {
            self.segments_ir[i].fill(Complex::new(0., 0.));
        }

        self.input_buffer.fill(0.);
        self.input_buffer_fill = 0;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if self.active_seg_count == 0 {
            output.fill(0.);
            return;
        }

        let mut processed = 0;
        while processed < output.len() {
            let input_buffer_was_empty = self.input_buffer_fill == 0;
            let processing = std::cmp::min(
                output.len() - processed,
                self.block_size - self.input_buffer_fill,
            );

            let input_buffer_pos = self.input_buffer_fill;
            self.input_buffer[input_buffer_pos..input_buffer_pos + processing]
                .clone_from_slice(&input[processed..processed + processing]);

            // Forward FFT
            copy_and_pad(&mut self.fft_buffer, &self.input_buffer, self.block_size);
            if let Err(_err) = self
                .fft
                .forward(&mut self.fft_buffer, &mut self.segments[self.current])
            {
                output.fill(0.);
                return; // error!
            }

            // complex multiplication
            if input_buffer_was_empty {
                self.pre_multiplied.fill(Complex { re: 0., im: 0. });
                for i in 1..self.active_seg_count {
                    let index_ir = i;
                    let index_audio = (self.current + i) % self.active_seg_count;
                    complex_multiply_accumulate(
                        &mut self.pre_multiplied,
                        &self.segments_ir[index_ir],
                        &self.segments[index_audio],
                    );
                }
            }
            self.conv.clone_from_slice(&self.pre_multiplied);
            complex_multiply_accumulate(
                &mut self.conv,
                &self.segments[self.current],
                &self.segments_ir[0],
            );

            // Backward FFT
            if let Err(_err) = self.fft.inverse(&mut self.conv, &mut self.fft_buffer) {
                output.fill(0.);
                return; // error!
            }

            // Add overlap
            sum(
                &mut output[processed..processed + processing],
                &self.fft_buffer[input_buffer_pos..input_buffer_pos + processing],
                &self.overlap[input_buffer_pos..input_buffer_pos + processing],
            );

            // Input buffer full => Next block
            self.input_buffer_fill += processing;
            if self.input_buffer_fill == self.block_size {
                // Input buffer is empty again now
                self.input_buffer.fill(0.);
                self.input_buffer_fill = 0;
                // Save the overlap
                self.overlap
                    .clone_from_slice(&self.fft_buffer[self.block_size..self.block_size * 2]);

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

#[test]
fn test_fft_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = FFTConvolver::init(&response, 1024, response.len());
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}

#[derive(Clone)]
pub struct TwoStageFFTConvolver {
    head_convolver: FFTConvolver,
    tail_convolver0: FFTConvolver,
    tail_output0: Vec<Sample>,
    tail_precalculated0: Vec<Sample>,
    tail_convolver: FFTConvolver,
    tail_output: Vec<Sample>,
    tail_precalculated: Vec<Sample>,
    tail_input: Vec<Sample>,
    tail_input_fill: usize,
    precalculated_pos: usize,
}

const HEAD_BLOCK_SIZE: usize = 128;
const TAIL_BLOCK_SIZE: usize = 1024;

impl Convolution for TwoStageFFTConvolver {
    fn init(impulse_response: &[Sample], _block_size: usize, max_response_length: usize) -> Self {
        let head_block_size = HEAD_BLOCK_SIZE;
        let tail_block_size = TAIL_BLOCK_SIZE;

        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);

        let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
        let head_convolver = FFTConvolver::init(
            &padded_ir[0..head_ir_len],
            head_block_size,
            max_response_length,
        );

        let tail_convolver0 = (max_response_length > tail_block_size)
            .then(|| {
                let tail_ir_len =
                    std::cmp::min(max_response_length - tail_block_size, tail_block_size);
                FFTConvolver::init(
                    &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                    head_block_size,
                    max_response_length,
                )
            })
            .unwrap_or_default();

        let tail_output0 = vec![0.0; tail_block_size];
        let tail_precalculated0 = vec![0.0; tail_block_size];

        let tail_convolver = (max_response_length > 2 * tail_block_size)
            .then(|| {
                let tail_ir_len = max_response_length - 2 * tail_block_size;
                FFTConvolver::init(
                    &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                    tail_block_size,
                    max_response_length,
                )
            })
            .unwrap_or_default();

        let tail_output = vec![0.0; tail_block_size];
        let tail_precalculated = vec![0.0; tail_block_size];
        let tail_input = vec![0.0; tail_block_size];
        let tail_input_fill = 0;
        let precalculated_pos = 0;

        TwoStageFFTConvolver {
            head_convolver,
            tail_convolver0,
            tail_output0,
            tail_precalculated0,
            tail_convolver,
            tail_output,
            tail_precalculated,
            tail_input,
            tail_input_fill,
            precalculated_pos,
        }
    }

    fn update(&mut self, _response: &[Sample]) {
        todo!()
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        // Head
        self.head_convolver.process(input, output);

        // Tail
        if self.tail_input.is_empty() {
            return;
        }

        let len = input.len();
        let mut processed = 0;

        while processed < len {
            let remaining = len - processed;
            let processing = std::cmp::min(
                remaining,
                HEAD_BLOCK_SIZE - (self.tail_input_fill % HEAD_BLOCK_SIZE),
            );

            // Sum head and tail
            let sum_begin = processed;
            let sum_end = processed + processing;

            // Sum: 1st tail block
            if self.tail_precalculated0.len() > 0 {
                let mut precalculated_pos = self.precalculated_pos;
                for i in sum_begin..sum_end {
                    output[i] += self.tail_precalculated0[precalculated_pos];
                    precalculated_pos += 1;
                }
            }

            // Sum: 2nd-Nth tail block
            if self.tail_precalculated.len() > 0 {
                let mut precalculated_pos = self.precalculated_pos;
                for i in sum_begin..sum_end {
                    output[i] += self.tail_precalculated[precalculated_pos];
                    precalculated_pos += 1;
                }
            }

            self.precalculated_pos += processing;

            // Fill input buffer for tail convolution
            self.tail_input[self.tail_input_fill..self.tail_input_fill + processing]
                .copy_from_slice(&input[processed..processed + processing]);
            self.tail_input_fill += processing;

            // Convolution: 1st tail block
            if self.tail_precalculated0.len() > 0 && self.tail_input_fill % HEAD_BLOCK_SIZE == 0 {
                assert!(self.tail_input_fill >= HEAD_BLOCK_SIZE);
                let block_offset = self.tail_input_fill - HEAD_BLOCK_SIZE;
                self.tail_convolver0.process(
                    &self.tail_input[block_offset..block_offset + HEAD_BLOCK_SIZE],
                    &mut self.tail_output0[block_offset..block_offset + HEAD_BLOCK_SIZE],
                );
                if self.tail_input_fill == TAIL_BLOCK_SIZE {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            // Convolution: 2nd-Nth tail block (might be done in some background thread)
            if self.tail_precalculated.len() > 0
                && self.tail_input_fill == TAIL_BLOCK_SIZE
                && self.tail_output.len() == TAIL_BLOCK_SIZE
            {
                std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                self.tail_convolver
                    .process(&self.tail_input, &mut self.tail_output);
            }

            if self.tail_input_fill == TAIL_BLOCK_SIZE {
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }
    }
}
