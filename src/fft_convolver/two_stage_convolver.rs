use crate::fft_convolver::fft_convolver::GenericFFTConvolver;
use crate::fft_convolver::traits::{ComplexOps, FftBackend};
use crate::Convolution;

const DEFAULT_HEAD_BLOCK_SIZE: usize = 128;
const DEFAULT_TAIL_BLOCK_SIZE: usize = 1024;

#[derive(Clone)]
pub struct GenericTwoStageFFTConvolver<F: FftBackend, C: ComplexOps<Complex = F::Complex>>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    head_convolver: GenericFFTConvolver<F, C>,
    tail_convolver0: GenericFFTConvolver<F, C>,
    tail_output0: Vec<f32>,
    tail_precalculated0: Vec<f32>,
    tail_convolver: GenericFFTConvolver<F, C>,
    tail_output: Vec<f32>,
    tail_precalculated: Vec<f32>,
    tail_input: Vec<f32>,
    tail_input_fill: usize,
    precalculated_pos: usize,
    head_block_size: usize,
    tail_block_size: usize,
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> GenericTwoStageFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    pub fn with_sizes(
        impulse_response: &[f32],
        _block_size: usize,
        max_response_length: usize,
        head_block_size: usize,
        tail_block_size: usize,
    ) -> Self {
        if max_response_length < impulse_response.len() {
            panic!(
                "max_response_length must be at least the length of the initial impulse response"
            );
        }
        let mut padded_ir = impulse_response.to_vec();
        padded_ir.resize(max_response_length, 0.);

        let head_ir_len = std::cmp::min(max_response_length, tail_block_size);
        let head_convolver = GenericFFTConvolver::<F, C>::init(
            &padded_ir[0..head_ir_len],
            head_block_size,
            max_response_length,
        );

        let tail_convolver0 = if max_response_length > tail_block_size {
            let tail_ir_len = std::cmp::min(max_response_length - tail_block_size, tail_block_size);
            GenericFFTConvolver::<F, C>::init(
                &padded_ir[tail_block_size..tail_block_size + tail_ir_len],
                head_block_size,
                max_response_length,
            )
        } else {
            GenericFFTConvolver::<F, C>::default()
        };

        let tail_output0 = vec![0.0; tail_block_size];
        let tail_precalculated0 = vec![0.0; tail_block_size];

        let tail_convolver = if max_response_length > 2 * tail_block_size {
            let tail_ir_len = max_response_length - 2 * tail_block_size;
            GenericFFTConvolver::<F, C>::init(
                &padded_ir[2 * tail_block_size..2 * tail_block_size + tail_ir_len],
                tail_block_size,
                max_response_length,
            )
        } else {
            GenericFFTConvolver::<F, C>::default()
        };

        let tail_output = vec![0.0; tail_block_size];
        let tail_precalculated = vec![0.0; tail_block_size];
        let tail_input = vec![0.0; tail_block_size];
        let tail_input_fill = 0;
        let precalculated_pos = 0;

        GenericTwoStageFFTConvolver {
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
            head_block_size,
            tail_block_size,
        }
    }
}

impl<F: FftBackend, C: ComplexOps<Complex = F::Complex>> Convolution
    for GenericTwoStageFFTConvolver<F, C>
where
    C: Default,
    F: Default,
    F::Complex: Clone + Default,
{
    fn init(impulse_response: &[f32], _block_size: usize, max_response_length: usize) -> Self {
        Self::with_sizes(
            impulse_response,
            _block_size,
            max_response_length,
            DEFAULT_HEAD_BLOCK_SIZE,
            DEFAULT_TAIL_BLOCK_SIZE,
        )
    }

    fn update(&mut self, _response: &[f32]) {
        todo!()
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let head_block_size = self.head_block_size;
        let tail_block_size = self.tail_block_size;

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
                head_block_size - (self.tail_input_fill % head_block_size),
            );

            // Sum head and tail
            let sum_begin = processed;

            // Sum: 1st tail block
            if !self.tail_precalculated0.is_empty() {
                C::add_to_buffer(
                    &mut output[sum_begin..sum_begin + processing],
                    &self.tail_precalculated0
                        [self.precalculated_pos..self.precalculated_pos + processing],
                );
            }

            // Sum: 2nd-Nth tail block
            if !self.tail_precalculated.is_empty() {
                C::add_to_buffer(
                    &mut output[sum_begin..sum_begin + processing],
                    &self.tail_precalculated
                        [self.precalculated_pos..self.precalculated_pos + processing],
                );
            }
            self.precalculated_pos += processing;

            // Fill input buffer for tail convolution
            C::copy_and_pad(
                &mut self.tail_input[self.tail_input_fill..],
                &input[processed..processed + processing],
                processing,
            );
            self.tail_input_fill += processing;

            // Convolution: 1st tail block
            if !self.tail_precalculated0.is_empty() && self.tail_input_fill % head_block_size == 0 {
                assert!(self.tail_input_fill >= head_block_size);
                let block_offset = self.tail_input_fill - head_block_size;
                self.tail_convolver0.process(
                    &self.tail_input[block_offset..block_offset + head_block_size],
                    &mut self.tail_output0[block_offset..block_offset + head_block_size],
                );
                if self.tail_input_fill == tail_block_size {
                    std::mem::swap(&mut self.tail_precalculated0, &mut self.tail_output0);
                }
            }

            // Convolution: 2nd-Nth tail block (might be done in some background thread)
            if !self.tail_precalculated.is_empty()
                && self.tail_input_fill == tail_block_size
                && self.tail_output.len() == tail_block_size
            {
                std::mem::swap(&mut self.tail_precalculated, &mut self.tail_output);
                self.tail_convolver
                    .process(&self.tail_input, &mut self.tail_output);
            }

            if self.tail_input_fill == tail_block_size {
                self.tail_input_fill = 0;
                self.precalculated_pos = 0;
            }

            processed += processing;
        }
    }
}
