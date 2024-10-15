use crate::fft_convolver::{copy_and_pad, FFTConvolver};
use crate::{Convolution, Sample};

#[derive(Clone)]
pub struct StepwiseUpdateConvolver {
    convolver: FFTConvolver,
    buffer: Vec<Sample>,
    stored_response: Vec<Sample>,
    next_response: Vec<Sample>,
    segment_to_load: usize,
    switching: bool,
    response_pending: bool,
}

impl StepwiseUpdateConvolver {
    pub fn new(
        convolver: FFTConvolver,
        max_response_length: usize,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            convolver,
            buffer: vec![0.0; max_buffer_size],
            stored_response: vec![0.0; max_response_length],
            next_response: vec![0.0; max_response_length],
            segment_to_load: 0,
            switching: false,
            response_pending: false,
        }
    }
}

impl Convolution for StepwiseUpdateConvolver {
    fn init(response: &[Sample], max_block_size: usize, max_response_length: usize) -> Self {
        let convolver = FFTConvolver::init(response, max_block_size, max_response_length);
        Self::new(convolver, response.len(), max_block_size)
    }

    fn update(&mut self, response: &[Sample]) {
        let response_len = response.len();
        assert!(response_len <= self.stored_response.len());

        if !self.switching {
            copy_and_pad(&mut self.stored_response[..], response, response_len);
            self.switching = true;
            self.response_pending = false;
            return;
        }

        copy_and_pad(&mut self.next_response[..], response, response_len);
        self.response_pending = true;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if !self.switching && self.response_pending {
            self.stored_response = self.next_response.clone();
            self.response_pending = false;
            self.switching = true;
        }

        if self.switching {
            self.convolver
                .update_segment(&self.stored_response, self.segment_to_load);
            self.segment_to_load += 1;
            if &self.segment_to_load == self.convolver.active_seg_count() {
                self.switching = false;
                self.segment_to_load = 0;
            }
        }

        self.convolver.process(input, &mut self.buffer);

        for i in 0..output.len() {
            // is this necessary?
            output[i] = self.buffer[i];
        }
    }
}

#[test]
fn test_crossfade_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = StepwiseUpdateConvolver::new(
        crate::fft_convolver::FFTConvolver::init(&response, 1024, 1024),
        1024,
        1024,
    );
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
