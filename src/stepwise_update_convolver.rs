use crate::fft_convolver::{copy_and_pad, FFTConvolver};
use crate::{Convolution, Sample};

#[derive(Clone)]
pub struct StepwiseUpdateConvolver {
    convolver: FFTConvolver,
    buffer: Vec<Sample>,
    current_response: Vec<Sample>,
    next_response: Vec<Sample>,
    queued_response: Vec<Sample>,
    segment_to_load: usize,
    scale_factor: usize,
    transition_counter: usize,
    switching: bool,
    response_pending: bool,
}

impl StepwiseUpdateConvolver {
    pub fn new(
        response: &[Sample],
        max_response_length: usize,
        max_buffer_size: usize,
        scale_factor: usize,
    ) -> Self {
        Self {
            convolver: FFTConvolver::init(response, max_buffer_size, max_response_length),
            buffer: vec![0.0; max_buffer_size],
            current_response: response.to_vec(),
            next_response: vec![0.0; max_response_length],
            queued_response: vec![0.0; max_response_length],
            segment_to_load: 0,
            scale_factor,
            transition_counter: 0,
            switching: false,
            response_pending: false,
        }
    }
}

impl Convolution for StepwiseUpdateConvolver {
    fn init(response: &[Sample], max_block_size: usize, max_response_length: usize) -> Self {
        Self::new(response, max_response_length, max_block_size, 1)
    }

    fn update(&mut self, response: &[Sample]) {
        let response_len = response.len();
        assert!(response_len <= self.next_response.len());

        if !self.switching {
            copy_and_pad(&mut self.next_response[..], response, response_len);
            self.switching = true;
            self.response_pending = false;
            return;
        }

        copy_and_pad(&mut self.queued_response[..], response, response_len);
        self.response_pending = true;
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if !self.switching && self.response_pending {
            self.next_response = self.queued_response.clone();
            self.response_pending = false;
            self.switching = true;
        }

        if self.switching {
            self.segment_to_load = self.transition_counter / self.scale_factor;
            let response = mix(
                &self.current_response,
                &self.next_response,
                ((self.transition_counter % self.scale_factor) as f32 + 1.0)
                    / self.scale_factor as f32,
            );

            self.convolver
                .update_segment(&response, self.segment_to_load);
            self.transition_counter += 1;
            if &(self.transition_counter / self.scale_factor) == self.convolver.active_seg_count() {
                self.current_response = self.next_response.clone();
                self.switching = false;
                self.transition_counter = 0;
            }
        }

        self.convolver.process(input, &mut self.buffer);

        for i in 0..output.len() {
            // is this necessary?
            output[i] = self.buffer[i];
        }
    }
}

fn mix(response_a: &[Sample], response_b: &[Sample], weight: f32) -> Vec<Sample> {
    assert_eq!(response_a.len(), response_b.len());
    assert!(weight <= 1.0 && weight >= 0.0);
    response_a
        .iter()
        .zip(response_b.iter())
        .map(|(&a, &b)| a * (1.0 - weight) + b * weight)
        .collect()
}

#[test]
fn test_crossfade_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = StepwiseUpdateConvolver::new(&response, 1024, 1024, 1);
    let input = vec![1.0; 1024];
    let mut output = vec![0.0; 1024];
    convolver.process(&input, &mut output);

    for i in 0..1024 {
        assert!((output[i] - 1.0).abs() < 1e-6);
    }
}
