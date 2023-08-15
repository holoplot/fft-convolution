use crate::{Conv, Sample, SmoothConvUpdate};

#[derive(Clone)]
struct CrossfadeConvolverCore<T: Conv> {
    convolver_a: T,
    convolver_b: T,
    crossfader: Crossfader<RaisedCosineMixer>,
}

#[derive(Clone)]
pub struct CrossfadeConvolver<Convolver: Conv> {
    core: CrossfadeConvolverCore<Convolver>,
    buffer_a: Vec<Sample>,
    buffer_b: Vec<Sample>,
    stored_response: Vec<f32>,
    response_pending: bool,
    _crossfade_samples: usize,
    _crossfade_counter: usize,
}

impl<T: Conv> CrossfadeConvolver<T> {
    pub fn new(
        convolver: T,
        max_response_length: usize,
        max_buffer_size: usize,
        crossfade_samples: usize,
    ) -> Self {
        let stored_response = vec![0.0; max_response_length];
        Self {
            core: CrossfadeConvolverCore {
                convolver_a: convolver.clone(),
                convolver_b: convolver,
                crossfader: Crossfader::new(RaisedCosineMixer, crossfade_samples),
            },
            buffer_a: vec![0.0; max_buffer_size],
            buffer_b: vec![0.0; max_buffer_size],
            stored_response,
            response_pending: false,
            _crossfade_samples: crossfade_samples,
            _crossfade_counter: 0,
        }
    }
}

impl<Convolver: Conv> Conv for CrossfadeConvolver<Convolver> {
    fn init(response: &[Sample], max_block_size: usize) -> Self {
        let convolver = Convolver::init(response, max_block_size);
        Self::new(convolver, response.len(), max_block_size, response.len())
    }

    fn set_response(&mut self, response: &[Sample]) {
        self.core.convolver_a.set_response(response);
        self.core.convolver_b.set_response(response);
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) {
        if !self.is_crossfading() && self.response_pending {
            swap(&mut self.core, &mut self.stored_response);
            self.response_pending = false;
        }

        self.core.convolver_a.process(input, &mut self.buffer_a);
        self.core.convolver_b.process(input, &mut self.buffer_b);

        for i in 0..output.len() {
            output[i] = self.core.crossfader.mix(self.buffer_a[i], self.buffer_b[i]);
        }
    }
}

impl<Convolver: Conv> SmoothConvUpdate for CrossfadeConvolver<Convolver> {
    fn evolve(&mut self, response: &[Sample]) {
        if !self.is_crossfading() {
            swap(&mut self.core, response);
            self.response_pending = false;
            return;
        }

        let response_len = response.len();
        assert!(response_len <= self.stored_response.len());

        self.stored_response[..response_len].copy_from_slice(response);
        self.stored_response[response_len..].fill(0.0);
        self.response_pending = true;
    }
}

impl<Convolver: Conv> CrossfadeConvolver<Convolver> {
    pub fn is_crossfading(&self) -> bool {
        match self.core.crossfader.fading_state {
            FadingState::Approaching(_) => true,
            FadingState::Reached(_) => false,
        }
    }
}

fn swap<T: Conv>(core: &mut CrossfadeConvolverCore<T>, response: &[Sample]) {
    match core.crossfader.fading_state.target() {
        Target::A => {
            core.convolver_b.set_response(response);
            core.crossfader.fade_into(Target::B);
        }
        Target::B => {
            core.convolver_a.set_response(response);
            core.crossfader.fade_into(Target::A);
        }
    }
}

#[test]
fn test_crossfade_convolver() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = CrossfadeConvolver::new(
        crate::fft_convolver::FFTConvolver::init(&response, 1024),
        1024,
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

pub trait Mixer {
    fn mix(&self, a: f32, b: f32, value: f32) -> f32;
}

struct LinearMixer;
impl Mixer for LinearMixer {
    fn mix(&self, a: f32, b: f32, value: f32) -> f32 {
        a * (1.0 - value) + b * value
    }
}

struct SquareRootMixer;
impl Mixer for SquareRootMixer {
    fn mix(&self, a: f32, b: f32, value: f32) -> f32 {
        let gain1 = (1.0 - value).sqrt();
        let gain2 = value.sqrt();
        a * gain1 + b * gain2
    }
}
const PI_HALF: f32 = std::f32::consts::PI * 0.5;

struct CosineMixer;
impl Mixer for CosineMixer {
    fn mix(&self, a: f32, b: f32, value: f32) -> f32 {
        let rad = PI_HALF * value;
        let gain1 = rad.cos();
        let gain2 = rad.sin();
        a * gain1 + b * gain2
    }
}

#[derive(Clone)]
struct RaisedCosineMixer;
impl Mixer for RaisedCosineMixer {
    fn mix(&self, a: f32, b: f32, value: f32) -> f32 {
        let rad = PI_HALF * value;
        let gain1 = rad.cos().powi(2);
        let gain2 = 1.0 - gain1;
        a * gain1 + b * gain2
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Target {
    A,
    B,
}

#[derive(Clone, Copy)]
enum FadingState {
    Reached(Target),
    Approaching(Target),
}

impl FadingState {
    fn target(&self) -> Target {
        match self {
            Self::Reached(target) => *target,
            Self::Approaching(target) => *target,
        }
    }
}

#[derive(Clone)]
pub struct Crossfader<T: Mixer> {
    mixer: T,
    samples: usize,
    counter: usize,
    step: f32,
    value: f32,
    fading_state: FadingState,
}

impl<T: Mixer> Crossfader<T> {
    fn new(mixer: T, samples: usize) -> Self {
        Self {
            mixer,
            samples,
            counter: 0,
            step: 1.0 / samples as f32,
            value: 0.0,
            fading_state: FadingState::Reached(Target::A),
        }
    }

    fn fade_into(&mut self, target: Target) {
        let current_target = self.fading_state.target();
        if current_target == target {
            return;
        }

        self.step = -self.step;
        self.fading_state = FadingState::Approaching(target);

        match self.fading_state {
            FadingState::Reached(_) => {
                self.counter = 0;
            }
            FadingState::Approaching(_) => {
                self.counter = self.samples - self.counter;
            }
        }
    }

    fn mix(&mut self, a: f32, b: f32) -> f32 {
        match self.fading_state {
            FadingState::Reached(target) => match target {
                Target::A => a,
                Target::B => b,
            },
            FadingState::Approaching(target) => {
                self.value += self.step;
                self.counter += 1;

                if self.counter == self.samples {
                    self.fading_state = FadingState::Reached(target);
                    match target {
                        Target::A => {
                            self.value = 0.0;
                            return a;
                        }
                        Target::B => {
                            self.value = 1.0;
                            return b;
                        }
                    }
                }

                self.mixer.mix(a, b, self.value)
            }
        }
    }
}
