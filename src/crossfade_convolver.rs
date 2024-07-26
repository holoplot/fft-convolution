use crate::{Convolution, Sample};

#[derive(Clone)]
struct CrossfadeConvolverCore<T: Convolution> {
    convolver_a: T,
    convolver_b: T,
    crossfader: Crossfader<RaisedCosineMixer>,
}

#[derive(Clone)]
pub struct CrossfadeConvolver<Convolver: Convolution> {
    core: CrossfadeConvolverCore<Convolver>,
    buffer_a: Vec<Sample>,
    buffer_b: Vec<Sample>,
    stored_response: Vec<Sample>,
    response_pending: bool,
}

impl<T: Convolution> CrossfadeConvolver<T> {
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
                crossfader: Crossfader::new(RaisedCosineMixer, crossfade_samples, max_buffer_size),
            },
            buffer_a: vec![0.0; max_buffer_size],
            buffer_b: vec![0.0; max_buffer_size],
            stored_response,
            response_pending: false,
        }
    }
}

impl<Convolver: Convolution> Convolution for CrossfadeConvolver<Convolver> {
    fn init(response: &[Sample], max_block_size: usize, max_response_length: usize) -> Self {
        let convolver = Convolver::init(response, max_block_size, max_response_length);
        Self::new(convolver, response.len(), max_block_size, response.len())
    }

    fn update(&mut self, response: &[Sample]) {
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

impl<Convolver: Convolution> CrossfadeConvolver<Convolver> {
    pub fn is_crossfading(&self) -> bool {
        match self.core.crossfader.fading_state {
            FadingState::Approaching(_) => true,
            FadingState::Reached(_) => false,
        }
    }
}

fn swap<T: Convolution>(core: &mut CrossfadeConvolverCore<T>, response: &[Sample]) {
    match core.crossfader.fading_state.target() {
        Target::A => {
            core.convolver_b.update(response);
            core.crossfader.fade_into(Target::B);
        }
        Target::B => {
            core.convolver_a.update(response);
            core.crossfader.fade_into(Target::A);
        }
    }
}

#[test]
fn test_crossfade_convolver_passthrough() {
    let mut response = [0.0; 1024];
    response[0] = 1.0;
    let mut convolver = CrossfadeConvolver::new(
        crate::fft_convolver::FFTConvolver::init(&response, 1024, response.len()),
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
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample;
}

#[allow(dead_code)]
struct LinearMixer;
impl Mixer for LinearMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        a * (1.0 - value) + b * value
    }
}

#[allow(dead_code)]
struct SquareRootMixer;
impl Mixer for SquareRootMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        let gain1 = (1.0 - value).sqrt();
        let gain2 = value.sqrt();
        a * gain1 + b * gain2
    }
}
const PI_HALF: Sample = std::f32::consts::PI * 0.5;

#[allow(dead_code)]
struct CosineMixer;
impl Mixer for CosineMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
        let rad = PI_HALF * value;
        let gain1 = rad.cos();
        let gain2 = rad.sin();
        a * gain1 + b * gain2
    }
}

#[derive(Clone)]
struct RaisedCosineMixer;
impl Mixer for RaisedCosineMixer {
    fn mix(&self, a: Sample, b: Sample, value: Sample) -> Sample {
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

#[derive(Clone, Copy, PartialEq)]
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
    fading_samples: i64,
    hold_samples: i64,
    counter: i64,
    mix_value_step: Sample,
    mix_value: Sample,
    fading_state: FadingState,
}

impl<T: Mixer> Crossfader<T> {
    fn new(mixer: T, fading_samples: usize, hold_samples: usize) -> Self {
        Self {
            mixer,
            fading_samples: fading_samples as i64,
            hold_samples: hold_samples as i64,
            counter: 0,
            mix_value_step: 1.0 / fading_samples as Sample,
            mix_value: 0.0,
            fading_state: FadingState::Reached(Target::A),
        }
    }

    fn fade_into(&mut self, target: Target) {
        let current_target = self.fading_state.target();
        if current_target == target {
            return;
        }

        match self.fading_state {
            FadingState::Reached(_) => {
                self.counter = -self.hold_samples;
                self.fading_state = FadingState::Approaching(target);
                self.mix_value_step = -self.mix_value_step;
            }
            FadingState::Approaching(_) => {
                // note: should never be the case in the context of the crossfade convolver,
                // which will swap responses only after a target is reached
                if self.counter >= 0 {
                    self.counter = self.fading_samples - self.counter;
                    self.fading_state = FadingState::Approaching(target);
                    self.mix_value_step = -self.mix_value_step;
                } else {
                    self.fading_state = FadingState::Reached(target);
                }
            }
        }
    }

    fn mix(&mut self, a: Sample, b: Sample) -> Sample {
        match self.fading_state {
            FadingState::Reached(target) => match target {
                Target::A => return a,
                Target::B => return b,
            },
            FadingState::Approaching(target) => {
                self.counter += 1;

                if self.counter <= 0 {
                    // holding the previous target
                    match target {
                        Target::A => return b,
                        Target::B => return a,
                    }
                }

                self.mix_value += self.mix_value_step;

                if self.counter == self.fading_samples {
                    self.fading_state = FadingState::Reached(target);
                    match target {
                        Target::A => {
                            self.mix_value = 0.0;
                            return a;
                        }
                        Target::B => {
                            self.mix_value = 1.0;
                            return b;
                        }
                    }
                }

                self.mixer.mix(a, b, self.mix_value)
            }
        }
    }
}

#[test]
fn test_crossfader() {
    let hold_samples = 4;
    let fading_samples = 4;
    let sample_a = 1.0;
    let sample_b = 10.0;
    let mut crossfader =
        Crossfader::<RaisedCosineMixer>::new(RaisedCosineMixer, fading_samples, hold_samples);

    let start = |target: Target| match target {
        Target::A => sample_b,
        Target::B => sample_a,
    };
    let end = |target: Target| match target {
        Target::A => sample_a,
        Target::B => sample_b,
    };

    for target in [Target::B, Target::A] {
        crossfader.fade_into(target);
        for i in 0..hold_samples + fading_samples {
            let mixed_value = crossfader.mix(sample_a, sample_b);
            if i < hold_samples {
                assert!(crossfader.fading_state == FadingState::Approaching(target));
                assert_eq!(mixed_value, start(target));
            } else if i < hold_samples + fading_samples - 1 {
                assert!(crossfader.fading_state == FadingState::Approaching(target));
                assert_ne!(mixed_value, start(target));
                assert_ne!(mixed_value, end(target));
            } else {
                assert_eq!(mixed_value, end(target));
                assert!(crossfader.fading_state == FadingState::Reached(target));
            }
        }
    }
}
