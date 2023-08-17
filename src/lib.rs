pub mod crossfade_convolver;
pub mod fft_convolver;

// todo: use a generic floating point type
pub type Sample = f32;

pub trait Convolution: Clone {
    fn init(response: &[Sample], max_block_size: usize) -> Self;

    // must be implemented in a real-time safe way, e.g. no heap allocations
    fn update(&mut self, response: &[Sample]);

    fn process(&mut self, input: &[Sample], output: &mut [Sample]);
}
