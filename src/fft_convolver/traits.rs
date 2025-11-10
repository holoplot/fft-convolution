pub trait FftBackend: Clone {
    type Complex;

    fn init(&mut self, size: usize);
    fn forward(
        &mut self,
        input: &mut [f32],
        output: &mut [Self::Complex],
    ) -> Result<(), Box<dyn std::error::Error>>;
    fn inverse(
        &mut self,
        input: &mut [Self::Complex],
        output: &mut [f32],
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait ComplexOps: Clone {
    type Complex: Clone + Default;

    fn complex_size(size: usize) -> usize;
    fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize);
    fn complex_multiply_accumulate(
        result: &mut [Self::Complex],
        a: &[Self::Complex],
        b: &[Self::Complex],
        temp_buffer: Option<&mut [Self::Complex]>,
    );
    fn sum(result: &mut [f32], a: &[f32], b: &[f32]);
    fn zero_complex(buffer: &mut [Self::Complex]);
    fn zero_real(buffer: &mut [f32]);
    fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]);
    fn add_to_buffer(dst: &mut [f32], src: &[f32]);
}
