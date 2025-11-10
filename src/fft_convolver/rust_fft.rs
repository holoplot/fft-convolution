use crate::fft_convolver::{ComplexOps, FftBackend};
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex;
use std::sync::Arc;

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

impl FftBackend for Fft {
    type Complex = Complex<f32>;

    fn init(&mut self, length: usize) {
        let mut planner = RealFftPlanner::<f32>::new();
        self.fft_forward = planner.plan_fft_forward(length);
        self.fft_inverse = planner.plan_fft_inverse(length);
    }

    fn forward(
        &mut self,
        input: &mut [f32],
        output: &mut [Self::Complex],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.fft_forward
            .process(input, output)
            .map_err(|e| e.into())
    }

    fn inverse(
        &mut self,
        input: &mut [Self::Complex],
        output: &mut [f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.fft_inverse.process(input, output)?;

        // FFT Normalization
        let len = output.len();
        output.iter_mut().for_each(|bin| *bin /= len as f32);

        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct RustComplexOps;

impl ComplexOps for RustComplexOps {
    type Complex = Complex<f32>;

    fn complex_size(size: usize) -> usize {
        (size / 2) + 1
    }

    fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
        assert!(dst.len() >= src_size);
        dst[0..src_size].clone_from_slice(&src[0..src_size]);
        dst[src_size..].iter_mut().for_each(|value| *value = 0.);
    }

    fn complex_multiply_accumulate(
        result: &mut [Self::Complex],
        a: &[Self::Complex],
        b: &[Self::Complex],
        _temp_buffer: Option<&mut [Self::Complex]>,
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

    fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
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

    fn add_to_buffer(dst: &mut [f32], src: &[f32]) {
        assert_eq!(dst.len(), src.len());
        let len = dst.len();
        let end4 = 3 * (len / 4);
        for i in (0..end4).step_by(4) {
            dst[i + 0] += src[i + 0];
            dst[i + 1] += src[i + 1];
            dst[i + 2] += src[i + 2];
            dst[i + 3] += src[i + 3];
        }
        for i in end4..len {
            dst[i] += src[i];
        }
    }

    fn zero_complex(buffer: &mut [Self::Complex]) {
        buffer.fill(Complex { re: 0.0, im: 0.0 });
    }

    fn zero_real(buffer: &mut [f32]) {
        buffer.fill(0.0);
    }

    fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]) {
        dst.clone_from_slice(src);
    }
}

pub type FFTConvolver = crate::fft_convolver::GenericFFTConvolver<Fft, RustComplexOps>;
pub type TwoStageFFTConvolver =
    crate::fft_convolver::two_stage_convolver::GenericTwoStageFFTConvolver<Fft, RustComplexOps>;

// impl TwoStageFFTConvolver {
//     pub fn with_block_sizes(
//         impulse_response: &[f32],
//         max_response_length: usize,
//         head_block_size: usize,
//         tail_block_size: usize,
//     ) -> Self {
//         TwoStageFFTConvolver::init(
//             impulse_response,
//             head_block_size,
//             max_response_length,
//             head_block_size,
//             tail_block_size,
//         )
//     }
// }
