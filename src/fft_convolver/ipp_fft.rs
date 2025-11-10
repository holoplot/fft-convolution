use crate::fft_convolver::{ComplexOps, FftBackend};
use ipp_sys::*;
use num_complex::Complex32;

pub struct Fft {
    size: usize,
    spec: Vec<u8>,
    scratch_buffer: Vec<u8>,
}

impl Default for Fft {
    fn default() -> Self {
        Self {
            size: 0,
            spec: Vec::new(),
            scratch_buffer: Vec::new(),
        }
    }
}

impl std::fmt::Debug for Fft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IppFft(size: {})", self.size)
    }
}

impl Clone for Fft {
    fn clone(&self) -> Self {
        let mut new_fft = Fft::default();
        if self.size > 0 {
            new_fft.init(self.size);
        }
        new_fft
    }
}

impl FftBackend for Fft {
    type Complex = Complex32;

    fn init(&mut self, size: usize) {
        assert!(size > 0, "FFT size must be greater than 0");
        assert!(size.is_power_of_two(), "FFT size must be a power of 2");

        let mut spec_size = 0;
        let mut init_size = 0;
        let mut work_buf_size = 0;

        unsafe {
            ippsDFTGetSize_R_32f(
                size as i32,
                8, // IPP_FFT_NODIV_BY_ANY
                0, // No special hint
                &mut spec_size,
                &mut init_size,
                &mut work_buf_size,
            );
        }

        let mut init_buffer = vec![0; init_size as usize];
        let scratch_buffer = vec![0; work_buf_size as usize];
        let mut spec = vec![0; spec_size as usize];

        unsafe {
            ippsDFTInit_R_32f(
                size as i32,
                8, // IPP_FFT_NODIV_BY_ANY
                0, // No special hint
                spec.as_mut_ptr() as *mut ipp_sys::IppsDFTSpec_R_32f,
                init_buffer.as_mut_ptr(),
            );
        }

        self.size = size;
        self.spec = spec;
        self.scratch_buffer = scratch_buffer;
    }

    fn forward(
        &mut self,
        input: &mut [f32],
        output: &mut [Self::Complex],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.size == 0 {
            return Err("FFT not initialized".into());
        }

        assert_eq!(input.len(), self.size, "Input length must match FFT size");
        assert_eq!(
            output.len(),
            self.size / 2 + 1,
            "Output length must be size/2 + 1"
        );

        unsafe {
            ippsDFTFwd_RToCCS_32f(
                input.as_ptr(),
                output.as_mut_ptr() as *mut Ipp32f, // the API takes the pointer to the first float member of the complex array
                self.spec.as_ptr() as *const DFTSpec_R_32f,
                self.scratch_buffer.as_mut_ptr(),
            );
        }

        Ok(())
    }

    fn inverse(
        &mut self,
        input: &mut [Self::Complex],
        output: &mut [f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.size == 0 {
            return Err("FFT not initialized".into());
        }

        assert_eq!(
            input.len(),
            self.size / 2 + 1,
            "Input length must be size/2 + 1"
        );
        assert_eq!(output.len(), self.size, "Output length must match FFT size");

        unsafe {
            ippsDFTInv_CCSToR_32f(
                input.as_ptr() as *const Ipp32f, // the API takes the pointer to the first float member of the complex array
                output.as_mut_ptr(),
                self.spec.as_ptr() as *const DFTSpec_R_32f,
                self.scratch_buffer.as_mut_ptr(),
            );
        }

        // Normalization
        unsafe {
            ippsDivC_32f_I(self.size as f32, output.as_mut_ptr(), self.size as i32);
        }

        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct IppComplexOps;

impl ComplexOps for IppComplexOps {
    type Complex = Complex32;
    fn complex_size(size: usize) -> usize {
        (size / 2) + 1
    }

    fn copy_and_pad(dst: &mut [f32], src: &[f32], src_size: usize) {
        assert!(dst.len() >= src_size, "Destination buffer too small");

        unsafe {
            ippsCopy_32f(src.as_ptr(), dst.as_mut_ptr(), src_size as i32);

            if dst.len() > src_size {
                ippsZero_32f(
                    dst.as_mut_ptr().add(src_size),
                    (dst.len() - src_size) as i32,
                );
            }
        }
    }

    fn complex_multiply_accumulate(
        result: &mut [Self::Complex],
        a: &[Self::Complex],
        b: &[Self::Complex],
        temp_buffer: Option<&mut [Self::Complex]>,
    ) {
        let temp_buffer = temp_buffer.expect("IPP implementation requires a temp buffer");
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());
        assert_eq!(temp_buffer.len(), a.len());

        unsafe {
            let len = result.len() as i32;

            ippsMul_32fc(
                a.as_ptr() as *const ipp_sys::Ipp32fc,
                b.as_ptr() as *const ipp_sys::Ipp32fc,
                temp_buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                len,
            );

            ippsAdd_32fc(
                temp_buffer.as_ptr() as *const ipp_sys::Ipp32fc,
                result.as_ptr() as *const ipp_sys::Ipp32fc,
                result.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                len,
            );
        }
    }

    fn sum(result: &mut [f32], a: &[f32], b: &[f32]) {
        assert_eq!(result.len(), a.len());
        assert_eq!(result.len(), b.len());

        unsafe {
            ippsAdd_32f(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                result.len() as i32,
            );
        }
    }

    fn zero_complex(buffer: &mut [Self::Complex]) {
        unsafe {
            ippsZero_32fc(
                buffer.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                buffer.len() as i32,
            );
        }
    }

    fn zero_real(buffer: &mut [f32]) {
        unsafe {
            ippsZero_32f(buffer.as_mut_ptr(), buffer.len() as i32);
        }
    }

    fn copy_complex(dst: &mut [Self::Complex], src: &[Self::Complex]) {
        unsafe {
            ippsCopy_32fc(
                src.as_ptr() as *const ipp_sys::Ipp32fc,
                dst.as_mut_ptr() as *mut ipp_sys::Ipp32fc,
                dst.len() as i32,
            );
        }
    }

    fn add_to_buffer(dst: &mut [f32], src: &[f32]) {
        unsafe {
            ippsAdd_32f_I(src.as_ptr(), dst.as_mut_ptr(), dst.len() as i32);
        }
    }
}

pub type FFTConvolver = crate::fft_convolver::GenericFFTConvolver<Fft, IppComplexOps>;
pub type TwoStageFFTConvolver =
    crate::fft_convolver::two_stage_convolver::GenericTwoStageFFTConvolver<Fft, IppComplexOps>;
