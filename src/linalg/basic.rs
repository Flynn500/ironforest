use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_dot_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = _mm256_setzero_pd();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let va = _mm256_loadu_pd(a_ptr.add(offset));
            let vb = _mm256_loadu_pd(b_ptr.add(offset));
        
            acc = _mm256_fmadd_pd(va, vb, acc);
        }
    }

    let hi = _mm256_extractf128_pd(acc, 1);
    let lo = _mm256_castpd256_pd128(acc);
    let sum128 = _mm_add_pd(hi, lo);
    let upper = _mm_unpackhi_pd(sum128, sum128);
    let mut result = _mm_cvtsd_f64(_mm_add_sd(sum128, upper));

    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}


#[inline]
pub fn simd_dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_dot_avx2_fma(a, b) };
        }
    }

    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl<T: Copy> NdArray<T> {
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose requires 2D array");
        let (n, m) = (self.shape().dims()[0], self.shape().dims()[1]);
        let mut data = Vec::with_capacity(n * m);
        for j in 0..m {
            for i in 0..n {
                data.push(*self.get(&[i, j]).unwrap());
            }
        }
        NdArray::from_vec(Shape::d2(m, n), data)
    }

    pub fn t(&self) -> Self {
        self.transpose()
    }

    pub fn ravel(&self) -> Self {
        NdArray::from_vec(Shape::d1(self.len()), self.as_slice().to_vec())
    }
}

impl NdArray<f64> {
    pub fn diagonal(&self, k: isize) -> Self {
        assert_eq!(self.ndim(), 2, "Input must be 2D");
        let (n, m) = (self.shape().dims()[0], self.shape().dims()[1]);

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        if row_start >= n || col_start >= m {
            return NdArray::from_vec(Shape::d1(0), vec![]);
        }

        let diag_len = (n - row_start).min(m - col_start);
        let mut data = Vec::with_capacity(diag_len);

        for i in 0..diag_len {
            data.push(*self.get(&[row_start + i, col_start + i]).unwrap());
        }

        NdArray::from_vec(Shape::d1(diag_len), data)
    }
    
    pub fn outer(a: &NdArray<f64>, b: &NdArray<f64>) -> Self {
        assert_eq!(a.ndim(), 1, "First input must be 1D");
        assert_eq!(b.ndim(), 1, "Second input must be 1D");

        let m = a.len();
        let n = b.len();
        let mut data = Vec::with_capacity(m * n);

        for &ai in a.as_slice() {
            for &bi in b.as_slice() {
                data.push(ai * bi);
            }
        }

        NdArray::from_vec(Shape::d2(m, n), data)
    }

    pub fn matmul(&self, other: &NdArray<f64>) -> Self {
        match (self.ndim(), other.ndim()) {
            (1, 1) => {
                assert_eq!(self.len(), other.len(), "Vectors must have same length");
                let sum: f64 = self.as_slice().iter()
                    .zip(other.as_slice().iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                NdArray::from_vec(Shape::d1(1), vec![sum])
            }
            (2, 1) => {
                let (n, k) = (self.shape().dims()[0], self.shape().dims()[1]);
                assert_eq!(k, other.len(), "Inner dimensions must match");
                let mut data = Vec::with_capacity(n);
                for i in 0..n {
                    let row = &self.as_slice()[i * k..(i + 1) * k];
                    let sum = simd_dot(row, other.as_slice());
                    data.push(sum);
                }
                NdArray::from_vec(Shape::d1(n), data)
            }
            (1, 2) => {
                let (k, m) = (other.shape().dims()[0], other.shape().dims()[1]);
                assert_eq!(self.len(), k, "Inner dimensions must match");
                let mut data = Vec::with_capacity(m);
                for j in 0..m {
                    let sum: f64 = (0..k)
                        .map(|i| self.get(&[i]).unwrap() * other.get(&[i, j]).unwrap())
                        .sum();
                    data.push(sum);
                }
                NdArray::from_vec(Shape::d1(m), data)
            }
            (2, 2) => {
                let (n, k1) = (self.shape().dims()[0], self.shape().dims()[1]);
                let (k2, m) = (other.shape().dims()[0], other.shape().dims()[1]);
                assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);
                let mut data = Vec::with_capacity(n * m);
                for i in 0..n {
                    for j in 0..m {
                        let sum: f64 = (0..k1)
                            .map(|k| self.get(&[i, k]).unwrap() * other.get(&[k, j]).unwrap())
                            .sum();
                        data.push(sum);
                    }
                }
                NdArray::from_vec(Shape::d2(n, m), data)
            }
            _ => panic!("matmul requires 1D or 2D arrays"),
        }
    }

    pub fn dot(&self, other: &NdArray<f64>) -> Self {
        self.matmul(other)
    }
}
