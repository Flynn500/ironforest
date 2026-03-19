use std::iter::Sum;
use serde::{Serialize, de::DeserializeOwned};
use num_traits::{Float as NumFloat, ToPrimitive};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Project-wide marker trait for float types usable across spatial trees,
/// decision trees, and other numeric subsystems.
///
/// Backed by `num_traits::Float` (which provides all arithmetic, constants,
/// sqrt/abs/etc. and PartialOrd) plus the bounds required for tree storage,
/// parallelism, and serialization.
///
/// Currently implemented for `f32` and `f64`. Designed so that `f16` (via
/// the `half` crate) and `f128` can be added behind feature flags without
/// structural changes.
pub trait IronFloat:
    NumFloat
    + Sum
    + Copy
    + Send
    + Sync
    + 'static
    + Serialize
    + DeserializeOwned
    + ToPrimitive
{
    /// Squared Euclidean distance between two same-length slices.
    /// Implementations may use SIMD acceleration.
    fn squared_euclidean_slice(a: &[Self], b: &[Self]) -> Self;
}

impl IronFloat for f64 {
    #[inline]
    fn squared_euclidean_slice(a: &[Self], b: &[Self]) -> Self {
        squared_euclidean(a, b)
    }
}

impl IronFloat for f32 {
    #[inline]
    fn squared_euclidean_slice(a: &[Self], b: &[Self]) -> Self {
        a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum()
    }
}

// =============================================================================
// f64 SIMD paths (moved from spatial/common.rs)
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_euclidean_single_acc(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = _mm256_setzero_pd();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm256_loadu_pd(a_ptr.add(offset));
            let vb = _mm256_loadu_pd(b_ptr.add(offset));
            let diff = _mm256_sub_pd(va, vb);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }

        let hi = _mm256_extractf128_pd(acc, 1);
        let lo = _mm256_castpd256_pd128(acc);
        let sum128 = _mm_add_pd(hi, lo);
        let upper = _mm_unpackhi_pd(sum128, sum128);
        let mut result = _mm_cvtsd_f64(_mm_add_sd(sum128, upper));

        let tail_start = chunks * 4;
        for i in 0..remainder {
            let d = a[tail_start + i] - b[tail_start + i];
            result += d * d;
        }

        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_euclidean_multi_acc(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 16;
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    unsafe {
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut acc2 = _mm256_setzero_pd();
        let mut acc3 = _mm256_setzero_pd();

        for i in 0..chunks {
            let offset = i * 16;

            let d0 = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(offset)),      _mm256_loadu_pd(b_ptr.add(offset)));
            let d1 = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(offset + 4)),  _mm256_loadu_pd(b_ptr.add(offset + 4)));
            let d2 = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(offset + 8)),  _mm256_loadu_pd(b_ptr.add(offset + 8)));
            let d3 = _mm256_sub_pd(_mm256_loadu_pd(a_ptr.add(offset + 12)), _mm256_loadu_pd(b_ptr.add(offset + 12)));

            acc0 = _mm256_fmadd_pd(d0, d0, acc0);
            acc1 = _mm256_fmadd_pd(d1, d1, acc1);
            acc2 = _mm256_fmadd_pd(d2, d2, acc2);
            acc3 = _mm256_fmadd_pd(d3, d3, acc3);
        }

        acc0 = _mm256_add_pd(acc0, acc1);
        acc2 = _mm256_add_pd(acc2, acc3);
        acc0 = _mm256_add_pd(acc0, acc2);

        let hi = _mm256_extractf128_pd(acc0, 1);
        let lo = _mm256_castpd256_pd128(acc0);
        let sum128 = _mm_add_pd(hi, lo);
        let upper = _mm_unpackhi_pd(sum128, sum128);
        let mut result = _mm_cvtsd_f64(_mm_add_sd(sum128, upper));

        let tail_start = chunks * 16;
        for i in tail_start..n {
            let d = a[i] - b[i];
            result += d * d;
        }

        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_squared_euclidean_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    unsafe {
        if n >= 32 {
            squared_euclidean_multi_acc(a, b)
        } else {
            squared_euclidean_single_acc(a, b)
        }
    }
}

/// f64-specific SIMD-accelerated squared Euclidean distance.
#[inline]
pub fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { simd_squared_euclidean_avx2_fma(a, b) };
        }
    }

    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}
