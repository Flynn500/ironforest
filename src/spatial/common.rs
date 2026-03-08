use std::{cmp::Ordering};
use crate::stats::special::gamma;
use serde::{Deserialize, Serialize};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::borrow::Cow;

//normalzie for cosine
#[inline]
fn normalize(a: &[f64]) -> Vec<f64> {
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        a.to_vec()
    } else {
        a.iter().map(|x| x / norm).collect()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine
}

impl DistanceMetric {
    //function applied to data before tree construnction (just normalizing  for cosine currently)
    #[inline]
    pub fn pre_transform<'a>(self, a: &'a [f64]) -> Cow<'a, [f64]> {
        match self {
            DistanceMetric::Cosine => Cow::Owned(normalize(a)),
            _ => Cow::Borrowed(a),
        }
    }

    //pre transform for radius queries as caller gives distance directly
    #[inline]
    pub fn pre_transform_radius(self, radius: f64) -> f64 {
        match self {
            DistanceMetric::Euclidean => radius * radius,
            _ => radius,
        }
    }


    //reduced distance function for in tree calculations
    #[inline]
    pub fn reduced_distance(self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => squared_euclidean(a, b),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
            DistanceMetric::Cosine => squared_euclidean(a, b),
        }
    }

    //distance function for in tree calculations
    #[inline]
    pub fn distance(self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => euclidean(a, b),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
            DistanceMetric::Cosine => euclidean(a, b),
        }
    }

    //transforms data from reduced function values to true distance values
    #[inline]
    pub fn post_transform(self, dist: f64) -> f64 {
        match self {
            DistanceMetric::Euclidean => dist.sqrt(),
            DistanceMetric::Cosine => dist / 2.0,
            _ => dist,
        }
    }
}


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

#[inline]
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean(a, b).sqrt()
}

#[inline]
fn manhattan(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[inline]
fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    let mut m: f64 = 0.0;
    for i in 0..a.len() {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Uniform,
    Triangular,
}

impl KernelType {
    pub fn evaluate(&self, dist: f64, h: f64) -> f64 {
        let u = dist / h;
        match self {
            KernelType::Gaussian => (-0.5 * u * u).exp(),
            KernelType::Epanechnikov => {
                if u < 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
            }
            KernelType::Uniform => {
                if u < 1.0 { 0.5 } else { 0.0 }
            }
            KernelType::Triangular => {
                if u < 1.0 { 1.0 - u } else { 0.0 }
            }
        }
    }

    pub fn normalization_constant(&self, dim: usize) -> f64 {
        let d = dim as f64;
        let unit_ball = std::f64::consts::PI.powf(d / 2.0) / gamma(d / 2.0 + 1.0);
        match self {
            KernelType::Gaussian => (2.0 * std::f64::consts::PI).powf(d / 2.0),
            KernelType::Uniform => unit_ball,
            KernelType::Epanechnikov => unit_ball * 2.0 / (d + 2.0),
            KernelType::Triangular => unit_ball * 1.0 / (d + 1.0),
        }
    }

    pub fn evaluate_second_derivative(&self, r: f64, h: f64) -> f64 {
        let u = r / h;
        match self {
            KernelType::Gaussian => {
                let k = (-0.5 * u * u).exp();
                (u * u / (h * h) - 1.0 / (h * h)) * k
            }

            KernelType::Epanechnikov => if u < 1.0 { -1.5 / (h * h) } else { 0.0 },
            KernelType::Triangular => 0.0,
            KernelType::Uniform => 0.0,
        }
    }

    pub fn third_derivative(&self, r: f64, h: f64) -> f64 {
        let u = r / h;
        let h3 = h * h * h;
        match self {
            KernelType::Gaussian => {
                let k = (-0.5 * u * u).exp();
                (3.0 * u - u * u * u) / h3 * k
            }
            _ => 0.0,
        }
    }

    pub fn fourth_derivative(&self, r: f64, h: f64) -> f64 {
        let u = r / h;
        let h4 = h.powi(4);
        match self {
            KernelType::Gaussian => {
                let k = (-0.5 * u * u).exp();
                (u.powi(4) - 6.0 * u * u + 3.0) * k / h4
            }
            _ => 0.0,
        }
    }

    pub fn node_error_bound(&self, n: f64, radius: f64, h: f64) -> f64 {
        match self {
            KernelType::Gaussian => {
                (n / 120.0) * 2.5221 * radius.powi(5) / h.powi(5)
            }
            KernelType::Epanechnikov => {
                n * (radius / h) * 0.75
            }
            KernelType::Uniform => {
                n * (radius / h) * 0.5
            }
            KernelType::Triangular => {
                n * (radius / h) * 1.0
            }
        }
    }
}

pub struct HeapItem {
        pub distance: f64,
        pub index: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}
