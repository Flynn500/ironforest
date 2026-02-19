use std::{cmp::Ordering};
use crate::stats::special::gamma;

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl DistanceMetric {
    #[inline]
    pub fn distance(self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => euclidean(a, b),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
        }
    }
}


#[derive(Clone, Copy, Debug)]
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
                (n / 120.0) * 7.0 * radius.powi(5) / h.powi(5)
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

#[inline]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
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