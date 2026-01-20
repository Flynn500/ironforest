use std::{cmp::Ordering};

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