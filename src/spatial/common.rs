use std::cmp::Ordering;
use crate::stats::special::gamma;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
pub use crate::iron_float::IronFloat;

// =============================================================================
// Cosine normalization (generic)
// =============================================================================

fn normalize<T: IronFloat>(a: &[T]) -> Vec<T> {
    let norm: T = a.iter().map(|x| *x * *x).sum::<T>();
    let norm = norm.sqrt();
    if norm == T::zero() {
        a.to_vec()
    } else {
        a.iter().map(|x| *x / norm).collect()
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
    /// Applied to data before tree construction (normalises for Cosine).
    #[inline]
    pub fn pre_transform<'a, T: IronFloat>(self, a: &'a [T]) -> Cow<'a, [T]> {
        match self {
            DistanceMetric::Cosine => Cow::Owned(normalize(a)),
            _ => Cow::Borrowed(a),
        }
    }

    /// Convert a true radius to its reduced-space equivalent.
    #[inline]
    pub fn to_reduced<T: IronFloat>(self, radius: T) -> T {
        match self {
            DistanceMetric::Euclidean => radius * radius,
            _ => radius,
        }
    }

    /// Reduced (cheaper) distance used for in-tree comparisons.
    #[inline]
    pub fn reduced_distance<T: IronFloat>(self, a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => T::squared_euclidean_slice(a, b),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
            DistanceMetric::Cosine => T::squared_euclidean_slice(a, b),
        }
    }

    /// True distance (used for output and non-reduced tree traversal).
    #[inline]
    pub fn distance<T: IronFloat>(self, a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => self.reduced_distance(a, b).sqrt(),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
            DistanceMetric::Cosine => self.reduced_distance(a, b).sqrt(),
        }
    }

    /// Convert a reduced-space distance back to the true distance.
    #[inline]
    pub fn post_transform<T: IronFloat>(self, dist: T) -> T {
        match self {
            DistanceMetric::Euclidean => dist.sqrt(),
            DistanceMetric::Cosine => dist / (T::one() + T::one()),
            _ => dist,
        }
    }
}


#[inline]
fn manhattan<T: IronFloat>(a: &[T], b: &[T]) -> T {
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + (a[i] - b[i]).abs();
    }
    sum
}

#[inline]
fn chebyshev<T: IronFloat>(a: &[T], b: &[T]) -> T {
    let mut m = T::zero();
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
    pub fn evaluate<T: IronFloat>(&self, dist: T, h: T) -> T {
        let u = dist / h;
        let half = T::from(0.5).unwrap();
        let one = T::one();
        let zero = T::zero();
        match self {
            KernelType::Gaussian => (T::from(-0.5).unwrap() * u * u).exp(),
            KernelType::Epanechnikov => {
                if u < one { T::from(0.75).unwrap() * (one - u * u) } else { zero }
            }
            KernelType::Uniform => {
                if u < one { half } else { zero }
            }
            KernelType::Triangular => {
                if u < one { one - u } else { zero }
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

    pub fn evaluate_second_derivative<T: IronFloat>(&self, r: T, h: T) -> T {
        let u = r / h;
        let one = T::one();
        let zero = T::zero();
        match self {
            KernelType::Gaussian => {
                let k = (T::from(-0.5).unwrap() * u * u).exp();
                (u * u / (h * h) - one / (h * h)) * k
            }
            KernelType::Epanechnikov => if u < one { T::from(-1.5).unwrap() / (h * h) } else { zero },
            KernelType::Triangular => zero,
            KernelType::Uniform => zero,
        }
    }

    pub fn third_derivative<T: IronFloat>(&self, r: T, h: T) -> T {
        let u = r / h;
        let h3 = h * h * h;
        let zero = T::zero();
        match self {
            KernelType::Gaussian => {
                let k = (T::from(-0.5).unwrap() * u * u).exp();
                (T::from(3.0).unwrap() * u - u * u * u) / h3 * k
            }
            _ => zero,
        }
    }

    pub fn fourth_derivative<T: IronFloat>(&self, r: T, h: T) -> T {
        let u = r / h;
        let h4 = h * h * h * h;
        let zero = T::zero();
        match self {
            KernelType::Gaussian => {
                let k = (T::from(-0.5).unwrap() * u * u).exp();
                let u2 = u * u;
                (u2 * u2 - T::from(6.0).unwrap() * u2 + T::from(3.0).unwrap()) * k / h4
            }
            _ => zero,
        }
    }

    pub fn node_error_bound<T: IronFloat>(&self, n: T, radius: T, h: T) -> T {
        let ratio = radius / h;
        match self {
            KernelType::Gaussian => {
                let r5 = ratio * ratio * ratio * ratio * ratio;
                (n / T::from(120.0).unwrap()) * T::from(2.5221).unwrap() * r5
            }
            KernelType::Epanechnikov => {
                n * ratio * T::from(0.75).unwrap()
            }
            KernelType::Uniform => {
                n * ratio * T::from(0.5).unwrap()
            }
            KernelType::Triangular => {
                n * ratio
            }
        }
    }
}

pub struct HeapItem<T: IronFloat> {
    pub distance: T,
    pub index: usize,
}

impl<T: IronFloat> PartialEq for HeapItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<T: IronFloat> Eq for HeapItem<T> {}

impl<T: IronFloat> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: IronFloat> Ord for HeapItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}
