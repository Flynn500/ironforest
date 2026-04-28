use crate::IronFloat;
use crate::array::ndarray::{NdArray};
use crate::array::shape::Shape;
impl NdArray<f64> {
    pub fn sum(&self) -> f64 {
        self.as_slice_unchecked().iter().sum()
    }

    pub fn any(&self) -> bool {
        self.as_slice_unchecked().iter().any(|&x| x != 0.0)
    }

    pub fn all(&self) -> bool {
        self.as_slice_unchecked().iter().all(|&x| x != 0.0)
    }

    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.sum() / self.len() as f64
    }

    pub fn var(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        let mean = self.mean();
        let sum_sq_dev: f64 = self.as_slice_unchecked()
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        sum_sq_dev / self.len() as f64
    }

    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    pub fn median(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        let mut sorted = self.to_contiguous().as_slice_unchecked().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }

    pub fn quantile(&self, q: f64) -> f64 {
        assert!(q >= 0.0 && q <= 1.0, "Quantile must be between 0 and 1");
        if self.is_empty() {
            return f64::NAN;
        }
        let mut sorted = self.to_contiguous().as_slice_unchecked().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pos = q * (sorted.len() - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        if lower == upper {
            sorted[lower]
        } else {
            let weight = pos - lower as f64;
            sorted[lower] * (1.0 - weight) + sorted[upper] * weight
        }
    }

    pub fn quantiles(&self, qs: &[f64]) -> NdArray<f64> {
        for &q in qs {
            assert!(q >= 0.0 && q <= 1.0, "Quantile must be between 0 and 1");
        }
        if self.is_empty() {
            return NdArray::from_vec(Shape::d1(qs.len()), vec![f64::NAN; qs.len()]);
        }
        let mut sorted = self.to_contiguous().as_slice_unchecked().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let results: Vec<f64> = qs.iter().map(|&q| {
            let pos = q * (n - 1) as f64;
            let lower = pos.floor() as usize;
            let upper = pos.ceil() as usize;
            if lower == upper {
                sorted[lower]
            } else {
                let weight = pos - lower as f64;
                sorted[lower] * (1.0 - weight) + sorted[upper] * weight
            }
        }).collect();
        NdArray::from_vec(Shape::d1(results.len()), results)
    }

    pub fn max(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.as_slice_unchecked()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.as_slice_unchecked()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }
}

impl<T> NdArray<T> { 
    pub fn mode(&self) -> T where T: PartialEq + Copy {
        assert!(!self.is_empty(), "mode() called on empty array");

        let slice = self.as_slice_unchecked();
        let mut best_val = slice[0];
        let mut best_count = 0usize;

        for &val in slice {
            let mut count = 0usize;
            for &other in slice {
                if val == other {
                    count += 1;
                }
            }
            if count > best_count {
                best_count = count;
                best_val = val;
            }
        }

        best_val
    }
}

impl<T: IronFloat> NdArray<T> {
    pub fn column_means(&self) -> NdArray<f64> {
        assert_eq!(self.ndim(), 2, "column_means() requires a 2D array");
        let dims = self.shape().dims();
        let n = dims[0];
        let d = dims[1];

        let mut sums = vec![0.0_f64; d];
        for i in 0..n {
            let row = self.row(i);
            for (j, &v) in row.iter().enumerate() {
                sums[j] += v.to_f64().unwrap_or(0.0);
            }
        }

        let inv_n = 1.0 / n as f64;
        for s in sums.iter_mut() {
            *s *= inv_n;
        }

        NdArray::from_vec(Shape::new(vec![d]), sums)
    }


}