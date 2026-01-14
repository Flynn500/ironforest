use crate::array::ndarray::{self, NdArray};

impl NdArray<f64> {
    pub fn sum(&self) -> f64 {
        self.as_slice().iter().sum()
    }

    pub fn any(&self) -> bool {
        self.as_slice().iter().any(|&x| x != 0.0)
    }

    pub fn all(&self) -> bool {
        self.as_slice().iter().all(|&x| x != 0.0)
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
        let sum_sq_dev: f64 = self.as_slice()
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
        let mut sorted = self.as_slice().to_vec();
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
        let mut sorted = self.as_slice().to_vec();
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

    fn _pearson(x: &NdArray<f64>, y: &NdArray<f64>) -> f64 {
        assert_eq!(x.len(), y.len(), "Arrays must have same length");
        assert_eq!(x.ndim(), 1, "Array must be 1d");
        assert_eq!(y.ndim(), 1, "Array must be 1d");
        let n = x.len() as f64;

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;
        
        let mut numerator: f64 = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len(){
            let x_diff = x.get(&[i]).unwrap() - mean_x;
            let y_diff = y.get(&[i]).unwrap() - mean_y;
            numerator += x_diff * y_diff;
            sum_sq_x += x_diff.powi(2);
            sum_sq_y += y_diff.powi(2);
        }

        let denom_x = sum_sq_x.powf(0.5);
        let denom_y = sum_sq_y.powf(0.5);

        if denom_x == 0.0 || denom_y == 0.0 {
            return 0.0
        }

        numerator / (denom_x * denom_y)
    }

    pub fn pearson(&self, other: &NdArray<f64>) -> f64 {
        Self::_pearson(self, other)
    }

    fn ranks(&self) -> NdArray<f64> {
        assert_eq!(self.ndim(), 1, "Array must be 1d");
        let n = self.len();

        let mut indices: Vec<usize> = (0..n).collect();

        indices.sort_by(|&a, &b| {
            self.get(&[a]).unwrap()
                .partial_cmp(self.get(&[b]).unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut rank_values = vec![0.0; n];
        let mut i = 0;

        while i < n {
            let mut j = i;

            while j < n && self.get(&[indices[j]]).unwrap() == self.get(&[indices[i]]).unwrap() {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                rank_values[indices[k]] = avg_rank;
            }
            i = j;
        }

        NdArray::from_vec(self.shape().clone(), rank_values)
    }

    pub fn spearman(&self, other: &NdArray<f64>) -> f64 {
        Self::pearson(&self.ranks(), &other.ranks())
    }
}