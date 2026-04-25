use crate::{Generator, IronFloat, NdArray, Shape};



impl<T: IronFloat> NdArray<T> {
    pub fn sample_rows(&self, k: usize, rng: &mut Generator) -> NdArray<T> {
        assert_eq!(self.ndim(), 2, "sample_rows() requires a 2D array");
        let dims = self.shape().dims();
        let n = dims[0];
        let d = dims[1];

        if k >= n {
            return self.clone();
        }

        let mut out = Vec::with_capacity(k * d);
        for _ in 0..k {
            let idx = rng.next_usize() % n;
            let row = self.row(idx);
            out.extend_from_slice(row);
        }

        NdArray::from_vec(Shape::new(vec![k, d]), out)
    }

    pub fn covariance(&self) -> NdArray<f64> {
        let mut sample = self.clone();
        sample.covariance_mut()
    }

    pub fn covariance_mut(&mut self) -> NdArray<f64> {
        assert_eq!(self.ndim(), 2, "covariance requires a 2D array");
        let dims = self.shape().dims();
        let n = dims[0];
        let d = dims[1];
        assert!(n >= 2, "covariance requires at least 2 samples");

        let means = self.column_means();
        let means_slice = means.as_slice_unchecked();

        let mut buf = vec![T::zero(); d];
        for i in 0..n {
            let row = self.row(i);
            for j in 0..d {
                let centred = row[j].to_f64().unwrap_or(0.0) - means_slice[j];
                buf[j] = T::from(centred).unwrap_or_else(T::zero);
            }
            self.set_row(i, &buf);
        }

        let xt = self.transpose();
        let gram = xt.matmul(self);

        let mut cov = gram.as_f64();
        let inv = 1.0 / (n - 1) as f64;
        for v in cov.as_mut_slice().expect("cov is owned").iter_mut() {
            *v *= inv;
        }
        cov
    }


    pub fn intrinsic_dim(
        &self,
        variance_threshold: f64,
        sample_cap: usize,
        rng: &mut Generator,
    ) -> Result<usize, &'static str> {
        assert_eq!(self.ndim(), 2, "intrinsic_dim requires a 2D array");
        let dims = self.shape().dims();
        let n = dims[0];
        let d = dims[1];

        if n < 2 {
            return Err("intrinsic_dim requires at least 2 samples");
        }
        if d == 0 {
            return Err("intrinsic_dim requires at least 1 feature");
        }

        let mut sample = self.sample_rows(sample_cap, rng);

        let cov = sample.covariance_mut();

        let (eigvals, _) = cov.eig()?;

        let mut vals: Vec<f64> = eigvals
            .as_slice_unchecked()
            .iter()
            .map(|&v| v.max(0.0))
            .collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let total: f64 = vals.iter().sum();
        if total <= 0.0 {
            return Ok(0);
        }

        let target = variance_threshold * total;
        let mut cumsum = 0.0;
        for (i, &v) in vals.iter().enumerate() {
            cumsum += v;
            if cumsum >= target {
                return Ok(i + 1);
            }
        }

        Ok(d)
    }
}