use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;


impl NdArray<f64> {
    pub fn cholesky(&self) -> Result<NdArray<f64>, &'static str> {
        if self.ndim() != 2 {
            return Err("Cholesky decomposition requires a 2D matrix");
        }

        let n = self.shape().dims()[0];
        let m = self.shape().dims()[1];

        if n != m {
            return Err("Cholesky decomposition requires a square matrix");
        }

        let mut l = NdArray::zeros(Shape::d2(n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                for k in 0..j {
                    sum += l.get(&[i, k]).unwrap() * l.get(&[j, k]).unwrap();
                }

                if i == j {
                    let diag = self.get(&[i, i]).unwrap() - sum;
                    if diag <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    *l.get_mut(&[i, j]).unwrap() = diag.sqrt();
                } else {
                    let l_jj = *l.get(&[j, j]).unwrap();
                    *l.get_mut(&[i, j]).unwrap() = (self.get(&[i, j]).unwrap() - sum) / l_jj;
                }
            }
        }

        Ok(l)
    }


    pub fn qr(&self) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        if self.ndim() != 2 {
            return Err("QR decomposition requires a 2D matrix");
        }

        let m = self.shape().dims()[0];
        let n = self.shape().dims()[1];

        let mut r = self.clone();
        let mut q = NdArray::eye(m, None, 0);

        let k_max = if m > n { n } else { m };

        for k in 0..k_max {
            let mut x = Vec::with_capacity(m - k);
            for i in k..m {
                x.push(*r.get(&[i, k]).unwrap());
            }

            let norm_x: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-14 {
                continue;
            }

            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            let mut v = x.clone();
            v[0] += sign * norm_x;

            let norm_v: f64 = v.iter().map(|&val| val * val).sum::<f64>().sqrt();
            if norm_v < 1e-14 {
                continue;
            }
            for val in &mut v {
                *val /= norm_v;
            }

            let mut vt_r = vec![0.0; n - k];
            for j in k..n {
                let mut sum = 0.0;
                for i in 0..(m - k) {
                    sum += v[i] * r.get(&[k + i, j]).unwrap();
                }
                vt_r[j - k] = sum;
            }

            for i in 0..(m - k) {
                for j in k..n {
                    let old_val = *r.get(&[k + i, j]).unwrap();
                    *r.get_mut(&[k + i, j]).unwrap() = old_val - 2.0 * v[i] * vt_r[j - k];
                }
            }

            let mut q_v = vec![0.0; m];
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..(m - k) {
                    sum += q.get(&[i, k + j]).unwrap() * v[j];
                }
                q_v[i] = sum;
            }

            for i in 0..m {
                for j in 0..(m - k) {
                    let old_val = *q.get(&[i, k + j]).unwrap();
                    *q.get_mut(&[i, k + j]).unwrap() = old_val - 2.0 * q_v[i] * v[j];
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                if i > j {
                    *r.get_mut(&[i, j]).unwrap() = 0.0;
                }
            }
        }

        Ok((q, r))
    }

    pub fn eig(&self) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        self.eig_with_params(1000, 1e-10)
    }

    pub fn eig_with_params(&self, max_iter: usize, tol: f64) -> Result<(NdArray<f64>, NdArray<f64>), &'static str> {
        if self.ndim() != 2 {
            return Err("Eigendecomposition requires a 2D matrix");
        }

        let n = self.shape().dims()[0];
        let m = self.shape().dims()[1];

        if n != m {
            return Err("Eigendecomposition requires a square matrix");
        }

        if n == 0 {
            return Ok((
                NdArray::from_vec(Shape::d1(0), vec![]),
                NdArray::from_vec(Shape::d2(0, 0), vec![]),
            ));
        }

        if n == 1 {
            let eigenvalue = *self.get(&[0, 0]).unwrap();
            return Ok((
                NdArray::from_vec(Shape::d1(1), vec![eigenvalue]),
                NdArray::from_vec(Shape::d2(1, 1), vec![1.0]),
            ));
        }

        let mut a = self.clone();

        let mut v = NdArray::eye(n, None, 0);

        for _ in 0..max_iter {
            let (q, r) = a.qr()?;

            a = r.matmul(&q);

            v = v.matmul(&q);

            let mut off_diag_sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let val = *a.get(&[i, j]).unwrap();
                        off_diag_sum += val * val;
                    }
                }
            }

            if off_diag_sum.sqrt() < tol {
                break;
            }
        }

        let mut eigenvalues = Vec::with_capacity(n);
        for i in 0..n {
            eigenvalues.push(*a.get(&[i, i]).unwrap());
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j].abs().partial_cmp(&eigenvalues[i].abs()).unwrap()
        });

        let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

        let mut sorted_eigenvectors = Vec::with_capacity(n * n);
        for i in 0..n {
            for &idx in &indices {
                sorted_eigenvectors.push(*v.get(&[i, idx]).unwrap());
            }
        }

        Ok((
            NdArray::from_vec(Shape::d1(n), sorted_eigenvalues),
            NdArray::from_vec(Shape::d2(n, n), sorted_eigenvectors),
        ))
    }

    pub fn eigvals(&self) -> Result<NdArray<f64>, &'static str> {
        let (eigenvalues, _) = self.eig()?;
        Ok(eigenvalues)
    }
}