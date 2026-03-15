use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;

impl NdArray<f64> {
    pub fn least_squares(&self, b: &NdArray<f64>) -> Result<NdArray<f64>, &'static str> {
        if self.ndim() != 2 {
            return Err("Matrix must be 2D for least squares");
        }
        if b.ndim() != 1 && b.ndim() != 2 {
            return Err("b must be 1D or 2D");
        }
        
        let m = self.shape().dims()[0];
        let n = self.shape().dims()[1];

        let b_rows = if b.ndim() == 1 { b.shape().dims()[0] } else { b.shape().dims()[0] };
        if b_rows != m {
            return Err("Dimension mismatch: A and b must have same number of rows");
        }

        let (q, r) = self.qr()?;
        
        //Q^T * b
        let qtb = if b.ndim() == 1 {
            let mut result = Vec::with_capacity(m);
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..m {
                    let q_val = *q.get(&[j, i]).unwrap();
                    let b_val = *b.get(&[j]).unwrap();
                    sum += q_val * b_val;
                }
                result.push(sum);
            }
            NdArray::from_vec(Shape::d1(m), result)
        } else {
            let b_cols = b.shape().dims()[1];
            let mut result = vec![0.0; m * b_cols];
            
            for i in 0..m {
                for col in 0..b_cols {
                    let mut sum = 0.0;
                    for j in 0..m {
                        let q_val = *q.get(&[j, i]).unwrap();
                        let b_val = *b.get(&[j, col]).unwrap();
                        sum += q_val * b_val;
                    }
                    result[i * b_cols + col] = sum;
                }
            }
            NdArray::from_vec(Shape::d2(m, b_cols), result)
        };
        
        //R * x = qtb
        if b.ndim() == 1 {
            let mut x = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = *qtb.get(&[i]).unwrap();
                for j in (i + 1)..n {
                    sum -= r.get(&[i, j]).unwrap() * x[j];
                }
                let r_ii = *r.get(&[i, i]).unwrap();
                if r_ii.abs() < 1e-14 {
                    return Err("Matrix is rank deficient");
                }
                x[i] = sum / r_ii;
            }
            Ok(NdArray::from_vec(Shape::d1(n), x))
        } else {
            let b_cols = b.shape().dims()[1];
            let mut x = vec![0.0; n * b_cols];
            for col in 0..b_cols {
                for i in (0..n).rev() {
                    let mut sum = *qtb.get(&[i, col]).unwrap();
                    for j in (i + 1)..n {
                        sum -= r.get(&[i, j]).unwrap() * x[j * b_cols + col];
                    }
                    let r_ii = *r.get(&[i, i]).unwrap();
                    if r_ii.abs() < 1e-14 {
                        return Err("Matrix is rank deficient");
                    }
                    x[i * b_cols + col] = sum / r_ii;
                }
            }
            Ok(NdArray::from_vec(Shape::d2(n, b_cols), x))
        }
    }

pub fn weighted_least_squares(&self, b: &NdArray<f64>, weights: &NdArray<f64>,) -> Result<NdArray<f64>, &'static str> {
        if weights.ndim() != 1 {
            return Err("Weights must be 1D");
        }
        
        let m = self.shape().dims()[0];
        let sqrt_w: Vec<f64> = weights.as_slice_unchecked()
            .iter()
            .map(|&w| {
                if w < 0.0 {
                    return Err("Weights must be non-negative");
                }
                Ok(w.sqrt())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let sqrt_w_col = NdArray::from_vec(Shape::d2(m, 1), sqrt_w);
        let weighted_a = self * &sqrt_w_col;
        let weighted_b = b * &sqrt_w_col;
        
        weighted_a.least_squares(&weighted_b)
    }
}
