use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;

impl NdArray<f64> {
    pub fn ones(shape: Shape) -> Self {
        Self::filled(shape, 1.0)
    }

    pub fn full(shape: Shape, value: f64) -> Self {
        Self::filled(shape, value)
    }
    
    pub fn eye(n: usize, m: Option<usize>, k: isize) -> Self {
        let m = m.unwrap_or(n);
        let mut arr = NdArray::zeros(Shape::d2(n, m));

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        let mut row = row_start;
        let mut col = col_start;
        while row < n && col < m {
            *arr.get_mut(&[row, col]).unwrap() = 1.0;
            row += 1;
            col += 1;
        }
        arr
    }

    pub fn from_diag(v: &NdArray<f64>, k: isize) -> Self {
        assert_eq!(v.ndim(), 1, "Input must be 1D");
        let n = v.len();
        let size = n + k.unsigned_abs();
        let mut arr = NdArray::zeros(Shape::d2(size, size));

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        for (i, &val) in v.as_slice().iter().enumerate() {
            *arr.get_mut(&[row_start + i, col_start + i]).unwrap() = val;
        }
        arr
    }

    pub fn column_stack(arrays: &[&NdArray<f64>]) -> Self {
        assert!(!arrays.is_empty(), "Need at least one array");

        let mut arrays_2d: Vec<NdArray<f64>> = Vec::new();
        for arr in arrays {
            if arr.ndim() == 1 {
                let n = arr.len();
                arrays_2d.push(NdArray::from_vec(Shape::d2(n, 1), arr.as_slice().to_vec()));
            } else if arr.ndim() == 2 {
                arrays_2d.push((*arr).clone());
            } else {
                panic!("column_stack only supports 1D and 2D arrays");
            }
        }

        let nrows = arrays_2d[0].shape().dims()[0];
        for arr in &arrays_2d {
            assert_eq!(
                arr.shape().dims()[0], nrows,
                "All arrays must have the same number of rows"
            );
        }

        let total_cols: usize = arrays_2d.iter()
            .map(|arr| arr.shape().dims()[1])
            .sum();

        let mut result_data = Vec::with_capacity(nrows * total_cols);

        for row in 0..nrows {
            for arr in &arrays_2d {
                let ncols = arr.shape().dims()[1];
                for col in 0..ncols {
                    result_data.push(*arr.get(&[row, col]).unwrap());
                }
            }
        }

        NdArray::from_vec(Shape::d2(nrows, total_cols), result_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_stack_1d_arrays() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![4.0, 5.0, 6.0]);
        let result = NdArray::column_stack(&[&a, &b]);

        assert_eq!(result.shape().dims(), &[3, 2]);
        // [[1, 4], [2, 5], [3, 6]] in row-major order
        assert_eq!(result.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn column_stack_2d_arrays() {
        let a = NdArray::from_vec(Shape::d2(3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d2(3, 1), vec![7.0, 8.0, 9.0]);
        let result = NdArray::column_stack(&[&a, &b]);

        assert_eq!(result.shape().dims(), &[3, 3]);
        // [[1, 2, 7], [3, 4, 8], [5, 6, 9]] in row-major order
        assert_eq!(result.as_slice(), &[1.0, 2.0, 7.0, 3.0, 4.0, 8.0, 5.0, 6.0, 9.0]);
    }

    #[test]
    fn column_stack_mixed_1d_2d() {
        let a = NdArray::from_vec(Shape::d1(3), vec![10.0, 11.0, 12.0]);
        let b = NdArray::from_vec(Shape::d2(3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = NdArray::column_stack(&[&a, &b]);

        assert_eq!(result.shape().dims(), &[3, 3]);
        // [[10, 1, 2], [11, 3, 4], [12, 5, 6]] in row-major order
        assert_eq!(result.as_slice(), &[10.0, 1.0, 2.0, 11.0, 3.0, 4.0, 12.0, 5.0, 6.0]);
    }

    #[test]
    fn column_stack_multiple_arrays() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![3.0, 4.0]);
        let c = NdArray::from_vec(Shape::d1(2), vec![5.0, 6.0]);
        let result = NdArray::column_stack(&[&a, &b, &c]);

        assert_eq!(result.shape().dims(), &[2, 3]);
        // [[1, 3, 5], [2, 4, 6]] in row-major order
        assert_eq!(result.as_slice(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn column_stack_single_array() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let result = NdArray::column_stack(&[&a]);

        assert_eq!(result.shape().dims(), &[3, 1]);
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "All arrays must have the same number of rows")]
    fn column_stack_mismatched_rows() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![4.0, 5.0]);
        NdArray::column_stack(&[&a, &b]);
    }

    #[test]
    #[should_panic(expected = "Need at least one array")]
    fn column_stack_empty() {
        NdArray::column_stack(&[]);
    }

    #[test]
    #[should_panic(expected = "column_stack only supports 1D and 2D arrays")]
    fn column_stack_3d_array() {
        let a = NdArray::from_vec(Shape::new(vec![2, 2, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        NdArray::column_stack(&[&a]);
    }
}
