use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;

impl NdArray<i64> {
    pub fn ones(shape: Shape) -> Self {
        Self::filled(shape, 1_i64)
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
            *arr.get_mut(&[row, col]).unwrap() = 1_i64;
            row += 1;
            col += 1;
        }
        arr
    }
}

impl NdArray<f32> {
    pub fn ones(shape: Shape) -> Self {
        Self::filled(shape, 1.0f32)
    }

    pub fn full(shape: Shape, value: f32) -> Self {
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
            *arr.get_mut(&[row, col]).unwrap() = 1.0f32;
            row += 1;
            col += 1;
        }
        arr
    }

    pub fn from_diag(v: &NdArray<f32>, k: isize) -> Self {
        assert_eq!(v.ndim(), 1, "Input must be 1D");
        let n = v.len();
        let size = n + k.unsigned_abs();
        let mut arr = NdArray::zeros(Shape::d2(size, size));

        let (row_start, col_start) = if k >= 0 {
            (0, k as usize)
        } else {
            ((-k) as usize, 0)
        };

        for (i, &val) in v.as_slice_unchecked().iter().enumerate() {
            *arr.get_mut(&[row_start + i, col_start + i]).unwrap() = val;
        }
        arr
    }
}

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

        for (i, &val) in v.as_slice_unchecked().iter().enumerate() {
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
                arrays_2d.push(NdArray::from_vec(Shape::d2(n, 1), arr.as_slice_unchecked().to_vec()));
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
