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
}
