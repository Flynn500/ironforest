use crate::array::shape::Shape;
use crate::array::storage::Storage;

#[derive(Debug, Clone)]
pub struct NdArray<T> {
    shape: Shape,
    strides: Vec<usize>,
    storage: Storage<T>,
}

impl<T> NdArray<T> {
    pub fn new(shape: Shape, storage: Storage<T>) -> Self {
        assert_eq!(
            shape.size(),
            storage.len(),
            "Storage length {} doesn't match shape size {}",
            storage.len(),
            shape.size()
        );

        let strides = shape.strides_row_major();
        
        NdArray {
            shape,
            strides,
            storage,
        }
    }

    pub fn from_vec(shape: Shape, data: Vec<T>) -> Self {
        Self::new(shape, Storage::from_vec(data))
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub fn len(&self) -> usize {
        self.shape.size()
    }

    pub fn is_empty(&self) -> bool {
        self.shape.size() == 0
    }

    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.ndim() {
            return None;
        }

        let mut offset = 0;
        for (i, (&idx, &dim)) in indices.iter().zip(self.shape.dims()).enumerate() {
            if idx >= dim {
                return None;
            }
            offset += idx * self.strides[i];
        }
        
        Some(offset)
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        self.flat_index(indices)
            .and_then(|i| self.storage.get(i))
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.flat_index(indices)
            .and_then(|i| self.storage.get_mut(i))
    }
}

impl<T: Copy> NdArray<T> {
    pub fn item(&self) -> T {
        assert_eq!(
            self.len(),
            1,
            "item() can only be called on arrays with exactly one element, got {} elements",
            self.len()
        );
        self.as_slice()[0]
    }
}

impl<T: Clone> NdArray<T> {
    pub fn filled(shape: Shape, value: T) -> Self {
        let storage = Storage::filled(value, shape.size());
        Self::new(shape, storage)
    }
}

impl<T: Default + Clone> NdArray<T> {
    pub fn zeros(shape: Shape) -> Self {
        let storage = Storage::zeros(shape.size());
        Self::new(shape, storage)
    }
}

impl<T: Copy> NdArray<T> {
    pub fn map<F, U>(&self, f: F) -> NdArray<U>
    where
        F: Fn(T) -> U,
        U: Copy,
    {
        let result_data: Vec<U> = self.as_slice().iter().map(|&x| f(x)).collect();
        NdArray::new(self.shape().clone(), Storage::from_vec(result_data))
    }

    pub fn take(&self, indices: &[usize]) -> NdArray<T> {
        let data: Vec<T> = indices.iter()
            .map(|&i| self.as_slice()[i])
            .collect();

        NdArray::from_vec(Shape::d1(data.len()), data)
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_basic() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.shape().dims(), &[2, 3]);
    }

    #[test]
    fn indexing_2d() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[0, 2]), Some(&3));
        assert_eq!(arr.get(&[1, 0]), Some(&4));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    #[test]
    fn indexing_out_of_bounds() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        
        assert_eq!(arr.get(&[2, 0]), None);
        assert_eq!(arr.get(&[0, 3]), None);
        assert_eq!(arr.get(&[0]), None);
    }

    #[test]
    fn zeros_creates_default_values() {
        let arr: NdArray<f64> = NdArray::zeros(Shape::d2(2, 2));
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn filled_creates_repeated_value() {
        let arr = NdArray::filled(Shape::d1(4), 7);
        assert_eq!(arr.as_slice(), &[7, 7, 7, 7]);
    }

    #[test]
    fn get_mut_modifies_element() {
        let mut arr = NdArray::from_vec(Shape::d2(2, 2), vec![1, 2, 3, 4]);
        *arr.get_mut(&[1, 1]).unwrap() = 99;
        assert_eq!(arr.get(&[1, 1]), Some(&99));
    }

    #[test]
    #[should_panic(expected = "Storage length")]
    fn mismatched_shape_panics() {
        NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3]);
    }

    #[test]
    fn eye_square() {
        let arr = NdArray::eye(3, None, 0);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_rectangular() {
        let arr = NdArray::eye(2, Some(4), 0);
        assert_eq!(arr.shape().dims(), &[2, 4]);
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn eye_positive_offset() {
        let arr = NdArray::eye(3, Some(4), 1);
        assert_eq!(arr.shape().dims(), &[3, 4]);
        // [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        assert_eq!(arr.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn eye_negative_offset() {
        let arr = NdArray::eye(3, Some(3), -1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn from_diag_main_diagonal() {
        let v = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let arr = NdArray::from_diag(&v, 0);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        assert_eq!(arr.as_slice(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn from_diag_positive_offset() {
        let v = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let arr = NdArray::from_diag(&v, 1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 1, 0], [0, 0, 2], [0, 0, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn from_diag_negative_offset() {
        let v = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let arr = NdArray::from_diag(&v, -1);
        assert_eq!(arr.shape().dims(), &[3, 3]);
        // [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
        assert_eq!(arr.as_slice(), &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn diagonal_main() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(0);
        assert_eq!(diag.shape().dims(), &[3]);
        assert_eq!(diag.as_slice(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn diagonal_positive_offset() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(1);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[2.0, 6.0]);
    }

    #[test]
    fn diagonal_negative_offset() {
        let arr = NdArray::from_vec(Shape::d2(3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let diag = arr.diagonal(-1);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[4.0, 8.0]);
    }

    #[test]
    fn diagonal_rectangular() {
        let arr = NdArray::from_vec(Shape::d2(2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let diag = arr.diagonal(0);
        assert_eq!(diag.shape().dims(), &[2]);
        assert_eq!(diag.as_slice(), &[1.0, 6.0]);
    }

    #[test]
    fn outer_product() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![3.0, 4.0, 5.0]);
        let result = NdArray::outer(&a, &b);
        assert_eq!(result.shape().dims(), &[2, 3]);
        // [[3, 4, 5], [6, 8, 10]]
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn outer_single_elements() {
        let a = NdArray::from_vec(Shape::d1(1), vec![2.0]);
        let b = NdArray::from_vec(Shape::d1(1), vec![3.0]);
        let result = NdArray::outer(&a, &b);
        assert_eq!(result.shape().dims(), &[1, 1]);
        assert_eq!(result.as_slice(), &[6.0]);
    }

    // Transpose tests
    #[test]
    fn transpose_square() {
        let arr = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let transposed = arr.transpose();
        assert_eq!(transposed.shape().dims(), &[2, 2]);
        // [[1, 2], [3, 4]] -> [[1, 3], [2, 4]]
        assert_eq!(transposed.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn transpose_rectangular() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = arr.transpose();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        assert_eq!(transposed.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_t_alias() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t1 = arr.transpose();
        let t2 = arr.t();
        assert_eq!(t1.as_slice(), t2.as_slice());
        assert_eq!(t1.shape().dims(), t2.shape().dims());
    }

    #[test]
    fn transpose_double_is_identity() {
        let arr = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let double_transposed = arr.transpose().transpose();
        assert_eq!(double_transposed.shape().dims(), arr.shape().dims());
        assert_eq!(double_transposed.as_slice(), arr.as_slice());
    }

    // Matmul tests - 1D x 1D (dot product)
    #[test]
    fn matmul_1d_1d() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![4.0, 5.0, 6.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[1]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result.as_slice(), &[32.0]);
    }

    // Matmul tests - 2D x 1D (matrix-vector)
    #[test]
    fn matmul_2d_1d() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2]);
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(result.as_slice(), &[14.0, 32.0]);
    }

    // Matmul tests - 1D x 2D (vector-matrix)
    #[test]
    fn matmul_1d_2d() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        let b = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[3]);
        // [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]
        assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
    }

    // Matmul tests - 2D x 2D (matrix-matrix)
    #[test]
    fn matmul_2d_2d_square() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2, 2]);
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(result.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_2d_2d_rectangular() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d2(3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = a.matmul(&b);
        assert_eq!(result.shape().dims(), &[2, 2]);
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]] = [[58, 64], [139, 154]]
        assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_identity() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let identity = NdArray::eye(2, None, 0);
        let result = a.matmul(&identity);
        assert_eq!(result.as_slice(), a.as_slice());
    }

    // Dot tests (alias for matmul)
    #[test]
    fn dot_is_matmul_alias() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![5.0, 6.0, 7.0, 8.0]);
        let matmul_result = a.matmul(&b);
        let dot_result = a.dot(&b);
        assert_eq!(matmul_result.as_slice(), dot_result.as_slice());
        assert_eq!(matmul_result.shape().dims(), dot_result.shape().dims());
    }

    #[test]
    fn dot_1d_vectors() {
        let a = NdArray::from_vec(Shape::d1(4), vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(Shape::d1(4), vec![2.0, 3.0, 4.0, 5.0]);
        let result = a.dot(&b);
        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_eq!(result.as_slice(), &[40.0]);
    }

    // Error cases
    #[test]
    #[should_panic(expected = "Inner dimensions must match")]
    fn matmul_2d_2d_dimension_mismatch() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Inner dimensions must match")]
    fn matmul_2d_1d_dimension_mismatch() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have same length")]
    fn matmul_1d_1d_length_mismatch() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(2), vec![1.0, 2.0]);
        a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "transpose requires 2D array")]
    fn transpose_1d_panics() {
        let arr = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        arr.transpose();
    }

    #[test]
    fn cholesky_basic() {
        // Symmetric positive-definite matrix
        // [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        let a = NdArray::from_vec(
            Shape::d2(3, 3),
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]
        );
        let l = a.cholesky().unwrap();
        
        // Expected L: [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        assert_eq!(l.shape().dims(), &[3, 3]);
        assert_eq!(l.as_slice(), &[2.0, 0.0, 0.0, 6.0, 1.0, 0.0, -8.0, 5.0, 3.0]);
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![1.0, 2.0, 2.0, 1.0]);
        assert!(a.cholesky().is_err());
    }

    #[test]
    fn cholesky_non_square_fails() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(a.cholesky().is_err());
    }

    // QR decomposition tests
    #[test]
    fn qr_square_matrix() {
        let a = NdArray::from_vec(Shape::d2(3, 3), vec![
            12.0, -51.0, 4.0,
            6.0, 167.0, -68.0,
            -4.0, 24.0, -41.0
        ]);
        let (q, r) = a.qr().unwrap();

        // Verify Q * R ≈ A
        let reconstructed = q.matmul(&r);
        for i in 0..9 {
            assert!((reconstructed.as_slice()[i] - a.as_slice()[i]).abs() < 1e-10);
        }

        // Verify Q is orthogonal: Q^T * Q ≈ I
        let qt_q = q.t().matmul(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*qt_q.get(&[i, j]).unwrap() - expected).abs() < 1e-10);
            }
        }

        // Verify R is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert!(r.get(&[i, j]).unwrap().abs() < 1e-10);
            }
        }
    }

    #[test]
    fn qr_rectangular_matrix() {
        let a = NdArray::from_vec(Shape::d2(4, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0
        ]);
        let (q, r) = a.qr().unwrap();

        // Q should be 4x4, R should be 4x2
        assert_eq!(q.shape().dims(), &[4, 4]);
        assert_eq!(r.shape().dims(), &[4, 2]);

        // Verify Q * R ≈ A
        let reconstructed = q.matmul(&r);
        for i in 0..8 {
            assert!((reconstructed.as_slice()[i] - a.as_slice()[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn qr_identity_matrix() {
        let a = NdArray::eye(3, None, 0);
        let (q, r) = a.qr().unwrap();

        // For identity, Q and R should both be close to identity (or Q=-I, R=-I)
        let reconstructed = q.matmul(&r);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((*reconstructed.get(&[i, j]).unwrap() - expected).abs() < 1e-10);
            }
        }
    }

    // Eigendecomposition tests
    #[test]
    fn eig_diagonal_matrix() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let a = NdArray::from_vec(Shape::d2(3, 3), vec![
            3.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0
        ]);
        let (eigenvalues, _eigenvectors) = a.eig().unwrap();

        // Eigenvalues should be 3, 2, 1 (sorted by absolute value descending)
        assert_eq!(eigenvalues.len(), 3);
        assert!((eigenvalues.as_slice()[0] - 3.0).abs() < 1e-8);
        assert!((eigenvalues.as_slice()[1] - 2.0).abs() < 1e-8);
        assert!((eigenvalues.as_slice()[2] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn eig_symmetric_matrix() {
        // Symmetric matrix [[2, 1], [1, 2]]
        // Eigenvalues: 3 and 1
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![2.0, 1.0, 1.0, 2.0]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        // Check eigenvalues (sorted by absolute value)
        assert_eq!(eigenvalues.len(), 2);
        let ev1 = eigenvalues.as_slice()[0];
        let ev2 = eigenvalues.as_slice()[1];
        assert!((ev1 - 3.0).abs() < 1e-8 || (ev1 - 1.0).abs() < 1e-8);
        assert!((ev2 - 3.0).abs() < 1e-8 || (ev2 - 1.0).abs() < 1e-8);
        assert!((ev1 - ev2).abs() > 1.5); // They should be different

        // Verify A * v = λ * v for each eigenvector
        for col in 0..2 {
            let lambda = eigenvalues.as_slice()[col];
            let v = NdArray::from_vec(Shape::d1(2), vec![
                *eigenvectors.get(&[0, col]).unwrap(),
                *eigenvectors.get(&[1, col]).unwrap()
            ]);
            let av = a.matmul(&v);
            let lambda_v = NdArray::from_vec(Shape::d1(2), vec![
                lambda * v.as_slice()[0],
                lambda * v.as_slice()[1]
            ]);

            for i in 0..2 {
                assert!((av.as_slice()[i] - lambda_v.as_slice()[i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn eig_1x1_matrix() {
        let a = NdArray::from_vec(Shape::d2(1, 1), vec![5.0]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        assert_eq!(eigenvalues.len(), 1);
        assert!((eigenvalues.as_slice()[0] - 5.0).abs() < 1e-10);
        assert_eq!(eigenvectors.shape().dims(), &[1, 1]);
        assert!((eigenvectors.as_slice()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eig_identity_matrix() {
        let a = NdArray::eye(3, None, 0);
        let (eigenvalues, _eigenvectors) = a.eig().unwrap();

        // All eigenvalues of identity matrix are 1
        assert_eq!(eigenvalues.len(), 3);
        for &ev in eigenvalues.as_slice() {
            assert!((ev - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn eig_non_square_fails() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(a.eig().is_err());
    }

    #[test]
    fn eig_1d_fails() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        assert!(a.eig().is_err());
    }

    #[test]
    fn eigvals_returns_only_eigenvalues() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![2.0, 1.0, 1.0, 2.0]);
        let eigenvalues = a.eigvals().unwrap();

        assert_eq!(eigenvalues.ndim(), 1);
        assert_eq!(eigenvalues.len(), 2);
    }

    #[test]
    fn eig_larger_symmetric_matrix() {
        // 4x4 symmetric matrix
        let a = NdArray::from_vec(Shape::d2(4, 4), vec![
            4.0, 1.0, 1.0, 1.0,
            1.0, 4.0, 1.0, 1.0,
            1.0, 1.0, 4.0, 1.0,
            1.0, 1.0, 1.0, 4.0
        ]);
        let (eigenvalues, eigenvectors) = a.eig().unwrap();

        assert_eq!(eigenvalues.len(), 4);

        // Verify reconstruction: A * V ≈ V * diag(eigenvalues)
        // For each eigenvector column, A * v should equal lambda * v
        for col in 0..4 {
            let lambda = eigenvalues.as_slice()[col];
            let mut v_data = Vec::with_capacity(4);
            for row in 0..4 {
                v_data.push(*eigenvectors.get(&[row, col]).unwrap());
            }
            let v = NdArray::from_vec(Shape::d1(4), v_data.clone());
            let av = a.matmul(&v);

            for i in 0..4 {
                let expected = lambda * v_data[i];
                assert!((av.as_slice()[i] - expected).abs() < 1e-5,
                    "Mismatch at eigenvector {}, element {}: av={}, λv={}",
                    col, i, av.as_slice()[i], expected);
            }
        }
    }

    // item() tests
    #[test]
    fn item_scalar_1d() {
        let scalar = NdArray::from_vec(Shape::d1(1), vec![42]);
        assert_eq!(scalar.item(), 42);
    }

    #[test]
    fn item_scalar_2d() {
        let scalar = NdArray::from_vec(Shape::d2(1, 1), vec![3.14]);
        assert_eq!(scalar.item(), 3.14);
    }

    #[test]
    fn item_dot_product_result() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_vec(Shape::d1(3), vec![4.0, 5.0, 6.0]);
        let result = a.dot(&b);
        assert_eq!(result.item(), 32.0);
    }

    #[test]
    #[should_panic(expected = "item() can only be called on arrays with exactly one element")]
    fn item_non_scalar_panics() {
        let arr = NdArray::from_vec(Shape::d1(3), vec![1, 2, 3]);
        arr.item();
    }

    #[test]
    #[should_panic(expected = "item() can only be called on arrays with exactly one element")]
    fn item_empty_panics() {
        let arr: NdArray<i32> = NdArray::from_vec(Shape::d1(0), vec![]);
        arr.item();
    }
}