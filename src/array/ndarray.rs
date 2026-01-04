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
}