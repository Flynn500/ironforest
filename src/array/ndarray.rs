use crate::array::shape::Shape;
use crate::array::storage::Storage;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn row(&self, i: usize) -> &[T] {
        assert!(self.ndim() >= 2, "row() requires at least 2 dimensions");
        assert!(i < self.shape.dims()[0], "Row index {} out of bounds for axis 0 with size {}", i, self.shape.dims()[0]);
        let start = i * self.strides[0];
        &self.as_slice()[start..start + self.strides[0]]
    }

    pub fn set_row(&mut self, i: usize, values: &[T]) where T: Copy {
        assert!(self.ndim() >= 2);
        assert!(i < self.shape.dims()[0]);
        let start = i * self.strides[0];
        let row = &mut self.storage.as_mut_slice()[start..start + self.strides[0]];
        row.copy_from_slice(values);
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

    pub fn broadcast_to(&self, target: &Shape) -> Option<NdArray<T>> {
        if !self.shape().broadcasts_to(target) {
            return None;
        }

        let mut data = Vec::with_capacity(target.size());
        let src_ndim = self.ndim();
        let tgt_ndim = target.ndim();

        for flat_i in 0..target.size() {
            let mut remaining = flat_i;
            let mut src_flat = 0;

            for dim in 0..tgt_ndim {
                let tgt_dim_size = target.dims()[dim];
                let stride = target.dims()[dim + 1..].iter().product::<usize>().max(1);
                let idx = (remaining / stride) % tgt_dim_size;
                remaining %= stride;

                let src_dim = tgt_ndim - src_ndim;
                let src_idx = if dim >= src_dim {
                    let s = self.shape().dims()[dim - src_dim];
                    if s == 1 { 0 } else { idx }
                } else {
                    0
                };

                let src_stride = self.strides().get(dim.saturating_sub(src_dim)).copied().unwrap_or(1);
                src_flat += src_idx * src_stride;
            }

            data.push(self.as_slice()[src_flat]);
        }

        Some(NdArray::from_vec(target.clone(), data))
    }
}

impl<T: Clone> NdArray<T> {
    pub fn filled(shape: Shape, value: T) -> Self {
        let storage = Storage::filled(value, shape.size());
        Self::new(shape, storage)
    }

    pub fn reshape(&self, new_dims: Vec<usize>) -> Self {
        let new_size: usize = new_dims.iter().product();
        assert_eq!(
            self.len(), new_size,
            "Cannot reshape array of size {} into shape {:?}",
            self.len(), new_dims
        );
        NdArray::from_vec(Shape::new(new_dims), self.as_slice().to_vec())
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

    pub fn boolean_mask(&self, mask: &[bool]) -> Self {
        let n_rows = if self.ndim() == 0 { 0 } else { self.shape().dims()[0] };
        assert_eq!(
            mask.len(), n_rows,
            "Boolean mask length {} must match first dimension {}",
            mask.len(), n_rows
        );
        let row_size = if n_rows == 0 { 0 } else { self.len() / n_rows };
        let data: Vec<T> = mask.iter().enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .flat_map(|i| self.as_slice()[i * row_size..(i + 1) * row_size].iter().copied())
            .collect();

        let n_selected = if row_size == 0 { 0 } else { data.len() / row_size };
        if self.ndim() <= 1 {
            NdArray::from_vec(Shape::d1(n_selected), data)
        } else {
            let mut new_dims = self.shape().dims().to_vec();
            new_dims[0] = n_selected;
            NdArray::from_vec(Shape::new(new_dims), data)
        }
    }
}

