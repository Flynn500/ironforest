use std::borrow::Cow;
use std::sync::Arc;
use pyo3::{Py, PyAny};
use crate::array::shape::Shape;
use crate::array::storage::Storage;
use crate::array::strided_iter::StridedIter;
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

    pub fn as_slice(&self) -> Option<&[T]> {
        self.storage.as_slice()
    }

    #[inline]
    pub fn as_slice_unchecked(&self) -> &[T] {
        self.storage.as_slice_unchecked()
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        self.storage.as_mut_slice()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
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
        &self.as_slice_unchecked()[start..start + self.strides[0]]
    }

    pub fn set_row(&mut self, i: usize, values: &[T]) where T: Copy {
        assert!(self.ndim() >= 2);
        assert!(i < self.shape.dims()[0]);
        let start = i * self.strides[0];
        let slice = self.storage.as_mut_slice()
            .expect("set_row() requires owned (mutable) storage");
        let row = &mut slice[start..start + self.strides[0]];
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

    pub fn is_contiguous(&self) -> bool {
        self.storage.is_contiguous()
    }

    pub fn is_owned(&self) -> bool {
        self.storage.is_owned()
    }

    #[inline(always)]
    pub(crate) fn linear_to_storage_offset(&self, linear: usize) -> usize {
        let mut remaining = linear;
        let mut src_offset = 0usize;
        for dim in 0..self.ndim() {
            let stride_below: usize =
                self.shape.dims()[dim + 1..].iter().product::<usize>().max(1);
            let idx = remaining / stride_below;
            remaining %= stride_below;
            src_offset += idx * self.strides[dim];
        }
        src_offset
    }

    #[inline(always)]
    pub(crate) fn storage_get(&self, offset: usize) -> Option<&T> {
        self.storage.get(offset)
    }

    pub unsafe fn from_external(owner: Py<PyAny>, ptr: *const T, shape: Shape) -> Self {
        let len = shape.size();
        let strides = shape.strides_row_major();
        NdArray {
            shape,
            strides,
            storage: Storage::External { owner, ptr, len },
        }
    }
}

impl<T: Copy> NdArray<T> {
    pub fn as_contiguous_slice(&self) -> Cow<[T]> {
        if self.is_contiguous() {
            Cow::Borrowed(self.storage.as_slice_unchecked())
        } else {
            Cow::Owned(self.to_contiguous().into_vec())
        }
    }

    pub fn iter_logical(&self) -> StridedIter<T> {
        StridedIter::new(self)
    }

    pub fn to_contiguous(&self) -> NdArray<T> {
        if self.is_contiguous() {
            NdArray::from_vec(self.shape.clone(), self.storage.as_slice_unchecked().to_vec())
        } else {
            let mut data = Vec::with_capacity(self.len());
            for flat in 0..self.len() {
                let mut remaining = flat;
                let mut src_offset = 0usize;
                for dim in 0..self.ndim() {
                    let stride_below: usize =
                        self.shape.dims()[dim + 1..].iter().product::<usize>().max(1);
                    let idx = remaining / stride_below;
                    remaining %= stride_below;
                    src_offset += idx * self.strides[dim];
                }
                data.push(*self.storage.get(src_offset).expect("index out of bounds in to_contiguous"));
            }
            NdArray::from_vec(self.shape.clone(), data)
        }
    }

    pub fn row_cow(&self, i: usize) -> Cow<[T]> {
        assert!(self.ndim() >= 2, "row_cow() requires at least 2 dimensions");
        assert!(i < self.shape.dims()[0]);
        if self.is_contiguous() {
            let start = i * self.strides[0];
            Cow::Borrowed(&self.storage.as_slice_unchecked()[start..start + self.strides[0]])
        } else {
            // Strided: copy the row into an owned Vec.
            let ncols = self.shape.dims()[1];
            let row_start = i * self.strides[0];
            let mut row = Vec::with_capacity(ncols);
            for j in 0..ncols {
                let idx = row_start + j * self.strides[1];
                row.push(*self.storage.get(idx).expect("strided row index out of bounds"));
            }
            Cow::Owned(row)
        }
    }

    pub fn slice_view(&self, axes: &[(usize, usize, usize)]) -> NdArray<T> {
        assert_eq!(axes.len(), self.ndim(), "axes.len() must equal ndim");
        let new_shape: Vec<usize> = axes.iter().map(|&(_, _, len)| len).collect();
        let new_len: usize = new_shape.iter().product();
        let extra_offset: usize = axes.iter().enumerate()
            .map(|(i, &(start, _, _))| start * self.strides[i])
            .sum();
        let new_strides: Vec<usize> = axes.iter().enumerate()
            .map(|(i, &(_, step, _))| self.strides[i] * step)
            .collect();

        let storage = match &self.storage {
            Storage::Owned(v) => {
                let arc: Arc<[T]> = v.clone().into_boxed_slice().into();
                Storage::Strided { base: arc, offset: extra_offset, len: new_len }
            }
            Storage::Strided { base, offset, .. } => {
                Storage::Strided {
                    base: Arc::clone(base),
                    offset: offset + extra_offset,
                    len: new_len,
                }
            }
            Storage::External { .. } => {
                let owned = self.to_contiguous();
                return owned.slice_view(axes);
            }
        };
        NdArray { shape: Shape::new(new_shape), strides: new_strides, storage }
    }

    pub fn item(&self) -> T {
        assert_eq!(
            self.len(),
            1,
            "item() can only be called on arrays with exactly one element, got {} elements",
            self.len()
        );
        self.as_slice_unchecked()[0]
    }

    pub fn broadcast_to(&self, target: &Shape) -> Option<NdArray<T>> {
        if !self.shape().broadcasts_to(target) {
            return None;
        }

        let contiguous;
        let src = if self.is_contiguous() {
            self
        } else {
            contiguous = self.to_contiguous();
            &contiguous
        };
        let src_data = src.as_slice_unchecked();

        let mut data = Vec::with_capacity(target.size());
        let src_ndim = src.ndim();
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
                    let s = src.shape().dims()[dim - src_dim];
                    if s == 1 { 0 } else { idx }
                } else {
                    0
                };

                let src_stride = src.strides().get(dim.saturating_sub(src_dim)).copied().unwrap_or(1);
                src_flat += src_idx * src_stride;
            }

            data.push(src_data[src_flat]);
        }

        Some(NdArray::from_vec(target.clone(), data))
    }

    pub fn map<F, U>(&self, f: F) -> NdArray<U>
    where
        F: Fn(T) -> U,
        U: Copy,
    {
        let result_data: Vec<U> = self.iter_logical().map(f).collect();
        NdArray::new(self.shape().clone(), Storage::from_vec(result_data))
    }

    pub fn take(&self, indices: &[usize]) -> NdArray<T> {
        let contiguous;
        let src = if self.is_contiguous() { self } else { contiguous = self.to_contiguous(); &contiguous };
        let flat = src.as_slice_unchecked();
        let data: Vec<T> = indices.iter().map(|&i| flat[i]).collect();
        NdArray::from_vec(Shape::d1(data.len()), data)
    }

    pub fn boolean_mask(&self, mask: &[bool]) -> Self {
        let n_rows = if self.ndim() == 0 { 0 } else { self.shape().dims()[0] };
        assert_eq!(
            mask.len(), n_rows,
            "Boolean mask length {} must match first dimension {}",
            mask.len(), n_rows
        );
        let contiguous;
        let src = if self.is_contiguous() { self } else { contiguous = self.to_contiguous(); &contiguous };
        let flat = src.as_slice_unchecked();
        let row_size = if n_rows == 0 { 0 } else { src.len() / n_rows };
        let data: Vec<T> = mask.iter().enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .flat_map(|i| flat[i * row_size..(i + 1) * row_size].iter().copied())
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

impl<T: Clone> NdArray<T> {
    pub fn as_view(&self) -> Self {
        match &self.storage {
            Storage::External { owner, ptr, len } => NdArray {
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                storage: Storage::External {
                    owner: pyo3::Python::attach(|py| owner.clone_ref(py)),
                    ptr: *ptr,
                    len: *len,
                },
            },
            _ => self.clone(),
        }
    }

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

        NdArray::from_vec(Shape::new(new_dims), self.storage.to_owned_vec())
    }
}

impl<T: Default + Clone> NdArray<T> {
    pub fn zeros(shape: Shape) -> Self {
        let storage = Storage::zeros(shape.size());
        Self::new(shape, storage)
    }
}
