use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyFloat, PyList, PySlice, PyTuple};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods, PyArrayMethods, IntoPyArray};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayLike};

#[pymethods]
impl PyArray {
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyArray {
            inner: NdArray::zeros(Shape::new(shape)),
        }
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        PyArray {
            inner: NdArray::ones(Shape::new(shape)),
        }
    }

    #[staticmethod]
    fn full(shape: Vec<usize>, fill_value: f64) -> Self {
        PyArray {
            inner: NdArray::full(Shape::new(shape), fill_value),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape=None))]
    fn asarray(data: ArrayLike, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let data_vec = data.into_vec().unwrap();
        let shape = if let Some(s) = shape {
            let expected_size: usize = s.iter().product();
            if data_vec.len() != expected_size {
                return Err(PyValueError::new_err(format!(
                    "Data length {} doesn't match shape {:?} (expected {})",
                    data_vec.len(), s, expected_size
                )));
            }
            Shape::new(s)
        } else {
            Shape::d1(data_vec.len())
        };

        Ok(PyArray {
            inner: NdArray::from_vec(shape, data_vec),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, k=None))]
    fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> Self {
        PyArray {
            inner: NdArray::eye(n, m, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (v, k=None))]
    fn diag(v: ArrayLike, k: Option<isize>) -> Self {
        let v_arr = v.into_ndarray().unwrap();
        PyArray {
            inner: NdArray::from_diag(&v_arr, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    fn outer(a: ArrayLike, b: ArrayLike) -> Self {
        let a_arr = a.into_ndarray().unwrap();
        let b_arr = b.into_ndarray().unwrap();
        PyArray {
            inner: NdArray::outer(&a_arr, &b_arr),
        }
    }

    #[pyo3(signature = (k=None))]
    fn diagonal(&self, k: Option<isize>) -> PyResult<PyArray> {
        if self.inner.ndim() != 2 {
            return Err(PyValueError::new_err("diagonal requires a 2D array"));
        }
        Ok(PyArray {
            inner: self.inner.diagonal(k.unwrap_or(0)),
        })
    }

    #[new]
    fn new(shape: Vec<usize>, data: Vec<f64>) -> PyResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(PyValueError::new_err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_size
            )));
        }
        Ok(PyArray {
            inner: NdArray::from_vec(Shape::new(shape), data),
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        self.inner.get(&indices)
            .copied()
            .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
    }

    pub fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.inner.shape().ndim() == 0 {
            return Ok(PyFloat::new(py, self.inner.as_slice()[0]).unbind().into_any());
        }
        
        self.to_pylist_recursive(py, 0, 0)
    }

    fn to_pylist_recursive(&self, py: Python<'_>, dim: usize, offset: usize) -> PyResult<Py<PyAny>> {
        let data = self.inner.as_slice();
        let dim_size = self.inner.shape().dim(dim).unwrap();

        if dim == self.inner.ndim() - 1 {
            let list = PyList::new(py, (0..dim_size).map(|i| {
                data[offset + i * self.inner.strides()[dim]]
            }))?;
            return Ok(list.into());
        }
        
        let items: Vec<Py<PyAny>> = (0..dim_size)
            .map(|i| {
                let new_offset = offset + i * self.inner.strides()[dim];
                self.to_pylist_recursive(py, dim + 1, new_offset)
            })
            .collect::<PyResult<_>>()?;
        
        Ok(PyList::new(py, items)?.into())
    }

    fn sin(&self) -> Self {
        PyArray { inner: self.inner.sin() }
    }

    fn cos(&self) -> Self {
        PyArray { inner: self.inner.cos() }
    }

    fn exp(&self) -> Self {
        PyArray { inner: self.inner.exp() }
    }

    fn sqrt(&self) -> Self {
        PyArray { inner: self.inner.sqrt() }
    }

    fn clip(&self, min: f64, max: f64) -> Self {
        PyArray { inner: self.inner.clip(min, max) }
    }

    fn tan(&self) -> Self {
        PyArray { inner: self.inner.tan() }
    }

    fn arcsin(&self) -> Self {
        PyArray { inner: self.inner.asin() }
    }

    fn arccos(&self) -> Self {
        PyArray { inner: self.inner.acos() }
    }

    fn arctan(&self) -> Self {
        PyArray { inner: self.inner.atan() }
    }

    fn log(&self) -> Self {
        PyArray { inner: self.inner.log() }
    }

    fn abs(&self) -> Self {
        PyArray { inner: self.inner.abs() }
    }

    fn sign(&self) -> Self {
        PyArray { inner: self.inner.sign() }
    }

    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn var(&self) -> f64 {
        self.inner.var()
    }

    fn std(&self) -> f64 {
        self.inner.std()
    }

    fn median(&self) -> f64 {
        self.inner.median()
    }

    fn quantile(&self, py: Python<'_>, q: ArrayLike) -> PyResult<Py<PyAny>> {
        match q {
            ArrayLike::Scalar(q_val) => {
                Ok(self.inner.quantile(q_val).into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                let q_arr = q.into_ndarray()?;
                Ok(PyArray {
                    inner: self.inner.quantiles(q_arr.as_slice()),
                }.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    fn any(&self) -> bool {
        self.inner.any()
    }

    fn all(&self) -> bool {
        self.inner.all()
    }

    fn __add__(&self, other: ArrayLike) -> Self {
        match other {
            ArrayLike::Scalar(s) => PyArray { inner: &self.inner + s },
            _ => {
                let arr = other.into_ndarray().unwrap();
                PyArray { inner: &self.inner + &arr }
            }
        }
    }

    fn __radd__(&self, other: f64) -> Self {
        PyArray { inner: other + &self.inner }
    }

    fn __sub__(&self, other: ArrayLike) -> Self {
        match other {
            ArrayLike::Scalar(s) => PyArray { inner: &self.inner - s },
            _ => {
                let arr = other.into_ndarray().unwrap();
                PyArray { inner: &self.inner - &arr }
            }
        }
    }

    fn __rsub__(&self, other: f64) -> Self {
        PyArray { inner: other - &self.inner }
    }

    fn __mul__(&self, other: ArrayLike) -> Self {
        match other {
            ArrayLike::Scalar(s) => PyArray { inner: &self.inner * s },
            _ => {
                let arr = other.into_ndarray().unwrap();
                PyArray { inner: &self.inner * &arr }
            }
        }
    }

    fn __rmul__(&self, other: f64) -> Self {
        PyArray { inner: other * &self.inner }
    }

    fn __truediv__(&self, other: ArrayLike) -> Self {
        match other {
            ArrayLike::Scalar(s) => PyArray { inner: &self.inner / s },
            _ => {
                let arr = other.into_ndarray().unwrap();
                PyArray { inner: &self.inner / &arr }
            }
        }
    }

    fn __rtruediv__(&self, other: f64) -> Self {
        PyArray { inner: other / &self.inner }
    }

    fn __neg__(&self) -> Self {
        PyArray { inner: -&self.inner }
    }

    fn __matmul__(&self, other: &PyArray) -> Self {
        PyArray { inner: self.inner.matmul(&other.inner) }
    }

    fn matmul(&self, other: &PyArray) -> Self {
        PyArray { inner: self.inner.matmul(&other.inner) }
    }

    fn dot(&self, other: &PyArray) -> Self {
        PyArray { inner: self.inner.dot(&other.inner) }
    }

    fn transpose(&self) -> Self {
        PyArray { inner: self.inner.transpose() }
    }

    fn t(&self) -> Self {
        PyArray { inner: self.inner.t() }
    }

    fn take(&self, indices: Vec<usize>) -> Self {
        PyArray { inner: self.inner.take(&indices) }
    }

    fn item(&self) -> PyResult<f64> {
        if self.inner.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "item() can only be called on arrays with exactly one element, got {} elements",
                self.inner.len()
            )));
        }
        Ok(self.inner.item())
    }

    #[staticmethod]
    fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let numpy_arr: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let shape: Vec<usize> = numpy_arr.shape().to_vec();
        let data: Vec<f64> = numpy_arr.as_slice()?.to_vec();

        Ok(PyArray {
            inner: NdArray::from_vec(Shape::new(shape), data),
        })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        use numpy::IntoPyArray;
        let shape = self.inner.shape().dims();
        let data = self.inner.as_slice().to_vec();

        data.into_pyarray(py).reshape(shape).unwrap()
    }

    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, data={:?})",
            self.inner.shape().dims(),
            self.inner.as_slice())
    }

    fn __len__(&self) -> usize {
        self.inner.as_slice().len()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dims = self.inner.shape().dims();
        let ndim = dims.len();

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();

            if tuple_len > ndim {
                return Err(PyValueError::new_err(format!(
                    "too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim, tuple_len
                )));
            }

            let mut indices: Vec<usize> = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                let item = tuple.get_item(i)?;
                let idx = item.extract::<isize>()?;
                let dim_size = dims[i] as isize;
                let normalized = if idx < 0 { dim_size + idx } else { idx };
                if normalized < 0 || normalized >= dim_size {
                    return Err(PyValueError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, i, dims[i]
                    )));
                }
                indices.push(normalized as usize);
            }

            if tuple_len == ndim {
                let value = self.inner.get(&indices)
                    .ok_or_else(|| PyValueError::new_err("index out of bounds"))?;
                return Ok((*value).into_pyobject(py)?.into_any().unbind());
            }

            let result_dims: Vec<usize> = dims[tuple_len..].to_vec();
            let result_size: usize = result_dims.iter().product();

            let mut start_offset = 0;
            let strides = self.inner.strides();
            for (i, &idx) in indices.iter().enumerate() {
                start_offset += idx * strides[i];
            }

            let data = self.inner.as_slice();
            let result_data: Vec<f64> = data[start_offset..start_offset + result_size].to_vec();

            return Ok(PyArray {
                inner: NdArray::from_vec(Shape::new(result_dims), result_data),
            }.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(idx) = key.extract::<isize>() {
            let dim0 = dims[0] as isize;
            let normalized = if idx < 0 { dim0 + idx } else { idx };
            if normalized < 0 || normalized >= dim0 {
                return Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, dims[0]
                )));
            }

            if ndim == 1 {
                let data = self.inner.as_slice();
                return Ok(data[normalized as usize].into_pyobject(py)?.into_any().unbind());
            }

            let result_dims: Vec<usize> = dims[1..].to_vec();
            let result_size: usize = result_dims.iter().product();
            let start_offset = normalized as usize * self.inner.strides()[0];
            let data = self.inner.as_slice();
            let result_data: Vec<f64> = data[start_offset..start_offset + result_size].to_vec();

            return Ok(PyArray {
                inner: NdArray::from_vec(Shape::new(result_dims), result_data),
            }.into_pyobject(py)?.into_any().unbind());
        }

        if let Ok(slice) = key.cast::<PySlice>() {
            let axis0_len = dims[0] as isize;
            let indices = slice.indices(axis0_len)?;

            if ndim == 1 {
                let mut result = Vec::new();
                let mut i = indices.start;
                let data = self.inner.as_slice();
                while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                    if i >= 0 && i < axis0_len {
                        result.push(data[i as usize]);
                    }
                    i += indices.step;
                }
                return Ok(PyArray {
                    inner: NdArray::from_vec(Shape::d1(result.len()), result),
                }.into_pyobject(py)?.into_any().unbind());
            }

            let row_size: usize = dims[1..].iter().product();
            let mut result = Vec::new();
            let mut i = indices.start;
            let data = self.inner.as_slice();
            let stride = self.inner.strides()[0];

            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && i < axis0_len {
                    let row_start = (i as usize) * stride;
                    result.extend_from_slice(&data[row_start..row_start + row_size]);
                }
                i += indices.step;
            }

            let num_rows = if indices.step > 0 {
                ((indices.stop - indices.start + indices.step - 1) / indices.step).max(0) as usize
            } else {
                ((indices.start - indices.stop - indices.step - 1) / (-indices.step)).max(0) as usize
            };

            let mut result_dims = vec![num_rows];
            result_dims.extend_from_slice(&dims[1..]);

            return Ok(PyArray {
                inner: NdArray::from_vec(Shape::new(result_dims), result),
            }.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err("indices must be integers, slices, or tuples of integers"))
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let dims = self.inner.shape().dims().to_vec();
        let ndim = dims.len();

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();

            if tuple_len != ndim {
                return Err(PyValueError::new_err(format!(
                    "cannot set item: expected {} indices, got {}",
                    ndim, tuple_len
                )));
            }

            let mut indices: Vec<usize> = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                let item = tuple.get_item(i)?;
                let idx = item.extract::<isize>()?;
                let dim_size = dims[i] as isize;
                let normalized = if idx < 0 { dim_size + idx } else { idx };
                if normalized < 0 || normalized >= dim_size {
                    return Err(PyValueError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, i, dims[i]
                    )));
                }
                indices.push(normalized as usize);
            }

            let elem = self.inner.get_mut(&indices)
                .ok_or_else(|| PyValueError::new_err("index out of bounds"))?;
            *elem = value;
            return Ok(());
        }

        if let Ok(idx) = key.extract::<isize>() {
            if ndim != 1 {
                return Err(PyValueError::new_err(
                    "single index assignment only supported for 1D arrays; use tuple indexing for nD arrays"
                ));
            }

            let dim0 = dims[0] as isize;
            let normalized = if idx < 0 { dim0 + idx } else { idx };
            if normalized < 0 || normalized >= dim0 {
                return Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, dims[0]
                )));
            }

            let data = self.inner.as_mut_slice();
            data[normalized as usize] = value;
            return Ok(());
        }

        Err(PyValueError::new_err("indices must be integers or tuples of integers for assignment"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyArrayIter {
        PyArrayIter {
            data: slf.inner.as_slice().to_vec(),
            index: 0,
        }
    }

    fn __contains__(&self, value: f64) -> bool {
        self.inner.as_slice().contains(&value)
    }
}

#[pyclass]
pub struct PyArrayIter {
    data: Vec<f64>,
    index: usize,
}

#[pymethods]
impl PyArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f64> {
        if slf.index < slf.data.len() {
            let val = slf.data[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

// Top-level array construction functions
#[pyfunction]
pub fn zeros(shape: Vec<usize>) -> PyArray {
    PyArray::zeros(shape)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=None))]
pub fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> PyArray {
    PyArray::eye(n, m, k)
}

#[pyfunction]
#[pyo3(signature = (v, k=None))]
pub fn diag(v: ArrayLike, k: Option<isize>) -> PyArray {
    PyArray::diag(v, k)
}

#[pyfunction]
pub fn column_stack(arrays: Vec<PyArray>) -> PyResult<PyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }

    let array_refs: Vec<&NdArray<f64>> = arrays.iter().map(|a| &a.inner).collect();
    Ok(PyArray {
        inner: NdArray::column_stack(&array_refs),
    })
}

#[pyfunction]
pub fn ones(shape: Vec<usize>) -> PyArray {
    PyArray::ones(shape)
}

#[pyfunction]
pub fn full(shape: Vec<usize>, fill_value: f64) -> PyArray {
    PyArray::full(shape, fill_value)
}

#[pyfunction]
#[pyo3(signature = (data, shape=None))]
pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>) -> PyResult<PyArray> {
    PyArray::asarray(data, shape)
}