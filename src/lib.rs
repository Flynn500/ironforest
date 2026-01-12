pub mod array;
pub mod ops;
pub mod random;

pub use array::{NdArray, Shape, Storage, BroadcastIter};
pub use random::Generator;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PySlice;

#[derive(FromPyObject)]
enum ArrayOrScalar {
    Array(PyArray),
    Scalar(f64),
}

#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    inner: NdArray<f64>,
}

#[pymethods]
impl PyArray {
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyArray {
            inner: NdArray::zeros(Shape::new(shape)),
        }
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
    fn diag(v: Vec<f64>, k: Option<isize>) -> Self {
        let v_arr = NdArray::from_vec(Shape::d1(v.len()), v);
        PyArray {
            inner: NdArray::from_diag(&v_arr, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    fn outer(a: Vec<f64>, b: Vec<f64>) -> Self {
        let a_arr = NdArray::from_vec(Shape::d1(a.len()), a);
        let b_arr = NdArray::from_vec(Shape::d1(b.len()), b);
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

    fn tolist(&self) -> Vec<f64> {
        self.inner.as_slice().to_vec()
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

    fn __add__(&self, other: ArrayOrScalar) -> Self {
        match other {
            ArrayOrScalar::Array(arr) => PyArray { inner: &self.inner + &arr.inner },
            ArrayOrScalar::Scalar(s) => PyArray { inner: &self.inner + s },
        }
    }

    fn __radd__(&self, other: f64) -> Self {
        PyArray { inner: other + &self.inner }
    }

    fn __sub__(&self, other: ArrayOrScalar) -> Self {
        match other {
            ArrayOrScalar::Array(arr) => PyArray { inner: &self.inner - &arr.inner },
            ArrayOrScalar::Scalar(s) => PyArray { inner: &self.inner - s },
        }
    }

    fn __rsub__(&self, other: f64) -> Self {
        PyArray { inner: other - &self.inner }
    }

    fn __mul__(&self, other: ArrayOrScalar) -> Self {
        match other {
            ArrayOrScalar::Array(arr) => PyArray { inner: &self.inner * &arr.inner },
            ArrayOrScalar::Scalar(s) => PyArray { inner: &self.inner * s },
        }
    }

    fn __rmul__(&self, other: f64) -> Self {
        PyArray { inner: other * &self.inner }
    }

    fn __truediv__(&self, other: ArrayOrScalar) -> Self {
        match other {
            ArrayOrScalar::Array(arr) => PyArray { inner: &self.inner / &arr.inner },
            ArrayOrScalar::Scalar(s) => PyArray { inner: &self.inner / s },
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

    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, data={:?})",
            self.inner.shape().dims(),
            self.inner.as_slice())
    }

    fn __len__(&self) -> usize {
        self.inner.as_slice().len()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let data = self.inner.as_slice();
        let len = data.len() as isize;

        if let Ok(idx) = key.extract::<isize>() {
            let idx = if idx < 0 { len + idx } else { idx };
            if idx < 0 || idx >= len {
                return Err(PyValueError::new_err("index out of range"));
            }
            Ok(data[idx as usize].into_pyobject(py)?.into_any().unbind())
        } else if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(len)?;
            let mut result = Vec::new();
            let mut i = indices.start;
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && i < len {
                    result.push(data[i as usize]);
                }
                i += indices.step;
            }
            Ok(PyArray {
                inner: NdArray::from_vec(Shape::d1(result.len()), result),
            }.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyValueError::new_err("indices must be integers or slices"))
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyArrayIter {
        PyArrayIter {
            data: slf.inner.as_slice().to_vec(),
            index: 0,
        }
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

#[pyclass(name = "Generator")]
pub struct PyGenerator {
    inner: Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new() -> Self {
        PyGenerator { inner: Generator::new() }
    }

    #[staticmethod]
    fn from_seed(seed: u64) -> Self {
        PyGenerator { inner: Generator::from_seed(seed) }
    }

    fn uniform(&mut self, low: f64, high: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.uniform(low, high, Shape::new(shape)),
        }
    }

    fn standard_normal(&mut self, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.standard_normal(Shape::new(shape)),
        }
    }

    fn normal(&mut self, mu: f64, sigma: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.normal(mu, sigma, Shape::new(shape)),
        }
    }

    fn randint(&mut self, low: i64, high: i64, shape: Vec<usize>) -> PyArray {
        let arr = self.inner.randint(low, high, Shape::new(shape.clone()));
        let data: Vec<f64> = arr.as_slice().iter().map(|&x| x as f64).collect();
        PyArray {
            inner: NdArray::from_vec(Shape::new(shape), data),
        }
    }
}

#[pymodule]
mod substratum {
    use super::*;

    #[pymodule_export]
    use super::PyArray;

    #[pymodule_export]
    use super::PyGenerator;

    #[pyfunction]
    fn zeros(shape: Vec<usize>) -> PyArray {
        PyArray::zeros(shape)
    }

    #[pyfunction]
    #[pyo3(signature = (n, m=None, k=None))]
    fn eye(n: usize, m: Option<usize>, k: Option<isize>) -> PyArray {
        PyArray::eye(n, m, k)
    }

    #[pyfunction]
    #[pyo3(signature = (v, k=None))]
    fn diag(v: Vec<f64>, k: Option<isize>) -> PyArray {
        PyArray::diag(v, k)
    }

    #[pyfunction]
    fn outer(a: Vec<f64>, b: Vec<f64>) -> PyArray {
        PyArray::outer(a, b)
    }
}
