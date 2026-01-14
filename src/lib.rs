pub mod array;
pub mod ops;
pub mod random;

pub use array::{NdArray, Shape, Storage, BroadcastIter};
pub use random::Generator;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PySlice, PyTuple};

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
    fn asarray(data: Vec<f64>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let shape = if let Some(s) = shape {
            let expected_size: usize = s.iter().product();
            if data.len() != expected_size {
                return Err(PyValueError::new_err(format!(
                    "Data length {} doesn't match shape {:?} (expected {})",
                    data.len(), s, expected_size
                )));
            }
            Shape::new(s)
        } else {
            Shape::d1(data.len())
        };

        Ok(PyArray {
            inner: NdArray::from_vec(shape, data),
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

    fn quantile(&self, q: f64) -> f64 {
        self.inner.quantile(q)
    }

    fn any(&self) -> bool {
        self.inner.any()
    }

    fn all(&self) -> bool {
        self.inner.all()
    }

    fn pearson(&self, other: &PyArray) -> f64 {
        self.inner.pearson(&other.inner)
    }

    fn spearman(&self, other: &PyArray) -> f64 {
        self.inner.spearman(&other.inner)
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

    fn cholesky(&self) -> PyResult<PyArray> {
        self.inner.cholesky()
            .map(|l| PyArray { inner: l })
            .map_err(|e| PyValueError::new_err(e))
    }

    fn qr(&self) -> PyResult<(PyArray, PyArray)> {
        self.inner.qr()
            .map(|(q, r)| (PyArray { inner: q }, PyArray { inner: r }))
            .map_err(|e| PyValueError::new_err(e))
    }

    fn eig(&self) -> PyResult<(PyArray, PyArray)> {
        self.inner.eig()
            .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
            .map_err(|e| PyValueError::new_err(e))
    }

    #[pyo3(signature = (max_iter=1000, tol=1e-10))]
    fn eig_with_params(&self, max_iter: usize, tol: f64) -> PyResult<(PyArray, PyArray)> {
        self.inner.eig_with_params(max_iter, tol)
            .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
            .map_err(|e| PyValueError::new_err(e))
    }

    fn eigvals(&self) -> PyResult<PyArray> {
        self.inner.eigvals()
            .map(|vals| PyArray { inner: vals })
            .map_err(|e| PyValueError::new_err(e))
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
        let dims = self.inner.shape().dims();
        let ndim = dims.len();

        if let Ok(tuple) = key.downcast::<PyTuple>() {
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

        if let Ok(slice) = key.downcast::<PySlice>() {
            let len = self.inner.as_slice().len() as isize;
            let indices = slice.indices(len)?;
            let mut result = Vec::new();
            let mut i = indices.start;
            let data = self.inner.as_slice();
            while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
                if i >= 0 && i < len {
                    result.push(data[i as usize]);
                }
                i += indices.step;
            }
            return Ok(PyArray {
                inner: NdArray::from_vec(Shape::d1(result.len()), result),
            }.into_pyobject(py)?.into_any().unbind());
        }

        Err(PyValueError::new_err("indices must be integers, slices, or tuples of integers"))
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: f64) -> PyResult<()> {
        let dims = self.inner.shape().dims().to_vec();
        let ndim = dims.len();

        if let Ok(tuple) = key.downcast::<PyTuple>() {
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

    fn gamma(&mut self, shape_param: f64, scale: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.gamma(shape_param, scale, Shape::new(shape)),
        }
    }

    fn beta(&mut self, alpha: f64, beta_param: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.beta(alpha, beta_param, Shape::new(shape)),
        }
    }

    fn lognormal(&mut self, mu: f64, sigma: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: self.inner.lognormal(mu, sigma, Shape::new(shape)),
        }
    }
}

#[pymodule]
mod substratum {
    use super::*;

    #[pymodule_export]
    use super::PyArray;

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

    #[pyfunction]
    fn ones(shape: Vec<usize>) -> PyArray {
        PyArray::ones(shape)
    }

    #[pyfunction]
    fn full(shape: Vec<usize>, fill_value: f64) -> PyArray {
        PyArray::full(shape, fill_value)
    }

    #[pyfunction]
    #[pyo3(signature = (data, shape=None))]
    fn asarray(data: Vec<f64>, shape: Option<Vec<usize>>) -> PyResult<PyArray> {
        PyArray::asarray(data, shape)
    }

    #[pymodule]
    mod linalg {
        use super::*;

        #[pyfunction]
        fn matmul(a: &PyArray, b: &PyArray) -> PyArray {
            PyArray { inner: a.inner.matmul(&b.inner) }
        }

        #[pyfunction]
        fn dot(a: &PyArray, b: &PyArray) -> PyArray {
            PyArray { inner: a.inner.dot(&b.inner) }
        }

        #[pyfunction]
        fn transpose(a: &PyArray) -> PyArray {
            PyArray { inner: a.inner.transpose() }
        }

        #[pyfunction]
        fn cholesky(a: &PyArray) -> PyResult<PyArray> {
            a.inner.cholesky()
                .map(|l| PyArray { inner: l })
                .map_err(|e| PyValueError::new_err(e))
        }

        #[pyfunction]
        fn qr(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
            a.inner.qr()
                .map(|(q, r)| (PyArray { inner: q }, PyArray { inner: r }))
                .map_err(|e| PyValueError::new_err(e))
        }

        #[pyfunction]
        fn eig(a: &PyArray) -> PyResult<(PyArray, PyArray)> {
            a.inner.eig()
                .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
                .map_err(|e| PyValueError::new_err(e))
        }

        #[pyfunction]
        #[pyo3(signature = (a, max_iter=1000, tol=1e-10))]
        fn eig_with_params(a: &PyArray, max_iter: usize, tol: f64) -> PyResult<(PyArray, PyArray)> {
            a.inner.eig_with_params(max_iter, tol)
                .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
                .map_err(|e| PyValueError::new_err(e))
        }

        #[pyfunction]
        fn eigvals(a: &PyArray) -> PyResult<PyArray> {
            a.inner.eigvals()
                .map(|vals| PyArray { inner: vals })
                .map_err(|e| PyValueError::new_err(e))
        }

        #[pyfunction]
        #[pyo3(signature = (a, k=None))]
        fn diagonal(a: &PyArray, k: Option<isize>) -> PyResult<PyArray> {
            if a.inner.ndim() != 2 {
                return Err(PyValueError::new_err("diagonal requires a 2D array"));
            }
            Ok(PyArray {
                inner: a.inner.diagonal(k.unwrap_or(0)),
            })
        }
    }

    #[pymodule]
    mod stats {
        use super::*;

        #[pyfunction]
        fn sum(a: &PyArray) -> f64 {
            a.inner.sum()
        }

        #[pyfunction]
        fn mean(a: &PyArray) -> f64 {
            a.inner.mean()
        }

        #[pyfunction]
        fn var(a: &PyArray) -> f64 {
            a.inner.var()
        }

        #[pyfunction]
        fn std(a: &PyArray) -> f64 {
            a.inner.std()
        }

        #[pyfunction]
        fn median(a: &PyArray) -> f64 {
            a.inner.median()
        }

        #[pyfunction]
        fn quantile(a: &PyArray, q: f64) -> f64 {
            a.inner.quantile(q)
        }

        #[pyfunction]
        fn any(a: &PyArray) -> bool {
            a.inner.any()
        }

        #[pyfunction]
        fn all(a: &PyArray) -> bool {
            a.inner.all()
        }

        #[pyfunction]
        fn pearson(a: &PyArray, b: &PyArray) -> f64 {
            a.inner.pearson(&b.inner)
        }

        #[pyfunction]
        fn spearman(a: &PyArray, b: &PyArray) -> f64 {
            a.inner.spearman(&b.inner)
        }
    }

    #[pymodule]
    mod random {
        use super::*;

        #[pymodule_export]
        use super::PyGenerator as Generator;

        #[pyfunction]
        fn seed(seed: u64) -> PyGenerator {
            PyGenerator::from_seed(seed)
        }
    }
}
