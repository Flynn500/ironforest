pub mod array;
pub mod ops;
pub mod random;
pub mod spatial;

pub use array::{NdArray, Shape, Storage, BroadcastIter};
pub use random::Generator;
pub use spatial::BallTree;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PySlice, PyTuple};

#[derive(FromPyObject)]
enum ArrayOrScalar {
    Array(PyArray),
    Scalar(f64),
}

/// Accepts either a Python list (Vec<f64>) or a PyArray, extracting the flat data.
#[derive(FromPyObject)]
enum VecOrArray {
    Array(PyArray),
    Vec(Vec<f64>),
}

impl VecOrArray {
    fn into_vec(self) -> Vec<f64> {
        match self {
            VecOrArray::Array(arr) => arr.inner.as_slice().to_vec(),
            VecOrArray::Vec(v) => v,
        }
    }

    fn into_ndarray(self) -> NdArray<f64> {
        match self {
            VecOrArray::Array(arr) => arr.inner,
            VecOrArray::Vec(v) => {
                let len = v.len();
                NdArray::from_vec(Shape::d1(len), v)
            }
        }
    }
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
    fn asarray(data: VecOrArray, shape: Option<Vec<usize>>) -> PyResult<Self> {
        // If it's already an Array and no reshape is requested, return a clone
        let data_vec = data.into_vec();
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
    fn diag(v: VecOrArray, k: Option<isize>) -> Self {
        let v_arr = v.into_ndarray();
        PyArray {
            inner: NdArray::from_diag(&v_arr, k.unwrap_or(0)),
        }
    }

    #[staticmethod]
    fn outer(a: VecOrArray, b: VecOrArray) -> Self {
        let a_arr = a.into_ndarray();
        let b_arr = b.into_ndarray();
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

    fn quantile(&self, py: Python<'_>, q: ArrayOrScalar) -> PyResult<Py<PyAny>> {
        match q {
            ArrayOrScalar::Scalar(q) => Ok(self.inner.quantile(q).into_pyobject(py)?.into_any().unbind()),
            ArrayOrScalar::Array(arr) => Ok(PyArray {
                inner: self.inner.quantiles(arr.inner.as_slice()),
            }.into_pyobject(py)?.into_any().unbind()),
        }
    }

    fn any(&self) -> bool {
        self.inner.any()
    }

    fn all(&self) -> bool {
        self.inner.all()
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
    fn diag(v: VecOrArray, k: Option<isize>) -> PyArray {
        PyArray::diag(v, k)
    }

    #[pyfunction]
    fn outer(a: VecOrArray, b: VecOrArray) -> PyArray {
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
    fn asarray(data: VecOrArray, shape: Option<Vec<usize>>) -> PyResult<PyArray> {
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
        fn quantile(py: Python<'_>, a: &PyArray, q: ArrayOrScalar) -> PyResult<Py<PyAny>> {
            match q {
                ArrayOrScalar::Scalar(q) => Ok(a.inner.quantile(q).into_pyobject(py)?.into_any().unbind()),
                ArrayOrScalar::Array(arr) => Ok(PyArray {
                    inner: a.inner.quantiles(arr.inner.as_slice()),
                }.into_pyobject(py)?.into_any().unbind()),
            }
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

        #[pyfunction]
        fn new() -> PyGenerator {
            PyGenerator::new()
        }
    }

    #[pymodule]
    mod spatial {
        use super::*;

        #[pyclass(name = "BallTree")]
        pub struct PyBallTree {
            inner: BallTree,
        }

        #[pymethods]
        impl PyBallTree {
            #[staticmethod]
            #[pyo3(signature = (array, leaf_size=20))]
            fn from_array(array: &PyArray, leaf_size: Option<usize>) -> PyResult<Self> {
                let leaf_size = leaf_size.unwrap_or(20);
                let tree = BallTree::from_ndarray(&array.inner, leaf_size);
                Ok(PyBallTree { inner: tree })
            }

            fn query_radius(&self, query: &Bound<'_, PyAny>, radius: f64) -> PyResult<PyArray> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let indices = self.inner.query_radius(&query_vec, radius);

                let indices_f64: Vec<f64> = indices.iter().map(|&i| i as f64).collect();

                Ok(PyArray {
                    inner: NdArray::from_vec(Shape::d1(indices_f64.len()), indices_f64),
                })
            }

            fn query_knn(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<PyArray> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let indices = self.inner.query_knn(&query_vec, k);

                let indices_f64: Vec<f64> = indices.iter().map(|&i| i as f64).collect();

                Ok(PyArray {
                    inner: NdArray::from_vec(Shape::d1(indices_f64.len()), indices_f64),
                })
            }
        }
    }
}
