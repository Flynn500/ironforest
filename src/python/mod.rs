use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyAny;
use pyo3::{FromPyObject, Borrowed};
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use crate::array::{NdArray, Shape};

/// Validates that `v` is a non-empty, rectangular 2-D nested list.
/// Returns `(rows, cols)` on success, or a `PyValueError` if the list is empty
/// or has rows of inconsistent length.
fn validate_vec2d<T>(v: &[Vec<T>]) -> PyResult<(usize, usize)> {
    if v.is_empty() {
        return Err(PyValueError::new_err("Cannot create array from empty nested list"));
    }
    let cols = v[0].len();
    for row in v {
        if row.len() != cols {
            return Err(PyValueError::new_err("Nested lists must have consistent dimensions"));
        }
    }
    Ok((v.len(), cols))
}

#[derive(Clone)]
pub enum ArrayData {
    Float(NdArray<f64>),
    Int(NdArray<i64>),
}

pub enum ArrayLike {
    Array(PyArray),
    Numpy { shape: Vec<usize>, data: Vec<f64> },
    Vec(Vec<f64>),
    Vec2D(Vec<Vec<f64>>),
    Scalar(f64),
    IntScalar(i64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ArrayLike {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(arr) = ob.extract::<PyArray>() {
            return Ok(ArrayLike::Array(arr));
        }

        if let Ok(arr) = ob.extract::<PyReadonlyArrayDyn<f64>>() {
            let shape = arr.shape().to_vec();
            let data = arr.as_slice()?.to_vec();
            return Ok(ArrayLike::Numpy { shape, data });
        }

        if let Ok(outer_list) = ob.extract::<Vec<Vec<f64>>>() {
            return Ok(ArrayLike::Vec2D(outer_list));
        }

        if let Ok(list) = ob.extract::<Vec<f64>>() {
            return Ok(ArrayLike::Vec(list));
        }

        if let Ok(s) = ob.extract::<i64>() {
            return Ok(ArrayLike::IntScalar(s));
        }

        if let Ok(scalar) = ob.extract::<f64>() {
            return Ok(ArrayLike::Scalar(scalar));
        }

        Err(PyValueError::new_err(
            "Expected Array, NumPy array, list, or scalar"
        ))
    }
}

impl ArrayLike {
    pub fn ndim(&self) -> usize {
        match self {
            ArrayLike::Array(a) => match &a.inner {
                ArrayData::Float(f) => f.shape().dims().len(),
                ArrayData::Int(i) => i.shape().dims().len(),
            },
            ArrayLike::Numpy { shape, .. } => shape.len(),
            ArrayLike::Vec(_) => 1,
            ArrayLike::Vec2D(_) => 2,
            ArrayLike::Scalar(_) | ArrayLike::IntScalar(_) => 0,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            ArrayLike::Array(a) => matches!(a.inner, ArrayData::Int(_)),
            ArrayLike::IntScalar(_) => true,
            _ => false,
        }
    }

    pub fn into_ndarray(self) -> PyResult<NdArray<f64>> {
        match self {
            ArrayLike::Array(arr) => match arr.inner {
                ArrayData::Float(f) => Ok(f),
                ArrayData::Int(_) => Err(PyTypeError::new_err(
                    "expected float array; got integer array"
                )),
            },
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s])),
            ArrayLike::Vec(v) => Ok(NdArray::from_vec(Shape::d1(v.len()), v)),
            ArrayLike::Vec2D(v) => {
                let (rows, cols) = validate_vec2d(&v)?;
                let flat: Vec<f64> = v.into_iter().flatten().collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), flat))
            }
            ArrayLike::Numpy { shape, data } => {
                Ok(NdArray::from_vec(Shape::new(shape), data))
            }
            ArrayLike::IntScalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s as f64])),
        }
    }

    pub fn into_spatial_query_ndarray(self, expected_dim: usize) -> PyResult<NdArray<f64>> {
        let arr = self.into_ndarray()?;
        let shape = arr.shape().dims();
        
        match shape.len() {
            1 => {
                if shape[0] != expected_dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} doesn't match expected dimension {}",
                        shape[0], expected_dim
                    )));
                }
                Ok(NdArray::from_vec(
                    Shape::new(vec![1, shape[0]]),
                    arr.as_slice().to_vec()
                ))
            }
            2 => {
                if shape[1] != expected_dim {
                    return Err(PyValueError::new_err(format!(
                        "Query dimension {} doesn't match expected dimension {}",
                        shape[1], expected_dim
                    )));
                }
                Ok(arr)
            }
            _ => Err(PyValueError::new_err("queries must be 1D or 2D array")),
        }
    }

    pub fn into_vec(self) -> PyResult<Vec<f64>> {
        let arr = self.into_ndarray()?;
        let shape = arr.shape().dims();
        
        if shape.len() == 1 || (shape.len() == 2 && shape[0] == 1) {
            Ok(arr.as_slice().to_vec())
        } else {
            Err(PyValueError::new_err("Expected 1D array"))
        }
    }
    
    pub fn into_vec_with_dim(self, expected_dim: usize) -> PyResult<Vec<f64>> {
        let vec = self.into_vec()?;
        if vec.len() != expected_dim {
            return Err(PyValueError::new_err(format!(
                "Array length {} doesn't match expected dimension {}",
                vec.len(), expected_dim
            )));
        }
        Ok(vec)
    }

    pub fn into_i64_ndarray(self) -> PyResult<NdArray<i64>> {
        match self {
            ArrayLike::Array(arr) => match arr.inner {
                ArrayData::Int(i) => Ok(i),
                ArrayData::Float(f) => {
                    let data: Vec<i64> = f.as_slice().iter().map(|&x| x as i64).collect();
                    Ok(NdArray::from_vec(f.shape().clone(), data))
                }
            },
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::new(vec![1]), vec![s as i64])),
            ArrayLike::Vec(v) => {
                let data: Vec<i64> = v.iter().map(|&x| x as i64).collect();
                Ok(NdArray::from_vec(Shape::d1(data.len()), data))
            }
            ArrayLike::Vec2D(v) => {
                let (rows, cols) = validate_vec2d(&v)?;
                let data: Vec<i64> = v.into_iter().flatten().map(|x| x as i64).collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), data))
            }
            ArrayLike::Numpy { shape, data } => {
                let data: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Ok(NdArray::from_vec(Shape::new(shape), data))
            }
            ArrayLike::IntScalar(s) => Ok(NdArray::from_vec(Shape::new(vec![1]), vec![s])),
        }
    }
}

#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    pub inner: ArrayData,
    pub alive: bool,
}

impl PyArray {

    pub fn as_float(&self) -> PyResult<&NdArray<f64>> {
        match &self.inner {
            ArrayData::Float(a) => Ok(a),
            ArrayData::Int(_) => Err(PyTypeError::new_err(
                "operation not supported for integer arrays"
            )),
        }
    }

    pub fn as_int(&self) -> PyResult<&NdArray<i64>> {
        match &self.inner {
            ArrayData::Int(a) => Ok(a),
            ArrayData::Float(_) => Err(PyTypeError::new_err(
                "operation not supported for integer arrays"
            )),
        }
    }
}

// macro for handling dead array case
macro_rules! check_alive {
    ($self:expr) => {
        if !$self.alive {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Array consumed by tree, use tree.data to access the data"
            ));
        }
    };
}

pub mod array;
pub mod ndutils;
pub mod linalg;
pub mod stats;
pub mod random;
pub mod spatial;
pub mod tree_engine;

pub use array::{PyArrayIter, PyIntArrayIter};
pub use random::PyGenerator;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_class::<PyArrayIter>()?;
    m.add_class::<PyIntArrayIter>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<spatial::PyBallTree>()?;
    m.add_class::<spatial::PyKDTree>()?;

    let ndutils_module = PyModule::new(m.py(), "ndutils")?;
    ndutils::register_module(&ndutils_module)?;
    m.add_submodule(&ndutils_module)?;

    let linalg_module = PyModule::new(m.py(), "linalg")?;
    linalg::register_module(&linalg_module)?;
    m.add_submodule(&linalg_module)?;

    let stats_module = PyModule::new(m.py(), "stats")?;
    stats::register_module(&stats_module)?;
    m.add_submodule(&stats_module)?;

    let random_module = PyModule::new(m.py(), "random")?;
    random::register_module(&random_module)?;
    m.add_submodule(&random_module)?;

    let spatial_module = PyModule::new(m.py(), "spatial")?;
    spatial::register_module(&spatial_module)?;
    m.add_submodule(&spatial_module)?;

    tree_engine::register_classes(m)?;

    Ok(())
}
