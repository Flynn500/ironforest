use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use pyo3::{FromPyObject, Borrowed};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use crate::array::{NdArray, Shape};

pub enum ArrayLike {
    Array(PyArray),
    Numpy { shape: Vec<usize>, data: Vec<f64> },
    Vec(Vec<f64>),
    Vec2D(Vec<Vec<f64>>),
    Scalar(f64),
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

        if let Ok(scalar) = ob.extract::<f64>() {
            return Ok(ArrayLike::Scalar(scalar));
        }

        Err(PyValueError::new_err(
            "Expected Array, NumPy array, list, or scalar"
        ))
    }
}

impl ArrayLike {
    pub fn into_ndarray(self) -> PyResult<NdArray<f64>> {
        match self {
            ArrayLike::Array(arr) => Ok(arr.inner),
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::new(vec![1]), vec![s])),
            ArrayLike::Vec(v) => Ok(NdArray::from_vec(Shape::d1(v.len()), v)),
            ArrayLike::Vec2D(v) => {
                if v.is_empty() {
                    return Err(PyValueError::new_err("Cannot create array from empty nested list"));
                }
                let rows = v.len();
                let cols = v[0].len();
                for row in &v {
                    if row.len() != cols {
                        return Err(PyValueError::new_err("Nested lists must have consistent dimensions"));
                    }
                }
                let flat: Vec<f64> = v.into_iter().flatten().collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), flat))
            }
            ArrayLike::Numpy { shape, data } => {
                Ok(NdArray::from_vec(Shape::new(shape), data))
            }
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
}

// Main PyArray wrapper
#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    pub inner: NdArray<f64>,
}

// Submodules
pub mod array;
pub mod linalg;
pub mod stats;
pub mod random;
pub mod spatial;

// Re-export for convenience
pub use array::PyArrayIter;
pub use random::PyGenerator;

// Main Python module
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register main types
    m.add_class::<PyArray>()?;
    m.add_class::<PyArrayIter>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<spatial::PyBallTree>()?;
    m.add_class::<spatial::PyKDTree>()?;

    // Top-level array functions
    m.add_function(wrap_pyfunction!(array::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(array::ones, m)?)?;
    m.add_function(wrap_pyfunction!(array::full, m)?)?;
    m.add_function(wrap_pyfunction!(array::asarray, m)?)?;
    m.add_function(wrap_pyfunction!(array::eye, m)?)?;
    m.add_function(wrap_pyfunction!(array::diag, m)?)?;
    m.add_function(wrap_pyfunction!(array::column_stack, m)?)?;

    // Submodules
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

    Ok(())
}
