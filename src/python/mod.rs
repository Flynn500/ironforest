use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::array::{NdArray, Shape};

#[derive(FromPyObject)]
pub enum ArrayOrScalar {
    Array(PyArray),
    Scalar(f64),
}

#[derive(FromPyObject)]
pub enum VecOrArray {
    Array(PyArray),
    Vec(Vec<f64>),
}

impl VecOrArray {
    pub fn into_vec(self) -> Vec<f64> {
        match self {
            VecOrArray::Array(arr) => arr.inner.as_slice().to_vec(),
            VecOrArray::Vec(v) => v,
        }
    }

    pub fn into_ndarray(self) -> NdArray<f64> {
        match self {
            VecOrArray::Array(arr) => arr.inner,
            VecOrArray::Vec(v) => {
                let len = v.len();
                NdArray::from_vec(Shape::d1(len), v)
            }
        }
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
fn substratum(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
