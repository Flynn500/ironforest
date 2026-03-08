use pyo3::prelude::*;
use pyo3::types::PyAny;
use super::{PyArray, ArrayData, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?; //to avoid conflict added _dev
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(pearson, m)?)?;
    m.add_function(wrap_pyfunction!(spearman, m)?)?;
    Ok(())
}

#[pyfunction]
fn sum(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.sum())
}

#[pyfunction]
fn mean(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.mean())
}

#[pyfunction]
fn var(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.var())
}

#[pyfunction]
#[pyo3(name = "std")]
fn std_dev(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.std())
}

#[pyfunction]
fn median(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.median())
}

#[pyfunction]
fn max(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.max())
}

#[pyfunction]
fn min(a: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.min())
}

#[pyfunction]
fn quantile(py: Python<'_>, a: &PyArray, q: ArrayLike) -> PyResult<Py<PyAny>> {
    let arr = a.as_float()?;
    match q {
        ArrayLike::Scalar(q_val) => {
            Ok(arr.quantile(q_val).into_pyobject(py)?.into_any().unbind())
        }
        ArrayLike::IntScalar(s) => {
            Ok(arr.quantile(s as f64).into_pyobject(py)?.into_any().unbind())
        }
        _ => {
            let q_arr = q.into_ndarray()?;
            Ok(PyArray {
                inner: ArrayData::Float(arr.quantiles(q_arr.as_slice())),
                alive: true
            }.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pyfunction]
fn any(a: ArrayLike) -> PyResult<bool> {
    Ok(a.into_ndarray()?.any())
}

#[pyfunction]
fn all(a: ArrayLike) -> PyResult<bool> {
    Ok(a.into_ndarray()?.all())
}

#[pyfunction]
fn pearson(a: ArrayLike, b: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.pearson(&b.into_ndarray()?))
}

#[pyfunction]
fn spearman(a: ArrayLike, b: ArrayLike) -> PyResult<f64> {
    Ok(a.into_ndarray()?.spearman(&b.into_ndarray()?))
}
