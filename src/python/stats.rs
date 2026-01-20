use pyo3::prelude::*;
use pyo3::types::PyAny;
use super::{PyArray, ArrayOrScalar};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(pearson, m)?)?;
    m.add_function(wrap_pyfunction!(spearman, m)?)?;
    Ok(())
}

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
#[pyo3(name = "std")]
fn std_dev(a: &PyArray) -> f64 {
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
