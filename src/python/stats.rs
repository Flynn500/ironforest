use pyo3::prelude::*;
use pyo3::types::PyAny;
use super::{PyArray, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?; //to avoid conflict added _dev
    m.add_function(wrap_pyfunction!(median, m)?)?;
    m.add_function(wrap_pyfunction!(quantile, m)?)?;
    m.add_function(wrap_pyfunction!(any, m)?)?;
    m.add_function(wrap_pyfunction!(all, m)?)?;
    m.add_function(wrap_pyfunction!(pearson, m)?)?;
    m.add_function(wrap_pyfunction!(spearman, m)?)?;
    Ok(())
}

#[pyfunction]
fn sum(a: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().sum()
}

#[pyfunction]
fn mean(a: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().mean()
}

#[pyfunction]
fn var(a: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().var()
}

#[pyfunction]
#[pyo3(name = "std")]
fn std_dev(a: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().std()
}

#[pyfunction]
fn median(a: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().median()
}

#[pyfunction]
fn quantile(py: Python<'_>, a: &PyArray, q: ArrayLike) -> PyResult<Py<PyAny>> {
    match q {
        ArrayLike::Scalar(q_val) => {
            Ok(a.inner.quantile(q_val).into_pyobject(py)?.into_any().unbind())
        }
        _ => {
            let q_arr = q.into_ndarray()?;
            Ok(PyArray {
                inner: a.inner.quantiles(q_arr.as_slice()),
            }.into_pyobject(py)?.into_any().unbind())
        }
    }
}

#[pyfunction]
fn any(a: ArrayLike) -> bool {
    a.into_ndarray().unwrap().any()
}

#[pyfunction]
fn all(a: ArrayLike) -> bool {
    a.into_ndarray().unwrap().all()
}

#[pyfunction]
fn pearson(a: ArrayLike, b: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().pearson(&b.into_ndarray().unwrap())
}

#[pyfunction]
fn spearman(a: ArrayLike, b: ArrayLike) -> f64 {
    a.into_ndarray().unwrap().spearman(&b.into_ndarray().unwrap())
}
