use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::array::NdArray;
use super::{PyArray, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(eig, m)?)?;
    m.add_function(wrap_pyfunction!(eig_with_params, m)?)?;
    m.add_function(wrap_pyfunction!(eigvals, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(lstsq, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_lstsq, m)?)?;
    Ok(())
}

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

#[pyfunction]
fn outer(a: ArrayLike, b: ArrayLike) -> PyArray {
    let a_arr = a.into_ndarray().unwrap();
    let b_arr = b.into_ndarray().unwrap();
    PyArray {
        inner: NdArray::outer(&a_arr, &b_arr),
    }
}

#[pyfunction]
fn lstsq(a: &PyArray, b: &PyArray) -> PyResult<(PyArray, PyArray)> {
    let x = a.inner.least_squares(&b.inner)
        .map_err(|e| PyValueError::new_err(e))?;

    let ax = a.inner.matmul(&x);
    let residuals = &b.inner - &ax;

    Ok((PyArray { inner: x }, PyArray { inner: residuals }))
}

#[pyfunction]
fn weighted_lstsq(a: &PyArray, b: &PyArray, weights: &PyArray) -> PyResult<(PyArray, PyArray)> {
    let x = a.inner.weighted_least_squares(&b.inner, &weights.inner)
        .map_err(|e| PyValueError::new_err(e))?;

    let ax = a.inner.matmul(&x);
    let diff = &b.inner - &ax;

    let sqrt_w: Vec<f64> = weights.inner.as_slice()
        .iter()
        .map(|&w| w.sqrt())
        .collect();

    let m = diff.shape().dims()[0];
    let weighted_res = if diff.ndim() == 1 {
        let res: Vec<f64> = (0..m)
            .map(|i| diff.get(&[i]).unwrap() * sqrt_w[i])
            .collect();
        NdArray::from_vec(crate::array::shape::Shape::d1(m), res)
    } else {
        let b_cols = diff.shape().dims()[1];
        let mut res = vec![0.0; m * b_cols];
        for i in 0..m {
            for j in 0..b_cols {
                res[i * b_cols + j] = diff.get(&[i, j]).unwrap() * sqrt_w[i];
            }
        }
        NdArray::from_vec(crate::array::shape::Shape::d2(m, b_cols), res)
    };

    Ok((PyArray { inner: x }, PyArray { inner: weighted_res }))
}
