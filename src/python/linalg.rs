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
fn matmul(a: ArrayLike, b: ArrayLike) -> PyArray {
    PyArray { inner: a.into_ndarray().unwrap().matmul(&b.into_ndarray().unwrap()) }
}

#[pyfunction]
fn dot(a: ArrayLike, b: ArrayLike) -> PyArray {
    PyArray { inner: a.into_ndarray().unwrap().dot(&b.into_ndarray().unwrap()) }
}

#[pyfunction]
fn transpose(a: ArrayLike) -> PyArray {
    PyArray { inner: a.into_ndarray().unwrap().transpose() }
}

#[pyfunction]
fn cholesky(a: ArrayLike) -> PyResult<PyArray> {
    a.into_ndarray().unwrap().cholesky()
        .map(|l| PyArray { inner: l })
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn qr(a: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray().unwrap().qr()
        .map(|(q, r)| (PyArray { inner: q }, PyArray { inner: r }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn eig(a: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray().unwrap().eig()
        .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (a, max_iter=1000, tol=1e-10))]
fn eig_with_params(a: ArrayLike, max_iter: usize, tol: f64) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray().unwrap().eig_with_params(max_iter, tol)
        .map(|(vals, vecs)| (PyArray { inner: vals }, PyArray { inner: vecs }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn eigvals(a: ArrayLike) -> PyResult<PyArray> {
    a.into_ndarray().unwrap().eigvals()
        .map(|vals| PyArray { inner: vals })
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (a, k=None))]
fn diagonal(a: ArrayLike, k: Option<isize>) -> PyResult<PyArray> {
    let arr = a.into_ndarray().unwrap();
    if arr.ndim() != 2 {
        return Err(PyValueError::new_err("diagonal requires a 2D array"));
    }
    Ok(PyArray {
        inner: arr.diagonal(k.unwrap_or(0)),
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
fn lstsq(a: ArrayLike, b: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    let aarr = a.into_ndarray().unwrap();
    let barr = b.into_ndarray().unwrap();
    let x = aarr.least_squares(&barr)
        .map_err(|e| PyValueError::new_err(e))?;

    let ax = aarr.matmul(&x);
    let residuals = &barr - &ax;

    Ok((PyArray { inner: x }, PyArray { inner: residuals }))
}

#[pyfunction]
fn weighted_lstsq(a: ArrayLike, b: ArrayLike, weights: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    let aarr = a.into_ndarray().unwrap();
    let barr = b.into_ndarray().unwrap();
    let warr = weights.into_ndarray().unwrap();
    let x = aarr.weighted_least_squares(&barr, &warr)
        .map_err(|e| PyValueError::new_err(e))?;

    let ax = aarr.matmul(&x);
    let diff = &barr - &ax;

    let sqrt_w: Vec<f64> = warr.as_slice()
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
