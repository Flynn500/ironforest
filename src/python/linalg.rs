use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::array::NdArray;
use super::{PyArray, ArrayData, ArrayLike};

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
fn matmul(a: ArrayLike, b: ArrayLike) -> PyResult<PyArray> {
    Ok(PyArray { inner: ArrayData::Float(a.into_ndarray()?.matmul(&b.into_ndarray()?)), alive: true })
}

#[pyfunction]
fn dot(a: ArrayLike, b: ArrayLike) -> PyResult<PyArray> {
    Ok(PyArray { inner: ArrayData::Float(a.into_ndarray()?.dot(&b.into_ndarray()?)), alive: true })
}

#[pyfunction]
fn transpose(a: ArrayLike) -> PyResult<PyArray> {
    Ok(PyArray { inner: ArrayData::Float(a.into_ndarray()?.transpose()), alive: true })
}

#[pyfunction]
fn cholesky(a: ArrayLike) -> PyResult<PyArray> {
    a.into_ndarray()?.cholesky()
        .map(|l| PyArray { inner: ArrayData::Float(l), alive: true })
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn qr(a: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray()?.qr()
        .map(|(q, r)| (PyArray { inner: ArrayData::Float(q), alive: true }, PyArray { inner: ArrayData::Float(r), alive: true }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn eig(a: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray()?.eig()
        .map(|(vals, vecs)| (PyArray { inner: ArrayData::Float(vals), alive: true }, PyArray { inner: ArrayData::Float(vecs), alive: true }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (a, max_iter=1000, tol=1e-10))]
fn eig_with_params(a: ArrayLike, max_iter: usize, tol: f64) -> PyResult<(PyArray, PyArray)> {
    a.into_ndarray()?.eig_with_params(max_iter, tol)
        .map(|(vals, vecs)| (PyArray { inner: ArrayData::Float(vals), alive: true }, PyArray { inner: ArrayData::Float(vecs), alive: true }))
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn eigvals(a: ArrayLike) -> PyResult<PyArray> {
    a.into_ndarray()?.eigvals()
        .map(|vals| PyArray { inner: ArrayData::Float(vals), alive: true })
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(signature = (a, k=None))]
fn diagonal(a: ArrayLike, k: Option<isize>) -> PyResult<PyArray> {
    let arr = a.into_ndarray()?;
    if arr.ndim() != 2 {
        return Err(PyValueError::new_err("diagonal requires a 2D array"));
    }
    Ok(PyArray {
        inner: ArrayData::Float(arr.diagonal(k.unwrap_or(0))),
        alive: true
    })
}

#[pyfunction]
fn outer(a: ArrayLike, b: ArrayLike) -> PyResult<PyArray> {
    let a_arr = a.into_ndarray()?;
    let b_arr = b.into_ndarray()?;
    Ok(PyArray {
        inner: ArrayData::Float(NdArray::outer(&a_arr, &b_arr)),
        alive: true
    })
}

#[pyfunction]
fn lstsq(a: ArrayLike, b: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    let aarr = a.into_ndarray()?;
    let barr = b.into_ndarray()?;
    let x = aarr.least_squares(&barr)
        .map_err(|e| PyValueError::new_err(e))?;

    let ax = aarr.matmul(&x);
    let residuals = &barr - &ax;

    Ok((PyArray { inner: ArrayData::Float(x), alive: true }, PyArray { inner: ArrayData::Float(residuals), alive: true }))
}

#[pyfunction]
fn weighted_lstsq(a: ArrayLike, b: ArrayLike, weights: ArrayLike) -> PyResult<(PyArray, PyArray)> {
    let aarr = a.into_ndarray()?;
    let barr = b.into_ndarray()?;
    let warr = weights.into_ndarray()?;
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

    Ok((PyArray { inner: ArrayData::Float(x), alive: true }, PyArray { inner: ArrayData::Float(weighted_res), alive: true }))
}
