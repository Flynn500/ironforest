use pyo3::{exceptions::PyTypeError, prelude::*};
use pyo3::types::PyAny;
use pyo3::exceptions::PyValueError;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayData, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(outer, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(asarray, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(diag, m)?)?;
    m.add_function(wrap_pyfunction!(column_stack, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::zeros(shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::ones(shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None))]
pub fn full(shape: Vec<usize>, fill_value: f64, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::full(shape, fill_value, dtype)
}

#[pyfunction]
#[pyo3(signature = (data, shape=None, dtype=None))]
pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::asarray(data, shape, dtype)
}

#[pyfunction]
#[pyo3(signature = (n, m=None, k=None, dtype=None))]
pub fn eye(n: usize, m: Option<usize>, k: Option<isize>, dtype: Option<&str>) -> PyResult<PyArray> {
    PyArray::eye(n, m, k, dtype)
}

#[pyfunction]
#[pyo3(signature = (v, k=None))]
pub fn diag(v: ArrayLike, k: Option<isize>) -> PyResult<PyArray> {
    PyArray::diag(v, k)
}

#[pyfunction]
pub fn column_stack(arrays: Vec<PyArray>) -> PyResult<PyArray> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("Need at least one array"));
    }
    let float_arrays: Vec<NdArray<f64>> = arrays.iter()
        .map(|a| a.as_float().map(|x| x.clone()))
        .collect::<PyResult<_>>()?;
    let array_refs: Vec<&NdArray<f64>> = float_arrays.iter().collect();
    Ok(PyArray {
        inner: ArrayData::Float(NdArray::column_stack(&array_refs)),
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
fn linspace(_py: Python<'_>, start: ArrayLike, stop: ArrayLike, num: i64) -> PyResult<PyArray> {
    let start_arr = start.into_ndarray().unwrap();
    let stop_arr = stop.into_ndarray().unwrap();

    if num < 2 {
        return Err(PyValueError::new_err("num must be >= 2"));
    }

    let broadcast_shape = start_arr.shape().broadcast(stop_arr.shape())
        .ok_or_else(|| PyValueError::new_err(
            "start and stop shapes are not broadcastable"
        ))?;

    let start_b = start_arr.broadcast_to(&broadcast_shape).unwrap();
    let stop_b = stop_arr.broadcast_to(&broadcast_shape).unwrap();

    let steps = (num - 1) as f64;
    let total_elements = broadcast_shape.size();
    let mut data = Vec::with_capacity(total_elements * num as usize);

    for j in 0..num as usize {
        for i in 0..total_elements {
            let s = start_b.as_slice_unchecked()[i];
            let e = stop_b.as_slice_unchecked()[i];
            data.push(s + (e - s) * (j as f64 / steps));
        }
    }

    let out_shape = Shape::new({
        let mut dims = vec![num as usize];
        dims.extend(broadcast_shape.dims());
        dims
    });
    let our_arr = NdArray::from_vec(out_shape, data);
    Ok(PyArray { inner: ArrayData::Float(our_arr), alive: true })
}

#[pyfunction]
fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let dtype = arr.getattr("dtype")?;
    let dtype_str: String = dtype.getattr("kind")?.extract()?;

    match dtype_str.as_str() {
        "f" => {
            let numpy_arr = arr.cast::<PyArrayDyn<f64>>()
                .map_err(|_| PyTypeError::new_err("expected float64 numpy array"))?;
            let readonly = numpy_arr.readonly();
            let shape = readonly.shape().to_vec();

            let inner = if readonly.is_c_contiguous() {
                let slice = readonly.as_slice()?;
                let owner = arr.clone().unbind();
                unsafe { NdArray::from_external(owner, slice.as_ptr(), Shape::new(shape)) }
            } else {
                let data: Vec<f64> = readonly.as_array().iter().copied().collect();
                NdArray::from_vec(Shape::new(shape), data)
            };

            Ok(PyArray { inner: ArrayData::Float(inner), alive: true })
        }
        "i" | "u" => {
            let numpy_arr = arr.cast::<PyArrayDyn<i64>>()
                .map_err(|_| PyTypeError::new_err("expected int64 numpy array"))?;
            let readonly = numpy_arr.readonly();
            let shape = readonly.shape().to_vec();

            let inner = if readonly.is_c_contiguous() {
                let slice = readonly.as_slice()?;
                let owner = arr.clone().unbind();
                unsafe { NdArray::from_external(owner, slice.as_ptr(), Shape::new(shape)) }
            } else {
                let data: Vec<i64> = readonly.as_array().iter().copied().collect();
                NdArray::from_vec(Shape::new(shape), data)
            };

            Ok(PyArray { inner: ArrayData::Int(inner), alive: true })
        }
        other => Err(PyTypeError::new_err(format!(
            "unsupported numpy dtype kind: '{other}'"
        ))),
    }
}

#[pyfunction]
fn to_numpy(py: Python<'_>, arr: &PyArray) -> PyResult<Py<PyAny>> {
    arr.to_numpy(py)
}
