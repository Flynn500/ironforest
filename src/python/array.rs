use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError};
use pyo3::types::{PyAny, PyFloat, PyList, PySlice, PyTuple};
use numpy::{PyArrayMethods, PyReadonlyArrayDyn};
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayData, ArrayLike, FloatNdArray};

#[derive(PartialEq)]
enum Dtype { Float64, Float32, Int64 }

fn parse_dtype(dtype: Option<&str>) -> PyResult<Dtype> {
    match dtype.unwrap_or("float") {
        "float" | "float64" | "f64" => Ok(Dtype::Float64),
        "float32" | "f32"           => Ok(Dtype::Float32),
        "int"    | "int64"  | "i64" => Ok(Dtype::Int64),
        other => Err(PyValueError::new_err(format!(
            "unsupported dtype '{}'; expected 'float', 'float32', or 'int'",
            other
        ))),
    }
}

// get item enums
enum AxisIndex {
    /// a[3] — selects one element, collapses this axis
    Single(usize),
    /// a[1:5:2] — selects a range, keeps this axis
    Slice {
        start: usize,
        step: isize,
        len: usize,
    },
}

fn parse_axis_index(key: &Bound<'_, PyAny>, axis: usize, dim_size: usize) -> PyResult<AxisIndex> {
    if let Ok(slice) = key.cast::<PySlice>() {
        let indices = slice.indices(dim_size as isize)?;
        let mut len = 0usize;
        let mut i = indices.start;
        while (indices.step > 0 && i < indices.stop) || (indices.step < 0 && i > indices.stop) {
            if i >= 0 && i < dim_size as isize {
                len += 1;
            }
            i += indices.step;
        }
        Ok(AxisIndex::Slice {
            start: indices.start as usize,
            step: indices.step,
            len,
        })
    } else if let Ok(idx) = key.extract::<isize>() {
        let dim = dim_size as isize;
        let normalized = if idx < 0 { dim + idx } else { idx };
        if normalized < 0 || normalized >= dim {
            return Err(PyValueError::new_err(format!(
                "index {} is out of bounds for axis {} with size {}",
                idx, axis, dim_size
            )));
        }
        Ok(AxisIndex::Single(normalized as usize))
    } else {
        Err(PyValueError::new_err(format!(
            "unsupported index type for axis {}", axis
        )))
    }
}

fn expand_axis_indices(axis: &AxisIndex) -> Vec<usize> {
    match axis {
        AxisIndex::Single(idx) => vec![*idx],
        AxisIndex::Slice { start, step, len } => {
            let mut v = Vec::with_capacity(*len);
            let mut i = *start as isize;
            for _ in 0..*len {
                v.push(i as usize);
                i += step;
            }
            v
        }
    }
}



#[pymethods]
impl PyArray {
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    pub fn zeros(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: match parse_dtype(dtype)? {
            Dtype::Int64   => ArrayData::Int(NdArray::zeros(Shape::new(shape))),
            Dtype::Float32 => ArrayData::Float32(NdArray::zeros(Shape::new(shape))),
            Dtype::Float64 => ArrayData::Float(NdArray::zeros(Shape::new(shape))),
        }, alive: true})
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    pub fn ones(shape: Vec<usize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: match parse_dtype(dtype)? {
            Dtype::Int64   => ArrayData::Int(NdArray::<i64>::ones(Shape::new(shape))),
            Dtype::Float32 => ArrayData::Float32(NdArray::<f32>::ones(Shape::new(shape))),
            Dtype::Float64 => ArrayData::Float(NdArray::<f64>::ones(Shape::new(shape))),
        }, alive: true})
    }

    #[staticmethod]
    #[pyo3(signature = (shape, fill_value, dtype=None))]
    pub fn full(shape: Vec<usize>, fill_value: f64, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: match parse_dtype(dtype)? {
            Dtype::Int64   => ArrayData::Int(NdArray::filled(Shape::new(shape), fill_value as i64)),
            Dtype::Float32 => ArrayData::Float32(NdArray::<f32>::full(Shape::new(shape), fill_value as f32)),
            Dtype::Float64 => ArrayData::Float(NdArray::<f64>::full(Shape::new(shape), fill_value)),
        }, alive: true})
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape=None, dtype=None))]
    pub fn asarray(data: ArrayLike, shape: Option<Vec<usize>>, dtype: Option<&str>) -> PyResult<Self> {
        let resolved_dtype = if dtype.is_none() {
            if data.is_int() { Dtype::Int64 }
            else if data.is_f32() { Dtype::Float32 }
            else { Dtype::Float64 }
        } else {
            parse_dtype(dtype)?
        };

        fn reshape_vec<T: Clone>(arr: NdArray<T>, shape: Option<Vec<usize>>) -> PyResult<NdArray<T>> {
            if let Some(s) = shape {
                let expected_size: usize = s.iter().product();
                if arr.len() != expected_size {
                    return Err(PyValueError::new_err(format!(
                        "Data length {} doesn't match shape {:?} (expected {})",
                        arr.len(), s, expected_size
                    )));
                }
                Ok(NdArray::from_vec(Shape::new(s), arr.as_slice_unchecked().to_vec()))
            } else {
                Ok(arr)
            }
        }

        match resolved_dtype {
            Dtype::Int64 => {
                let arr = reshape_vec(data.into_i64_ndarray()?, shape)?;
                Ok(PyArray { inner: ArrayData::Int(arr), alive: true })
            }
            Dtype::Float32 => {
                let arr = reshape_vec(data.into_f32_ndarray()?, shape)?;
                Ok(PyArray { inner: ArrayData::Float32(arr), alive: true })
            }
            Dtype::Float64 => {
                let arr = reshape_vec(data.into_ndarray()?, shape)?;
                Ok(PyArray { inner: ArrayData::Float(arr), alive: true })
            }
        }
    }

    #[staticmethod]
    #[pyo3(signature = (n, m=None, k=None, dtype=None))]
    pub fn eye(n: usize, m: Option<usize>, k: Option<isize>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(PyArray { inner: match parse_dtype(dtype)? {
            Dtype::Int64   => ArrayData::Int(NdArray::<i64>::eye(n, m, k.unwrap_or(0))),
            Dtype::Float32 => ArrayData::Float32(NdArray::<f32>::eye(n, m, k.unwrap_or(0))),
            Dtype::Float64 => ArrayData::Float(NdArray::<f64>::eye(n, m, k.unwrap_or(0))),
        }, alive: true})
    }

    #[staticmethod]
    #[pyo3(signature = (v, k=None))]
    pub fn diag(v: ArrayLike, k: Option<isize>) -> PyResult<Self> {
        match v.into_float_ndarray()? {
            FloatNdArray::F32(arr) => Ok(PyArray { inner: ArrayData::Float32(NdArray::<f32>::from_diag(&arr, k.unwrap_or(0))), alive: true }),
            FloatNdArray::F64(arr) => Ok(PyArray { inner: ArrayData::Float(NdArray::<f64>::from_diag(&arr, k.unwrap_or(0))), alive: true }),
        }
    }

    #[pyo3(signature = (k=None))]
    fn diagonal(&self, k: Option<isize>) -> PyResult<PyArray> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                if a.ndim() != 2 {
                    return Err(PyValueError::new_err("diagonal requires a 2D array"));
                }
                Ok(PyArray { inner: ArrayData::Float(a.diagonal(k.unwrap_or(0))), alive: true })
            }
            ArrayData::Float32(a) => {
                if a.ndim() != 2 {
                    return Err(PyValueError::new_err("diagonal requires a 2D array"));
                }
                Ok(PyArray { inner: ArrayData::Float32(a.diagonal(k.unwrap_or(0))), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("diagonal not supported for integer arrays")),
        }
    }

    #[new]
    #[pyo3(signature = (shape, data, dtype=None))]
    fn new(shape: Vec<usize>, data: Vec<f64>, dtype: Option<&str>) -> PyResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(PyValueError::new_err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected_size
            )));
        }
        Ok(PyArray { inner: match parse_dtype(dtype)? {
            Dtype::Int64   => ArrayData::Int(NdArray::from_vec(Shape::new(shape), data.iter().map(|&x| x as i64).collect())),
            Dtype::Float32 => ArrayData::Float32(NdArray::from_vec(Shape::new(shape), data.iter().map(|&x| x as f32).collect())),
            Dtype::Float64 => ArrayData::Float(NdArray::from_vec(Shape::new(shape), data)),
        }, alive: true })
    }

    // -------------------------------------------------------------------------
    // Shape & Properties
    // -------------------------------------------------------------------------

    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        check_alive!(self);
        Ok(self.dims())
    }

    #[getter]
    fn ndim(&self) -> PyResult<usize> {
        check_alive!(self);
        Ok(self.ndim_val())
    }

    #[getter]
    fn dtype(&self) -> PyResult<&'static str> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(_)   => Ok("float64"),
            ArrayData::Float32(_) => Ok("float32"),
            ArrayData::Int(_)     => Ok("int64"),
        }
    }

    #[getter]
    fn alive(&self) -> PyResult<bool> {
        Ok(self.alive)
    }

    fn get(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => a.get(&indices).copied()
                .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
                .and_then(|v| Ok(v.into_pyobject(py)?.into_any().unbind())),
            ArrayData::Float32(a) => a.get(&indices).copied()
                .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
                .and_then(|v| Ok((v as f64).into_pyobject(py)?.into_any().unbind())),
            ArrayData::Int(a) => a.get(&indices).copied()
                .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
                .and_then(|v| Ok(v.into_pyobject(py)?.into_any().unbind())),
        }
    }

    // -------------------------------------------------------------------------
    // Conversion & Display
    // -------------------------------------------------------------------------

    pub fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                if a.shape().ndim() == 0 {
                    return Ok(PyFloat::new(py, a.as_slice_unchecked()[0]).unbind().into_any());
                }
                self.to_pylist_recursive(a, py, 0, 0)
            }
            ArrayData::Float32(a) => {
                if a.shape().ndim() == 0 {
                    return Ok(PyFloat::new(py, a.as_slice_unchecked()[0] as f64).unbind().into_any());
                }
                // convert to f64 for Python list (f32 doesn't impl IntoPyObject directly)
                let f64_arr: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                self.to_pylist_recursive(&f64_arr, py, 0, 0)
            }
            ArrayData::Int(a) => {
                if a.shape().ndim() == 0 {
                    return Ok(a.as_slice_unchecked()[0].into_pyobject(py)?.into_any().unbind());
                }
                self.to_pylist_recursive(a, py, 0, 0)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Math & Reduction
    // -------------------------------------------------------------------------

    fn sin(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.sin()), alive: true }),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                let result: Vec<f32> = f64a.sin().as_slice_unchecked().iter().map(|&x| x as f32).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("sin not supported for integer arrays")),
        }
    }

    fn cos(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.cos()), alive: true }),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                let result: Vec<f32> = f64a.cos().as_slice_unchecked().iter().map(|&x| x as f32).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("cos not supported for integer arrays")),
        }
    }

    fn exp(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.exp()), alive: true }),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                let result: Vec<f32> = f64a.exp().as_slice_unchecked().iter().map(|&x| x as f32).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("exp not supported for integer arrays")),
        }
    }

    fn sqrt(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.sqrt()), alive: true }),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                let result: Vec<f32> = f64a.sqrt().as_slice_unchecked().iter().map(|&x| x as f32).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("sqrt not supported for integer arrays")),
        }
    }

    fn clip(&self, min: f64, max: f64) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.clip(min, max)), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.clamp(min as f32, max as f32)).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("clip not supported for integer arrays")),
        }
    }

    fn tan(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.tan()), alive: true }),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                let result: Vec<f32> = f64a.tan().as_slice_unchecked().iter().map(|&x| x as f32).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("tan not supported for integer arrays")),
        }
    }

    fn arcsin(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.asin()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.asin()).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("arcsin not supported for integer arrays")),
        }
    }

    fn arccos(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.acos()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.acos()).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("arccos not supported for integer arrays")),
        }
    }

    fn arctan(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.atan()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.atan()).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("arctan not supported for integer arrays")),
        }
    }

    fn log(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.log()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.ln()).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("log not supported for integer arrays")),
        }
    }

    fn abs(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.abs()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| x.abs()).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("abs not supported for integer arrays")),
        }
    }

    fn sign(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.sign()), alive: true }),
            ArrayData::Float32(a) => {
                let result: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), result)), alive: true })
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("sign not supported for integer arrays")),
        }
    }

    fn sum(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.sum()),
            ArrayData::Float32(a) => Ok(a.as_slice_unchecked().iter().map(|&x| x as f64).sum()),
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("sum not supported for integer arrays")),
        }
    }

    fn mean(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.mean()),
            ArrayData::Float32(a) => {
                let s: f64 = a.as_slice_unchecked().iter().map(|&x| x as f64).sum();
                Ok(s / a.len() as f64)
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("mean not supported for integer arrays")),
        }
    }

    fn mode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.mode().into_pyobject(py)?.into_any().unbind()),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                Ok(f64a.mode().into_pyobject(py)?.into_any().unbind())
            }
            ArrayData::Int(a) => Ok(a.mode().into_pyobject(py)?.into_any().unbind()),
        }
    }

    fn var(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.var()),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                Ok(f64a.var())
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("var not supported for integer arrays")),
        }
    }

    fn std(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.std()),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                Ok(f64a.std())
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("std not supported for integer arrays")),
        }
    }

    fn median(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.median()),
            ArrayData::Float32(a) => {
                let f64a: NdArray<f64> = NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                Ok(f64a.median())
            }
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("median not supported for integer arrays")),
        }
    }

    fn max(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.max()),
            ArrayData::Float32(a) => Ok(a.as_slice_unchecked().iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64),
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("max not supported for integer arrays")),
        }
    }

    fn min(&self) -> PyResult<f64> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.min()),
            ArrayData::Float32(a) => Ok(a.as_slice_unchecked().iter().cloned().fold(f32::INFINITY, f32::min) as f64),
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("min not supported for integer arrays")),
        }
    }

    fn quantile(&self, py: Python<'_>, q: ArrayLike) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        let a = match &self.inner {
            ArrayData::Float(a) => a.clone(),
            ArrayData::Float32(a) => NdArray::from_vec(a.shape().clone(), a.as_slice_unchecked().iter().map(|&x| x as f64).collect()),
            ArrayData::Int(_) => return Err(pyo3::exceptions::PyTypeError::new_err("quantile not supported for integer arrays")),
        };
        match q {
            ArrayLike::Scalar(q_val) => {
                Ok(a.quantile(q_val).into_pyobject(py)?.into_any().unbind())
            }
            ArrayLike::IntScalar(s) => {
                Ok(a.quantile(s as f64).into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                let q_arr = q.into_ndarray()?;
                Ok(PyArray {
                    inner: ArrayData::Float(a.quantiles(q_arr.as_slice_unchecked())), alive: true
                }.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    fn any(&self) -> PyResult<bool> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.any()),
            ArrayData::Float32(a) => Ok(a.as_slice_unchecked().iter().any(|&x| x != 0.0)),
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("any not supported for integer arrays")),
        }
    }

    fn all(&self) -> PyResult<bool> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(a.all()),
            ArrayData::Float32(a) => Ok(a.as_slice_unchecked().iter().all(|&x| x != 0.0)),
            ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err("all not supported for integer arrays")),
        }
    }

    // -------------------------------------------------------------------------
    // Arithmetic ops
    // -------------------------------------------------------------------------

    fn __add__(&self, other: ArrayLike) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs + s), alive: true }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs + (s as f64)), alive: true }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs + &other.into_ndarray()?), alive: true }),
            },
            ArrayData::Float32(lhs) => {
                if other.is_f32() {
                    let rhs = other.into_f32_ndarray()?;
                    let data: Vec<f32> = lhs.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&a, &b)| a + b).collect();
                    Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(lhs.shape().clone(), data)), alive: true })
                } else {
                    let lhs_f64 = NdArray::<f64>::from_vec(lhs.shape().clone(), lhs.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 + s), alive: true }),
                        ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 + (s as f64)), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f64 + &other.into_ndarray()?), alive: true }),
                    }
                }
            }
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs + &other.into_i64_ndarray()?), alive: true })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f + s), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f + &other.into_ndarray()?), alive: true }),
                    }
                }
            }
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s + a), alive: true })
            }
            ArrayData::Float32(a) => {
                let s: f32 = other.extract::<f64>()? as f32;
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| s + x).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s + a.clone()), alive: true })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s + &af), alive: true })
                }
            }
        }
    }

    fn __sub__(&self, other: ArrayLike) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs - s), alive: true }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs - (s as f64)), alive: true }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs - &other.into_ndarray()?), alive: true }),
            },
            ArrayData::Float32(lhs) => {
                if other.is_f32() {
                    let rhs = other.into_f32_ndarray()?;
                    let data: Vec<f32> = lhs.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&a, &b)| a - b).collect();
                    Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(lhs.shape().clone(), data)), alive: true })
                } else {
                    let lhs_f64 = NdArray::<f64>::from_vec(lhs.shape().clone(), lhs.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 - s), alive: true }),
                        ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 - (s as f64)), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f64 - &other.into_ndarray()?), alive: true }),
                    }
                }
            }
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs - &other.into_i64_ndarray()?), alive: true })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f - s), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f - &other.into_ndarray()?), alive: true }),
                    }
                }
            }
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s - a), alive: true })
            }
            ArrayData::Float32(a) => {
                let s: f32 = other.extract::<f64>()? as f32;
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| s - x).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s - a.clone()), alive: true })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s - &af), alive: true})
                }
            }
        }
    }

    fn __mul__(&self, other: ArrayLike) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs * s), alive: true }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs * (s as f64)), alive: true }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs * &other.into_ndarray()?), alive: true }),
            },
            ArrayData::Float32(lhs) => {
                if other.is_f32() {
                    let rhs = other.into_f32_ndarray()?;
                    let data: Vec<f32> = lhs.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&a, &b)| a * b).collect();
                    Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(lhs.shape().clone(), data)), alive: true })
                } else {
                    let lhs_f64 = NdArray::<f64>::from_vec(lhs.shape().clone(), lhs.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 * s), alive: true }),
                        ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 * (s as f64)), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f64 * &other.into_ndarray()?), alive: true }),
                    }
                }
            }
            ArrayData::Int(lhs) => {
                if other.is_int() {
                    Ok(PyArray { inner: ArrayData::Int(lhs * &other.into_i64_ndarray()?), alive: true })
                } else {
                    let lhs_f = lhs.map(|x| x as f64);
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f * s), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f * &other.into_ndarray()?), alive: true }),
                    }
                }
            }
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s * a), alive: true })
            }
            ArrayData::Float32(a) => {
                let s: f32 = other.extract::<f64>()? as f32;
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| s * x).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                if let Ok(s) = other.extract::<i64>() {
                    Ok(PyArray { inner: ArrayData::Int(s * a.clone()), alive: true })
                } else {
                    let s: f64 = other.extract()?;
                    let af = a.map(|x| x as f64);
                    Ok(PyArray { inner: ArrayData::Float(s * &af), alive: true })
                }
            }
        }
    }

    fn __truediv__(&self, other: ArrayLike) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(lhs) => match other {
                ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs / s), alive: true }),
                ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs / (s as f64)), alive: true }),
                _ => Ok(PyArray { inner: ArrayData::Float(lhs / &other.into_ndarray()?), alive: true }),
            },
            ArrayData::Float32(lhs) => {
                if other.is_f32() {
                    let rhs = other.into_f32_ndarray()?;
                    let data: Vec<f32> = lhs.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&a, &b)| a / b).collect();
                    Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(lhs.shape().clone(), data)), alive: true })
                } else {
                    let lhs_f64 = NdArray::<f64>::from_vec(lhs.shape().clone(), lhs.as_slice_unchecked().iter().map(|&x| x as f64).collect());
                    match other {
                        ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 / s), alive: true }),
                        ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f64 / (s as f64)), alive: true }),
                        _ => Ok(PyArray { inner: ArrayData::Float(lhs_f64 / &other.into_ndarray()?), alive: true }),
                    }
                }
            }
            ArrayData::Int(lhs) => {
                let lhs_f = lhs.map(|x| x as f64);
                match other {
                    ArrayLike::Scalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f / s), alive: true }),
                    ArrayLike::IntScalar(s) => Ok(PyArray { inner: ArrayData::Float(lhs_f / (s as f64)), alive: true }),
                    _ => Ok(PyArray { inner: ArrayData::Float(lhs_f / &other.into_ndarray()?), alive: true }),
                }
            }
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let s: f64 = other.extract()?;
                Ok(PyArray { inner: ArrayData::Float(s / a), alive: true })
            }
            ArrayData::Float32(a) => {
                let s: f32 = other.extract::<f64>()? as f32;
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| s / x).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                let s: f64 = other.extract()?;
                let af = a.map(|x| x as f64);
                Ok(PyArray { inner: ArrayData::Float(s / &af), alive: true })
            }
        }
    }

    fn __neg__(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(-a), alive: true }),
            ArrayData::Float32(a) => {
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| -x).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => Ok(PyArray { inner: ArrayData::Int(-a), alive: true }),
        }
    }

    // -------------------------------------------------------------------------
    // Comparison
    // -------------------------------------------------------------------------

    fn __richcmp__(&self, other: ArrayLike, op: pyo3::basic::CompareOp) -> PyResult<PyArray> {
        match op {
            pyo3::basic::CompareOp::Lt => self.apply_cmp(other, |a, b| a < b),
            pyo3::basic::CompareOp::Le => self.apply_cmp(other, |a, b| a <= b),
            pyo3::basic::CompareOp::Gt => self.apply_cmp(other, |a, b| a > b),
            pyo3::basic::CompareOp::Ge => self.apply_cmp(other, |a, b| a >= b),
            pyo3::basic::CompareOp::Eq => self.apply_cmp(other, |a, b| a == b),
            pyo3::basic::CompareOp::Ne => self.apply_cmp(other, |a, b| a != b),
        }
    }

    fn __pow__(&self, exp: ArrayLike, _modulo: Option<i64>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = match exp {
                    ArrayLike::Scalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| x.powf(e)).collect(),
                    ArrayLike::IntScalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| x.powf(e as f64)).collect(),
                    ArrayLike::Array(arr) => {
                        let borrowed = arr.borrow();
                        match &borrowed.inner {
                            ArrayData::Float(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| x.powf(e)).collect(),
                            ArrayData::Float32(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| x.powf(e as f64)).collect(),
                            ArrayData::Int(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| x.powf(e as f64)).collect(),
                        }
                    },
                    _ => {
                        let rhs = exp.into_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&x, &e)| x.powf(e)).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Float32(a) => {
                let data: Vec<f32> = match exp {
                    ArrayLike::Scalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| x.powf(e as f32)).collect(),
                    ArrayLike::IntScalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| x.powf(e as f32)).collect(),
                    _ => {
                        let rhs = exp.into_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&x, &e)| x.powf(e as f32)).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                let data: Vec<f64> = match exp {
                    ArrayLike::Scalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| (x as f64).powf(e)).collect(),
                    ArrayLike::IntScalar(e) =>
                        a.as_slice_unchecked().iter().map(|&x| (x as f64).powf(e as f64)).collect(),
                    ArrayLike::Array(arr) => {
                        let borrowed = arr.borrow();
                        match &borrowed.inner {
                            ArrayData::Float(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| (x as f64).powf(e)).collect(),
                            ArrayData::Float32(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| (x as f64).powf(e as f64)).collect(),
                            ArrayData::Int(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter()).map(|(&x, &e)| (x as f64).powf(e as f64)).collect(),
                        }
                    },
                    _ => {
                        let rhs = exp.into_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter()).map(|(&x, &e)| (x as f64).powf(e)).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
        }
    }

    fn __rpow__(&self, base: &Bound<'_, PyAny>, _modulo: Option<i64>) -> PyResult<Self> {
        check_alive!(self);
        let b = if let Ok(v) = base.extract::<f64>() { v }
                else { base.extract::<i64>()? as f64 };
        match &self.inner {
            ArrayData::Float(a) =>
                Ok(PyArray { inner: ArrayData::Float(a.map(|x| b.powf(x))), alive: true }),
            ArrayData::Float32(a) => {
                let data: Vec<f32> = a.as_slice_unchecked().iter().map(|&x| (b as f32).powf(x)).collect();
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                let data: Vec<f64> = a.as_slice_unchecked().iter().map(|&x| b.powf(x as f64)).collect();
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
        }
    }

    fn __matmul__(&self, other: &PyArray) -> PyResult<Self> {
        check_alive!(self);
        match (&self.inner, &other.inner) {
            (ArrayData::Float32(a), ArrayData::Float32(b)) =>
                Ok(PyArray { inner: ArrayData::Float32(a.matmul(b)), alive: true }),
            _ => Ok(PyArray { inner: ArrayData::Float(self.as_float()?.matmul(other.as_float()?)), alive: true }),
        }
    }

    fn matmul(&self, other: &PyArray) -> PyResult<Self> {
        check_alive!(self);
        match (&self.inner, &other.inner) {
            (ArrayData::Float32(a), ArrayData::Float32(b)) =>
                Ok(PyArray { inner: ArrayData::Float32(a.matmul(b)), alive: true }),
            _ => Ok(PyArray { inner: ArrayData::Float(self.as_float()?.matmul(other.as_float()?)), alive: true }),
        }
    }

    fn dot(&self, other: &PyArray) -> PyResult<Self> {
        check_alive!(self);
        match (&self.inner, &other.inner) {
            (ArrayData::Float32(a), ArrayData::Float32(b)) =>
                Ok(PyArray { inner: ArrayData::Float32(a.dot(b)), alive: true }),
            _ => Ok(PyArray { inner: ArrayData::Float(self.as_float()?.dot(other.as_float()?)), alive: true }),
        }
    }

    fn transpose(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.transpose()), alive: true }),
            ArrayData::Float32(a) => Ok(PyArray { inner: ArrayData::Float32(a.transpose()), alive: true }),
            ArrayData::Int(a) => Ok(PyArray { inner: ArrayData::Int(a.transpose()), alive: true }),
        }
    }

    fn t(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.t()), alive: true }),
            ArrayData::Float32(a) => Ok(PyArray { inner: ArrayData::Float32(a.t()), alive: true }),
            ArrayData::Int(a) => Ok(PyArray { inner: ArrayData::Int(a.t()), alive: true }),
        }
    }

    fn ravel(&self) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.ravel()), alive: true }),
            ArrayData::Float32(a) => Ok(PyArray { inner: ArrayData::Float32(a.ravel()), alive: true }),
            ArrayData::Int(a) => Ok(PyArray { inner: ArrayData::Int(a.ravel()), alive: true }),
        }
    }

    fn take(&self, indices: Vec<usize>) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(PyArray { inner: ArrayData::Float(a.take(&indices)), alive: true }),
            ArrayData::Float32(a) => Ok(PyArray { inner: ArrayData::Float32(a.take(&indices)), alive: true }),
            ArrayData::Int(a) => Ok(PyArray { inner: ArrayData::Int(a.take(&indices)), alive: true }),
        }
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        check_alive!(self);
        let total: usize = shape.iter().product();
        match &self.inner {
            ArrayData::Float(a) => {
                if a.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Cannot reshape array of size {} into shape {:?}", a.len(), shape)));
                }
                Ok(PyArray { inner: ArrayData::Float(a.reshape(shape)), alive: true })
            }
            ArrayData::Float32(a) => {
                if a.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Cannot reshape array of size {} into shape {:?}", a.len(), shape)));
                }
                Ok(PyArray { inner: ArrayData::Float32(a.reshape(shape)), alive: true })
            }
            ArrayData::Int(a) => {
                if a.len() != total {
                    return Err(PyValueError::new_err(format!(
                        "Cannot reshape array of size {} into shape {:?}", a.len(), shape)));
                }
                Ok(PyArray { inner: ArrayData::Int(a.reshape(shape)), alive: true })
            }
        }
    }

    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                if a.len() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "item() can only be called on arrays with exactly one element, got {} elements",
                        a.len()
                    )));
                }
                Ok(a.item().into_pyobject(py)?.into_any().unbind())
            }
            ArrayData::Float32(a) => {
                if a.len() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "item() can only be called on arrays with exactly one element, got {} elements",
                        a.len()
                    )));
                }
                Ok((a.as_slice_unchecked()[0] as f64).into_pyobject(py)?.into_any().unbind())
            }
            ArrayData::Int(a) => {
                if a.len() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "item() can only be called on arrays with exactly one element, got {} elements",
                        a.len()
                    )));
                }
                Ok(a.item().into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        use numpy::IntoPyArray;
        match &self.inner {
            ArrayData::Float(a) => {
                let shape = a.shape().dims();
                let data = a.as_contiguous_slice().into_owned();
                Ok(data.into_pyarray(py).reshape(shape).unwrap().into_any().unbind())
            }
            ArrayData::Float32(a) => {
                let shape = a.shape().dims();
                let data = a.as_contiguous_slice().into_owned();
                Ok(data.into_pyarray(py).reshape(shape).unwrap().into_any().unbind())
            }
            ArrayData::Int(a) => {
                let shape = a.shape().dims();
                let data = a.as_contiguous_slice().into_owned();
                Ok(data.into_pyarray(py).reshape(shape).unwrap().into_any().unbind())
            }
        }
    }

    fn astype(&self, dtype: &str) -> PyResult<Self> {
        check_alive!(self);
        match parse_dtype(Some(dtype))? {
            Dtype::Float64 => {
                let data: Vec<f64> = match &self.inner {
                    ArrayData::Float(a) => a.as_slice_unchecked().to_vec(),
                    ArrayData::Float32(a) => a.as_slice_unchecked().iter().map(|&x| x as f64).collect(),
                    ArrayData::Int(a) => a.as_slice_unchecked().iter().map(|&x| x as f64).collect(),
                };
                let shape = match &self.inner {
                    ArrayData::Float(a) => a.shape().clone(),
                    ArrayData::Float32(a) => a.shape().clone(),
                    ArrayData::Int(a) => a.shape().clone(),
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(shape, data)), alive: true })
            }
            Dtype::Float32 => {
                let data: Vec<f32> = match &self.inner {
                    ArrayData::Float(a) => a.as_slice_unchecked().iter().map(|&x| x as f32).collect(),
                    ArrayData::Float32(a) => a.as_slice_unchecked().to_vec(),
                    ArrayData::Int(a) => a.as_slice_unchecked().iter().map(|&x| x as f32).collect(),
                };
                let shape = match &self.inner {
                    ArrayData::Float(a) => a.shape().clone(),
                    ArrayData::Float32(a) => a.shape().clone(),
                    ArrayData::Int(a) => a.shape().clone(),
                };
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(shape, data)), alive: true })
            }
            Dtype::Int64 => {
                let data: Vec<i64> = match &self.inner {
                    ArrayData::Float(a) => a.as_slice_unchecked().iter().map(|&x| x as i64).collect(),
                    ArrayData::Float32(a) => a.as_slice_unchecked().iter().map(|&x| x as i64).collect(),
                    ArrayData::Int(a) => a.as_slice_unchecked().to_vec(),
                };
                let shape = match &self.inner {
                    ArrayData::Float(a) => a.shape().clone(),
                    ArrayData::Float32(a) => a.shape().clone(),
                    ArrayData::Int(a) => a.shape().clone(),
                };
                Ok(PyArray { inner: ArrayData::Int(NdArray::from_vec(shape, data)), alive: true })
            }
        }
    }

    #[pyo3(signature = (dtype=None))]
    fn __array__(&self, py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        let arr = self.to_numpy(py)?;

        if let Some(dtype) = dtype {
            let np = py.import("numpy")?;
            return Ok(np.call_method1("asarray", (arr, dtype))?.unbind());
        }
        
        Ok(arr)
    }

    fn __repr__(&self) -> PyResult<String> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => Ok(format!(
                "Array(dtype=float64, shape={:?}, data={:?})",
                a.shape().dims(), a.as_slice_unchecked()
            )),
            ArrayData::Float32(a) => Ok(format!(
                "Array(dtype=float32, shape={:?}, data={:?})",
                a.shape().dims(), a.as_slice_unchecked()
            )),
            ArrayData::Int(a) => Ok(format!(
                "Array(dtype=int64, shape={:?}, data={:?})",
                a.shape().dims(), a.as_slice_unchecked()
            )),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        check_alive!(self);
        Ok(self.len_val())
    }

    // -------------------------------------------------------------------------
    // Indexing
    // -------------------------------------------------------------------------

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        let dims = self.dims();
        let ndim = dims.len();

        // --- Boolean mask indexing ---
        // Python: a[np.array([True, False, True])] or a[[True, False, True]]
        if let Ok(bool_mask) = key.extract::<Vec<bool>>() {
            return self.apply_bool_mask_py(&bool_mask, py);
        }
        if let Ok(np_bool) = key.extract::<PyReadonlyArrayDyn<bool>>() {
            let mask: Vec<bool> = np_bool.as_slice()?.iter().copied().collect();
            return self.apply_bool_mask_py(&mask, py);
        }

        // --- Integer index array (custom PyArray) ---
        // Python: a[pyarray_of_ints]
        if let Ok(py_arr) = key.extract::<PyRef<PyArray>>() {
            return match &py_arr.inner {
                ArrayData::Int(idx_arr) => {
                    let indices: Vec<isize> = idx_arr.as_slice_unchecked().iter().map(|&x| x as isize).collect();
                    self.apply_int_index_array_py(&indices, py)
                }
                ArrayData::Float(_) | ArrayData::Float32(_) => Err(PyValueError::new_err(
                    "index arrays must be integer type"
                )),
            };
        }

        // --- Unified path: normalize key into Vec<AxisIndex> ---
        // Handles:
        //   a[3]          — single int
        //   a[1:5]        — single slice
        //   a[1, 2]       — tuple of ints
        //   a[1, 2:5]     — tuple with mixed int/slice
        //   a[:, 3]       — tuple with slice and int
        //   a[0:2, 1:3]   — tuple of slices
        let axes: Vec<AxisIndex> = if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();
            if tuple_len > ndim {
                return Err(PyValueError::new_err(format!(
                    "too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim, tuple_len
                )));
            }
            let mut v = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                v.push(parse_axis_index(&tuple.get_item(i)?, i, dims[i])?);
            }
            v
        } else {
            // Single int or single slice on axis 0
            vec![parse_axis_index(key, 0, dims[0])?]
        };

        // Pad unspecified trailing axes with full slices
        // e.g. a[3] on a 3D array becomes a[3, :, :]
        let mut full_axes: Vec<AxisIndex> = axes;
        for i in full_axes.len()..ndim {
            full_axes.push(AxisIndex::Slice {
                start: 0,
                step: 1,
                len: dims[i],
            });
        }

        // Build result shape — Single axes are collapsed (removed)
        let result_dims: Vec<usize> = full_axes.iter()
            .filter_map(|a| match a {
                AxisIndex::Single(_) => None,
                AxisIndex::Slice { len, .. } => Some(*len),
            })
            .collect();

        // Fast path: all axes are positive-step slices → zero-copy Strided view.
        // Single axes (which collapse a dimension) and negative steps fall through
        // to the existing gather path below.
        let all_pos_slices = full_axes.iter().all(|a| {
            matches!(a, AxisIndex::Slice { step, .. } if *step >= 1)
        });
        if all_pos_slices {
            let specs: Vec<(usize, usize, usize)> = full_axes.iter()
                .map(|a| match a {
                    AxisIndex::Slice { start, step, len } => (*start, *step as usize, *len),
                    _ => unreachable!(),
                })
                .collect();
            let arr = match &self.inner {
                ArrayData::Float(a) =>
                    PyArray { inner: ArrayData::Float(a.slice_view(&specs)), alive: true },
                ArrayData::Float32(a) =>
                    PyArray { inner: ArrayData::Float32(a.slice_view(&specs)), alive: true },
                ArrayData::Int(a) =>
                    PyArray { inner: ArrayData::Int(a.slice_view(&specs)), alive: true },
            };
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }

        let strides = self.strides_val();

        // Expand each axis into its selected indices
        let per_axis: Vec<Vec<usize>> = full_axes.iter()
            .map(|a| expand_axis_indices(a))
            .collect();

        // Scalar result — all axes were Single
        // Python: a[1, 2, 3] on a 3D array
        if result_dims.is_empty() {
            let flat_idx: usize = per_axis.iter()
                .enumerate()
                .map(|(i, v)| v[0] * strides[i])
                .sum();
            return self.scalar_at_flat(flat_idx, py);
        }

        // Gather via cartesian product of per-axis indices
        let total: usize = per_axis.iter().map(|v| v.len()).product();
        let mut flat_indices = Vec::with_capacity(total);
        let mut coord = vec![0usize; ndim];

        loop {
            let flat_idx: usize = coord.iter()
                .enumerate()
                .map(|(i, &c)| per_axis[i][c] * strides[i])
                .sum();
            flat_indices.push(flat_idx);

            // Increment coord, rightmost first (row-major order)
            let mut axis = ndim - 1;
            loop {
                coord[axis] += 1;
                if coord[axis] < per_axis[axis].len() {
                    break;
                }
                coord[axis] = 0;
                if axis == 0 {
                    // Done — all combinations enumerated
                    let arr = self.gather_flat_indices_with_dims(flat_indices, result_dims);
                    return Ok(arr.into_pyobject(py)?.into_any().unbind());
                }
                axis -= 1;
            }
        }
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        check_alive!(self);
        let dims = self.dims();
        let ndim = dims.len();

        if let Ok(tuple) = key.cast::<PyTuple>() {
            let tuple_len = tuple.len();

            if tuple_len != ndim {
                return Err(PyValueError::new_err(format!(
                    "cannot set item: expected {} indices, got {}",
                    ndim, tuple_len
                )));
            }

            let mut indices: Vec<usize> = Vec::with_capacity(tuple_len);
            for i in 0..tuple_len {
                let item = tuple.get_item(i)?;
                let idx = item.extract::<isize>()?;
                let dim_size = dims[i] as isize;
                let normalized = if idx < 0 { dim_size + idx } else { idx };
                if normalized < 0 || normalized >= dim_size {
                    return Err(PyValueError::new_err(format!(
                        "index {} is out of bounds for axis {} with size {}",
                        idx, i, dims[i]
                    )));
                }
                indices.push(normalized as usize);
            }

            match &mut self.inner {
                ArrayData::Float(a) => {
                    let v: f64 = value.extract()?;
                    *a.get_mut(&indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))? = v;
                }
                ArrayData::Float32(a) => {
                    let v: f32 = value.extract::<f64>()? as f32;
                    *a.get_mut(&indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))? = v;
                }
                ArrayData::Int(a) => {
                    let v: i64 = value.extract()?;
                    *a.get_mut(&indices).ok_or_else(|| PyValueError::new_err("index out of bounds"))? = v;
                }
            }
            return Ok(());
        }

        if let Ok(idx) = key.extract::<isize>() {
            if ndim != 1 {
                return Err(PyValueError::new_err(
                    "single index assignment only supported for 1D arrays; use tuple indexing for nD arrays"
                ));
            }

            let dim0 = dims[0] as isize;
            let normalized = if idx < 0 { dim0 + idx } else { idx };
            if normalized < 0 || normalized >= dim0 {
                return Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}",
                    idx, dims[0]
                )));
            }

            match &mut self.inner {
                ArrayData::Float(a) => {
                    let v: f64 = value.extract()?;
                    a.as_mut_slice().expect("setitem requires owned array")[normalized as usize] = v;
                }
                ArrayData::Float32(a) => {
                    let v: f32 = value.extract::<f64>()? as f32;
                    a.as_mut_slice().expect("setitem requires owned array")[normalized as usize] = v;
                }
                ArrayData::Int(a) => {
                    let v: i64 = value.extract()?;
                    a.as_mut_slice().expect("setitem requires owned array")[normalized as usize] = v;
                }
            }
            return Ok(());
        }

        Err(PyValueError::new_err("indices must be integers or tuples of integers for assignment"))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        match &slf.inner {
            ArrayData::Float(a) => {
                Py::new(py, PyArrayIter { data: a.as_slice_unchecked().to_vec(), index: 0 })
                    .map(|o| o.into_any())
            }
            ArrayData::Float32(a) => {
                let data: Vec<f64> = a.as_slice_unchecked().iter().map(|&x| x as f64).collect();
                Py::new(py, PyArrayIter { data, index: 0 })
                    .map(|o| o.into_any())
            }
            ArrayData::Int(a) => {
                Py::new(py, PyIntArrayIter { data: a.as_slice_unchecked().to_vec(), index: 0 })
                    .map(|o| o.into_any())
            }
        }
    }

    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let v: f64 = value.extract()?;
                Ok(a.as_slice_unchecked().contains(&v))
            }
            ArrayData::Float32(a) => {
                let v: f32 = value.extract::<f64>()? as f32;
                Ok(a.as_slice_unchecked().contains(&v))
            }
            ArrayData::Int(a) => {
                let v: i64 = value.extract()?;
                Ok(a.as_slice_unchecked().contains(&v))
            }
        }
    }
}

// Private helpers — not exposed to Python. These support `__getitem__`, `__richcmp__`,
// `tolist`, and other methods in the #[pymethods] block above.
impl PyArray {
    /// Applies a scalar comparison element-wise, returning a same-shape array of
    /// `1.0`/`0.0` (float arrays) or `1`/`0` (int arrays). Both operands are
    /// compared as `f64`; int arrays are cast before the predicate is evaluated.
    fn apply_cmp(&self, other: ArrayLike, cmp: impl Fn(f64, f64) -> bool) -> PyResult<Self> {
        check_alive!(self);
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = match other {
                    ArrayLike::Scalar(s) =>
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x, s) { 1.0 } else { 0.0 }).collect(),
                    ArrayLike::IntScalar(s) => {
                        let s = s as f64;
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x, s) { 1.0 } else { 0.0 }).collect()
                    }
                    ArrayLike::Array(arr) => {
                        let borrowed = arr.borrow();
                        match &borrowed.inner {
                            ArrayData::Float(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x, y) { 1.0 } else { 0.0 }).collect(),
                            ArrayData::Float32(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x, y as f64) { 1.0 } else { 0.0 }).collect(),
                            ArrayData::Int(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x, y as f64) { 1.0 } else { 0.0 }).collect(),
                        }
                    },
                    _ => {
                        let rhs = other.into_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter())
                            .map(|(&x, &y)| if cmp(x, y) { 1.0 } else { 0.0 }).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Float32(a) => {
                // compare as f64, result is f32 array
                let data: Vec<f32> = match other {
                    ArrayLike::Scalar(s) =>
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x as f64, s) { 1.0 } else { 0.0 }).collect(),
                    ArrayLike::IntScalar(s) =>
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x as f64, s as f64) { 1.0 } else { 0.0 }).collect(),
                    _ => {
                        let rhs = other.into_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter())
                            .map(|(&x, &y)| if cmp(x as f64, y) { 1.0 } else { 0.0 }).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Float32(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
            ArrayData::Int(a) => {
                let data: Vec<i64> = match other {
                    ArrayLike::Scalar(s) =>
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x as f64, s) { 1 } else { 0 }).collect(),
                    ArrayLike::IntScalar(s) => {
                        let s = s as f64;
                        a.as_slice_unchecked().iter().map(|&x| if cmp(x as f64, s) { 1 } else { 0 }).collect()
                    }
                    ArrayLike::Array(arr) => {
                        let borrowed = arr.borrow();
                        match &borrowed.inner {
                            ArrayData::Float(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x as f64, y) { 1 } else { 0 }).collect(),
                            ArrayData::Float32(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x as f64, y as f64) { 1 } else { 0 }).collect(),
                            ArrayData::Int(b) =>
                                a.as_slice_unchecked().iter().zip(b.as_slice_unchecked().iter())
                                    .map(|(&x, &y)| if cmp(x as f64, y as f64) { 1 } else { 0 }).collect(),
                        }
                    },
                    _ => {
                        let rhs = other.into_i64_ndarray()?;
                        a.as_slice_unchecked().iter().zip(rhs.as_slice_unchecked().iter())
                            .map(|(&x, &y)| if cmp(x as f64, y as f64) { 1 } else { 0 }).collect()
                    }
                };
                Ok(PyArray { inner: ArrayData::Int(NdArray::from_vec(a.shape().clone(), data)), alive: true })
            }
        }
    }

    /// Filters the first axis by a boolean mask and returns the surviving rows as a PyArray.
    /// Errors if `mask.len()` doesn't match the first dimension.
    fn apply_bool_mask_py(&self, mask: &[bool], py: Python<'_>) -> PyResult<Py<PyAny>> {
        check_alive!(self);
        let dims = self.dims();
        if dims.is_empty() {
            return Err(PyValueError::new_err("Cannot apply boolean mask to scalar"));
        }
        if mask.len() != dims[0] {
            return Err(PyValueError::new_err(format!(
                "Boolean mask length {} doesn't match first dimension {}",
                mask.len(), dims[0]
            )));
        }
        let result = match &self.inner {
            ArrayData::Float(a) => PyArray { inner: ArrayData::Float(a.boolean_mask(mask)), alive: true },
            ArrayData::Float32(a) => PyArray { inner: ArrayData::Float32(a.boolean_mask(mask)), alive: true },
            ArrayData::Int(a) => PyArray { inner: ArrayData::Int(a.boolean_mask(mask)), alive: true },
        };
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }

    fn dims(&self) -> Vec<usize> {
        match &self.inner {
            ArrayData::Float(a) => a.shape().dims().to_vec(),
            ArrayData::Float32(a) => a.shape().dims().to_vec(),
            ArrayData::Int(a) => a.shape().dims().to_vec(),
        }
    }

    fn ndim_val(&self) -> usize {
        match &self.inner {
            ArrayData::Float(a) => a.ndim(),
            ArrayData::Float32(a) => a.ndim(),
            ArrayData::Int(a) => a.ndim(),
        }
    }

    fn len_val(&self) -> usize {
        match &self.inner {
            ArrayData::Float(a) => a.as_slice_unchecked().len(),
            ArrayData::Float32(a) => a.as_slice_unchecked().len(),
            ArrayData::Int(a) => a.as_slice_unchecked().len(),
        }
    }

    fn strides_val(&self) -> Vec<usize> {
        match &self.inner {
            ArrayData::Float(a) => a.strides().to_vec(),
            ArrayData::Float32(a) => a.strides().to_vec(),
            ArrayData::Int(a) => a.strides().to_vec(),
        }
    }

    fn scalar_at_flat(&self, flat_idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            ArrayData::Float(a) => Ok(a.as_slice_unchecked()[flat_idx].into_pyobject(py)?.into_any().unbind()),
            ArrayData::Float32(a) => Ok((a.as_slice_unchecked()[flat_idx] as f64).into_pyobject(py)?.into_any().unbind()),
            ArrayData::Int(a) => Ok(a.as_slice_unchecked()[flat_idx].into_pyobject(py)?.into_any().unbind()),
        }
    }

    /// Gathers elements at an arbitrary set of flat indices into a new 1-D array.
    /// Used by fancy integer-array indexing on 1-D arrays.
    fn gather_flat_indices(&self, flat_indices: Vec<usize>) -> PyArray {
        let len = flat_indices.len();
        self.gather_flat_indices_with_dims(flat_indices, vec![len])
    }

    /// Gathers elements at arbitrary flat indices into an array with the given shape.
    /// Used by the unified __getitem__ gather routine.
    fn gather_flat_indices_with_dims(&self, flat_indices: Vec<usize>, dims: Vec<usize>) -> PyArray {
        match &self.inner {
            ArrayData::Float(a) => {
                let data: Vec<f64> = flat_indices.iter().map(|&i| a.as_slice_unchecked()[i]).collect();
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(dims), data)), alive: true }
            }
            ArrayData::Float32(a) => {
                let data: Vec<f32> = flat_indices.iter().map(|&i| a.as_slice_unchecked()[i]).collect();
                PyArray { inner: ArrayData::Float32(NdArray::from_vec(Shape::new(dims), data)), alive: true }
            }
            ArrayData::Int(a) => {
                let data: Vec<i64> = flat_indices.iter().map(|&i| a.as_slice_unchecked()[i]).collect();
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(dims), data)), alive: true }
            }
        }
    }

    /// Gathers rows (or sub-tensors) starting at each flat offset in `row_starts`, each of length
    /// `row_size`, and stacks them into an array with shape `result_dims`.
    /// Used by fancy integer-array indexing on multi-dimensional arrays.
    fn gather_rows(&self, row_starts: Vec<usize>, row_size: usize, result_dims: Vec<usize>) -> PyArray {
        match &self.inner {
            ArrayData::Float(a) => {
                let mut result = Vec::new();
                for start in row_starts {
                    result.extend_from_slice(&a.as_slice_unchecked()[start..start + row_size]);
                }
                PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(result_dims), result)), alive: true }
            }
            ArrayData::Float32(a) => {
                let mut result = Vec::new();
                for start in row_starts {
                    result.extend_from_slice(&a.as_slice_unchecked()[start..start + row_size]);
                }
                PyArray { inner: ArrayData::Float32(NdArray::from_vec(Shape::new(result_dims), result)), alive: true }
            }
            ArrayData::Int(a) => {
                let mut result = Vec::new();
                for start in row_starts {
                    result.extend_from_slice(&a.as_slice_unchecked()[start..start + row_size]);
                }
                PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(result_dims), result)), alive: true }
            }
        }
    }

    /// Recursively converts an `NdArray<T>` to a nested Python list.
    /// Works for both `f64` and `i64` elements; the recursion walks each dimension
    /// in turn, using strides to slice the underlying flat buffer correctly.
    fn to_pylist_recursive<T>(&self, a: &NdArray<T>, py: Python<'_>, dim: usize, offset: usize) -> PyResult<Py<PyAny>>
    where
        T: Copy + for<'py> pyo3::IntoPyObject<'py>,
    {
        let data = a.as_slice_unchecked();
        let dim_size = a.shape().dim(dim).unwrap();
        let strides = a.strides();

        if dim == a.ndim() - 1 {
            let list = PyList::new(py, (0..dim_size).map(|i| data[offset + i * strides[dim]]))?;
            return Ok(list.into());
        }

        let items: Vec<Py<PyAny>> = (0..dim_size)
            .map(|i| self.to_pylist_recursive(a, py, dim + 1, offset + i * strides[dim]))
            .collect::<PyResult<_>>()?;
        Ok(PyList::new(py, items)?.into())
    }

    /// Applies fancy integer-array indexing along axis 0. Negative indices are normalised.
    /// Returns a scalar for 1-D arrays or a sub-array for N-D arrays.
    fn apply_int_index_array_py(&self, indices: &[isize], py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dims = self.dims();
        let dim0 = dims[0] as isize;

        let normalized: Vec<usize> = indices.iter().map(|&idx| {
            let n = if idx < 0 { dim0 + idx } else { idx };
            if n < 0 || n >= dim0 {
                Err(PyValueError::new_err(format!(
                    "index {} is out of bounds for axis 0 with size {}", idx, dim0
                )))
            } else {
                Ok(n as usize)
            }
        }).collect::<PyResult<_>>()?;

        if dims.len() == 1 {
            let arr = self.gather_flat_indices(normalized);
            return Ok(arr.into_pyobject(py)?.into_any().unbind());
        }

        let stride0 = self.strides_val()[0];
        let row_starts: Vec<usize> = normalized.iter().map(|&i| i * stride0).collect();
        let row_size: usize = dims[1..].iter().product();
        let mut result_dims = vec![normalized.len()];
        result_dims.extend_from_slice(&dims[1..]);

        let arr = self.gather_rows(row_starts, row_size, result_dims);
        Ok(arr.into_pyobject(py)?.into_any().unbind())
    }
}

// -------------------------------------------------------------------------
// Buffer protocol
// -------------------------------------------------------------------------

struct BufferViewInternal {
    shape: Vec<isize>,
    strides: Vec<isize>,
    format: std::ffi::CString,
}

#[pymethods]
impl PyArray {
    unsafe fn __getbuffer__(
        slf: pyo3::Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::ffi::c_int,
    ) -> pyo3::PyResult<()> {
        use std::ffi::c_void;
        use std::mem::size_of;
        use pyo3::ffi;

        let py_arr = slf.borrow();
        if !py_arr.alive {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "Array has been consumed by a tree"
            ));
        }

        let wants_write = (flags & ffi::PyBUF_WRITABLE) != 0;

        macro_rules! fill_view {
            ($arr:expr, $fmt:literal, $T:ty) => {{
                if wants_write && !$arr.is_owned() {
                    return Err(pyo3::exceptions::PyBufferError::new_err(
                        "Object is not writable"
                    ));
                }
                let is_contiguous = $arr.is_contiguous();
                if !is_contiguous && (flags & ffi::PyBUF_STRIDES) == 0 {
                    return Err(pyo3::exceptions::PyBufferError::new_err(
                        "Array is not contiguous; request strides (PyBUF_STRIDES) \
                         or call to_contiguous() first"
                    ));
                }
                let (raw_ptr, elem_count) = $arr.as_raw_parts();
                let ptr = raw_ptr as *mut c_void;
                let itemsize = size_of::<$T>() as isize;
                let ndim = $arr.shape().dims().len();
                let shape: Vec<isize> = $arr.shape().dims().iter().map(|&d| d as isize).collect();

                let strides: Vec<isize> = $arr.strides().iter()
                    .map(|&s| s as isize * itemsize)
                    .collect();
                let format = std::ffi::CString::new($fmt).unwrap();
                let internal = Box::new(BufferViewInternal { shape, strides, format });
                let internal_raw = Box::into_raw(internal);

                unsafe {
                    (*view).obj = slf.clone().into_any().unbind().into_ptr();
                    (*view).buf = ptr;
                    (*view).len = elem_count as isize * itemsize;
                    (*view).itemsize = itemsize;
                    (*view).readonly = if $arr.is_owned() && !wants_write { 0 } else { 1 };
                    (*view).ndim = ndim as std::ffi::c_int;
                    (*view).format = if (flags & ffi::PyBUF_FORMAT) != 0 {
                        (*internal_raw).format.as_ptr() as *mut _
                    } else {
                        std::ptr::null_mut()
                    };
                    (*view).shape = (*internal_raw).shape.as_mut_ptr();
                    (*view).strides = (*internal_raw).strides.as_mut_ptr();
                    (*view).suboffsets = std::ptr::null_mut();
                    (*view).internal = internal_raw as *mut c_void;
                }
            }};
        }

        match &py_arr.inner {
            super::ArrayData::Float(arr)   => fill_view!(arr, "d", f64),
            super::ArrayData::Float32(arr) => fill_view!(arr, "f", f32),
            super::ArrayData::Int(arr)     => fill_view!(arr, "q", i64),
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe {
            if !(*view).internal.is_null() {
                drop(Box::from_raw((*view).internal as *mut BufferViewInternal));
                (*view).internal = std::ptr::null_mut();
            }
        }
    }
}

#[pyclass]
pub struct PyArrayIter {
    data: Vec<f64>,
    index: usize,
}

#[pymethods]
impl PyArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<f64> {
        if slf.index < slf.data.len() {
            let val = slf.data[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

#[pyclass]
pub struct PyIntArrayIter {
    data: Vec<i64>,
    index: usize,
}

#[pymethods]
impl PyIntArrayIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<i64> {
        if slf.index < slf.data.len() {
            let val = slf.data[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}