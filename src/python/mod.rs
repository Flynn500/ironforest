use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::PyAny;
use pyo3::{FromPyObject, Borrowed};
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use crate::array::{NdArray, Shape};

/// Validates that `v` is a non-empty, rectangular 2-D nested list.
/// Returns `(rows, cols)` on success, or a `PyValueError` if the list is empty
fn validate_vec2d<T>(v: &[Vec<T>]) -> PyResult<(usize, usize)> {
    if v.is_empty() {
        return Err(PyValueError::new_err("Cannot create array from empty nested list"));
    }
    let cols = v[0].len();
    for row in v {
        if row.len() != cols {
            return Err(PyValueError::new_err("Nested lists must have consistent dimensions"));
        }
    }
    Ok((v.len(), cols))
}

#[derive(Clone)]
pub enum ArrayData {
    Float(NdArray<f64>),
    Float32(NdArray<f32>),
    Int(NdArray<i64>),
}

pub enum FloatNdArray {
    F32(NdArray<f32>),
    F64(NdArray<f64>),
}

pub enum ArrayLike<'py> {
    Array(Bound<'py, PyArray>),
    NumPy(Bound<'py, PyUntypedArray>),
    Buffer(Bound<'py, PyAny>),
    Scalar(f64),
    Vec(Vec<f64>),
    Vec2D(Vec<Vec<f64>>),
    IntScalar(i64),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ArrayLike<'py> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(arr) = ob.cast::<PyArray>() {
            return Ok(ArrayLike::Array(arr.to_owned()));
        }

        if let Ok(arr) = ob.cast::<PyUntypedArray>() {
            return Ok(ArrayLike::NumPy(arr.to_owned()));
        }

        //Try convert to numpy array if possible
        if ob.hasattr("to_numpy").unwrap_or(false) {
            let py = ob.py();
            let np = py.import("numpy")?;
            let arr = ob.call_method1("to_numpy", (np.getattr("float64")?,))?;
            if let Ok(arr) = arr.cast::<PyUntypedArray>() {
                return Ok(ArrayLike::NumPy(arr.to_owned()));
            }
        }

        if unsafe { !(*(*ob.as_ptr()).ob_type).tp_as_buffer.is_null() } {
            return Ok(ArrayLike::Buffer(ob.to_owned()));
        }

        if let Ok(outer_list) = ob.extract::<Vec<Vec<f64>>>() {
            return Ok(ArrayLike::Vec2D(outer_list));
        }

        if let Ok(list) = ob.extract::<Vec<f64>>() {
            return Ok(ArrayLike::Vec(list));
        }

        if let Ok(s) = ob.extract::<i64>() {
            return Ok(ArrayLike::IntScalar(s));
        }

        if let Ok(scalar) = ob.extract::<f64>() {
            return Ok(ArrayLike::Scalar(scalar));
        }

        Err(PyValueError::new_err(
            "Expected Array, NumPy array, list, or scalar"
        ))
    }
}

impl<'py> ArrayLike<'py> {
    pub fn into_ndarray(self) -> PyResult<NdArray<f64>> {
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                match &inner.inner {
                    ArrayData::Float(f) => Ok(f.clone()),
                    ArrayData::Float32(f) => {
                        let data: Vec<f64> = f.as_slice_unchecked().iter().map(|&x| x as f64).collect();
                        Ok(NdArray::from_vec(f.shape().clone(), data))
                    }
                    ArrayData::Int(_) => Err(PyTypeError::new_err(
                        "expected float array; got integer array"
                    )),
                }
            }
            ArrayLike::NumPy(bound) => {
                if let Ok(arr) = bound.cast::<PyArrayDyn<f64>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();

                    if readonly.is_c_contiguous() {
                        let slice = readonly.as_slice()?;
                        let owner = bound.clone().into_any().unbind();
                        Ok(unsafe { NdArray::from_external(owner, slice.as_ptr(), Shape::new(shape)) })
                    } else {
                        let data: Vec<f64> = readonly.as_array().iter().copied().collect();
                        Ok(NdArray::from_vec(Shape::new(shape), data))
                    }
                } else if let Ok(arr) = bound.cast::<PyArrayDyn<f32>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data: Vec<f64> = readonly.as_array().iter().map(|&x| x as f64).collect();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                } else {
                    Err(PyTypeError::new_err("expected float32 or float64 numpy array"))
                }
            }
            ArrayLike::Buffer(bound) => {
                let buf = pyo3::buffer::PyBuffer::<f64>::get(&bound)
                    .map_err(|_| PyTypeError::new_err(
                        "buffer contents are not compatible with f64"
                    ))?;
                NdArray::from_buffer(bound.py(), buf)
            }
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s])),
            ArrayLike::Vec(v) => Ok(NdArray::from_vec(Shape::d1(v.len()), v)),
            ArrayLike::Vec2D(v) => {
                let (rows, cols) = validate_vec2d(&v)?;
                let flat: Vec<f64> = v.into_iter().flatten().collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), flat))
            }
            ArrayLike::IntScalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s as f64])),
        }
    }

    pub fn into_i64_ndarray(self) -> PyResult<NdArray<i64>> {
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                match &inner.inner {
                    ArrayData::Int(i) => Ok(i.clone()),
                    ArrayData::Float(f) => {
                        let data: Vec<i64> = f.as_slice_unchecked().iter().map(|&x| x as i64).collect();
                        Ok(NdArray::from_vec(f.shape().clone(), data))
                    }
                    ArrayData::Float32(f) => {
                        let data: Vec<i64> = f.as_slice_unchecked().iter().map(|&x| x as i64).collect();
                        Ok(NdArray::from_vec(f.shape().clone(), data))
                    }
                }
            }
            //add zero copy path
            ArrayLike::NumPy(bound) => {
                if let Ok(arr) = bound.cast::<PyArrayDyn<i64>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data = readonly.as_slice()?.to_vec();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                } else if let Ok(arr) = bound.cast::<PyArrayDyn<f64>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data: Vec<i64> = readonly.as_slice()?.iter().map(|&x| x as i64).collect();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                } else {
                    Err(PyTypeError::new_err("expected int64 or float64 numpy array"))
                }
            }
            ArrayLike::Buffer(bound) => {
                if let Ok(buf) = pyo3::buffer::PyBuffer::<i64>::get(&bound) {
                    NdArray::from_buffer(bound.py(), buf)
                } else {
                    let buf = pyo3::buffer::PyBuffer::<f64>::get(&bound)?;
                    let shape: Vec<usize> = buf.shape().iter().map(|&d| d as usize).collect();
                    let data: Vec<i64> = buf.to_vec(bound.py())?.into_iter().map(|x| x as i64).collect();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                }
            }
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::new(vec![1]), vec![s as i64])),
            ArrayLike::Vec(v) => {
                let data: Vec<i64> = v.iter().map(|&x| x as i64).collect();
                Ok(NdArray::from_vec(Shape::d1(data.len()), data))
            }
            ArrayLike::Vec2D(v) => {
                let (rows, cols) = validate_vec2d(&v)?;
                let data: Vec<i64> = v.into_iter().flatten().map(|x| x as i64).collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), data))
            }
            ArrayLike::IntScalar(s) => Ok(NdArray::from_vec(Shape::new(vec![1]), vec![s])),
        }
    }

    pub fn ndim(&self) -> usize {
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                match &inner.inner {
                    ArrayData::Float(f) => f.shape().dims().len(),
                    ArrayData::Float32(f) => f.shape().dims().len(),
                    ArrayData::Int(i) => i.shape().dims().len(),
                }
            }
            ArrayLike::NumPy(bound) => bound.ndim(),
            ArrayLike::Buffer(bound) => {
                pyo3::buffer::PyBuffer::<u8>::get(bound)
                    .map(|b| b.dimensions())
                    .unwrap_or(0)
            }
            ArrayLike::Vec(_) => 1,
            ArrayLike::Vec2D(_) => 2,
            ArrayLike::Scalar(_) | ArrayLike::IntScalar(_) => 0,
        }
    }

    pub fn is_f32(&self) -> bool {
        match self {
            ArrayLike::NumPy(bound) => bound.cast::<PyArrayDyn<f32>>().is_ok(),
            ArrayLike::Array(bound) => matches!(bound.borrow().inner, ArrayData::Float32(_)),
            _ => false,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                matches!(inner.inner, ArrayData::Int(_))
            }
            ArrayLike::IntScalar(_) => true,

            ArrayLike::Buffer(_) => false,
            _ => false,
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
                    arr.as_slice_unchecked().to_vec()
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
            Ok(arr.as_slice_unchecked().to_vec())
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

    pub fn into_f32_ndarray(self) -> PyResult<NdArray<f32>> {
        use numpy::PyArrayDyn;
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                match &inner.inner {
                    ArrayData::Float32(f) => Ok(f.clone()),
                    ArrayData::Float(f) => {
                        let data: Vec<f32> = f.as_slice_unchecked().iter().map(|&x| x as f32).collect();
                        Ok(NdArray::from_vec(f.shape().clone(), data))
                    }
                    ArrayData::Int(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "expected float array; got integer array"
                    )),
                }
            }
            ArrayLike::NumPy(bound) => {
                if let Ok(arr) = bound.cast::<PyArrayDyn<f32>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data = readonly.as_slice()?.to_vec();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                } else if let Ok(arr) = bound.cast::<PyArrayDyn<f64>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data: Vec<f32> = readonly.as_slice()?.iter().map(|&x| x as f32).collect();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                } else {
                    Err(pyo3::exceptions::PyTypeError::new_err("expected float32 or float64 numpy array"))
                }
            }
            ArrayLike::Buffer(bound) => {
                if let Ok(buf) = pyo3::buffer::PyBuffer::<f32>::get(&bound) {
                    NdArray::from_buffer(bound.py(), buf)
                } else {
                    let buf = pyo3::buffer::PyBuffer::<f64>::get(&bound)
                        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("buffer not compatible with f32 or f64"))?;
                    let shape: Vec<usize> = buf.shape().iter().map(|&d| d as usize).collect();
                    let data: Vec<f32> = buf.to_vec(bound.py())?.into_iter().map(|x| x as f32).collect();
                    Ok(NdArray::from_vec(Shape::new(shape), data))
                }
            }
            ArrayLike::Scalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s as f32])),
            ArrayLike::Vec(v) => Ok(NdArray::from_vec(Shape::d1(v.len()), v.iter().map(|&x| x as f32).collect())),
            ArrayLike::Vec2D(v) => {
                let (rows, cols) = validate_vec2d(&v)?;
                let flat: Vec<f32> = v.into_iter().flatten().map(|x| x as f32).collect();
                Ok(NdArray::from_vec(Shape::new(vec![rows, cols]), flat))
            }
            ArrayLike::IntScalar(s) => Ok(NdArray::from_vec(Shape::scalar(), vec![s as f32])),
        }
    }

    pub fn into_float_ndarray(self) -> PyResult<FloatNdArray> {
        use numpy::PyArrayDyn;
        match self {
            ArrayLike::Array(bound) => {
                let inner = bound.borrow();
                match &inner.inner {
                    ArrayData::Float32(f) => Ok(FloatNdArray::F32(f.clone())),
                    ArrayData::Float(f) => Ok(FloatNdArray::F64(f.clone())),
                    ArrayData::Int(_) => Err(PyTypeError::new_err(
                        "expected float array; got integer array"
                    )),
                }
            }
            ArrayLike::NumPy(bound) => {
                if let Ok(arr) = bound.cast::<PyArrayDyn<f32>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    let data = readonly.as_slice()?.to_vec();
                    Ok(FloatNdArray::F32(NdArray::from_vec(Shape::new(shape), data)))
                } else if let Ok(arr) = bound.cast::<PyArrayDyn<f64>>() {
                    let readonly = arr.readonly();
                    let shape = readonly.shape().to_vec();
                    if readonly.is_c_contiguous() {
                        let slice = readonly.as_slice()?;
                        let owner = bound.clone().into_any().unbind();
                        Ok(FloatNdArray::F64(unsafe { NdArray::from_external(owner, slice.as_ptr(), Shape::new(shape)) }))
                    } else {
                        let data: Vec<f64> = readonly.as_array().iter().copied().collect();
                        Ok(FloatNdArray::F64(NdArray::from_vec(Shape::new(shape), data)))
                    }
                } else {
                    Err(PyTypeError::new_err("expected float32 or float64 numpy array"))
                }
            }
            other => {
                let arr = other.into_ndarray()?;
                Ok(FloatNdArray::F64(arr))
            }
        }
    }

    pub fn into_f32_spatial_query_ndarray(self, expected_dim: usize) -> PyResult<NdArray<f32>> {
        let arr = self.into_f32_ndarray()?;
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
                    arr.as_slice_unchecked().to_vec()
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
}

#[pyclass(name = "Array")]
#[derive(Clone)]
pub struct PyArray {
    pub inner: ArrayData,
    pub alive: bool,
}

impl PyArray {

    pub fn as_float(&self) -> PyResult<&NdArray<f64>> {
        match &self.inner {
            ArrayData::Float(a) => Ok(a),
            ArrayData::Float32(_) | ArrayData::Int(_) => Err(PyTypeError::new_err(
                "operation not supported for non-float64 arrays"
            )),
        }
    }

    pub fn as_float32(&self) -> PyResult<&NdArray<f32>> {
        match &self.inner {
            ArrayData::Float32(a) => Ok(a),
            ArrayData::Float(_) | ArrayData::Int(_) => Err(PyTypeError::new_err(
                "operation not supported for non-float32 arrays"
            )),
        }
    }

    pub fn as_int(&self) -> PyResult<&NdArray<i64>> {
        match &self.inner {
            ArrayData::Int(a) => Ok(a),
            ArrayData::Float(_) | ArrayData::Float32(_) => Err(PyTypeError::new_err(
                "operation not supported for integer arrays"
            )),
        }
    }

}

// macro for handling dead array case
macro_rules! check_alive {
    ($self:expr) => {
        if !$self.alive {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Array consumed by tree, use tree.data to access the data"
            ));
        }
    };
}

impl PyArray {
    /// Returns a zero-copy view of the inner `NdArray<f64>`.
    ///
    /// For `External` (numpy-backed) arrays this increments the Python refcount
    /// and copies the raw pointer — no data allocation.  For `Owned` arrays it
    /// falls back to a materialising `clone()`.
    pub fn as_view_float(&self) -> PyResult<NdArray<f64>> {
        check_alive!(self);
        Ok(self.as_float()?.as_view())
    }

    pub fn as_view_float32(&self) -> PyResult<NdArray<f32>> {
        check_alive!(self);
        Ok(self.as_float32()?.as_view())
    }

    /// Moves the inner `NdArray<f64>` out of this `PyArray`, marks it dead, and
    /// returns ownership to the caller.
    pub fn take_float(&mut self) -> PyResult<NdArray<f64>> {
        check_alive!(self);
        if !matches!(self.inner, ArrayData::Float(_)) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "operation not supported for non-float64 arrays",
            ));
        }
        let dummy = ArrayData::Float(NdArray::from_vec(Shape::new(vec![0, 0]), vec![]));
        let taken = std::mem::replace(&mut self.inner, dummy);
        self.alive = false;
        match taken {
            ArrayData::Float(nd) => Ok(nd),
            _ => unreachable!(),
        }
    }

    pub fn take_float32(&mut self) -> PyResult<NdArray<f32>> {
        check_alive!(self);
        if !matches!(self.inner, ArrayData::Float32(_)) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "operation not supported for non-float32 arrays",
            ));
        }
        let dummy = ArrayData::Float32(NdArray::from_vec(Shape::new(vec![0, 0]), vec![]));
        let taken = std::mem::replace(&mut self.inner, dummy);
        self.alive = false;
        match taken {
            ArrayData::Float32(nd) => Ok(nd),
            _ => unreachable!(),
        }
    }
}

pub mod array;
pub mod ndutils;
pub mod linalg;
pub mod stats;
pub mod random;
pub mod spatial;
pub mod tree_engine;

pub use array::{PyArrayIter, PyIntArrayIter};
pub use random::PyGenerator;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArray>()?;
    m.add_class::<PyArrayIter>()?;
    m.add_class::<PyIntArrayIter>()?;
    m.add_class::<PyGenerator>()?;
    m.add_class::<spatial::PyBallTree>()?;
    m.add_class::<spatial::PyKDTree>()?;

    let ndutils_module = PyModule::new(m.py(), "ndutils")?;
    ndutils::register_module(&ndutils_module)?;
    m.add_submodule(&ndutils_module)?;

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

    tree_engine::register_classes(m)?;

    Ok(())
}
