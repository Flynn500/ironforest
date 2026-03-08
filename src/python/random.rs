use pyo3::prelude::*;
use crate::random::Generator;
use crate::array::{NdArray, Shape};
use super::{PyArray, ArrayData};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGenerator>()?;
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(new, m)?)?;
    Ok(())
}

#[pyfunction]
fn seed(seed: u64) -> PyGenerator {
    PyGenerator::from_seed(seed)
}

#[pyfunction]
fn new() -> PyGenerator {
    PyGenerator::new()
}

#[pyclass(name = "Generator")]
pub struct PyGenerator {
    inner: Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new() -> Self {
        PyGenerator { inner: Generator::new() }
    }

    #[staticmethod]
    fn from_seed(seed: u64) -> Self {
        PyGenerator { inner: Generator::from_seed(seed) }
    }

    fn uniform(&mut self, low: f64, high: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.uniform(low, high, Shape::new(shape))),
            alive: true
        }
    }

    fn standard_normal(&mut self, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.standard_normal(Shape::new(shape))),
            alive: true
        }
    }

    fn normal(&mut self, mu: f64, sigma: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.normal(mu, sigma, Shape::new(shape))),
            alive: true
        }
    }

    fn randint(&mut self, low: i64, high: i64, shape: Vec<usize>) -> PyArray {
        let arr = self.inner.randint(low, high, Shape::new(shape.clone()));
        let data: Vec<f64> = arr.as_slice().iter().map(|&x| x as f64).collect();
        PyArray {
            inner: ArrayData::Float(NdArray::from_vec(Shape::new(shape), data)),
            alive: true
        }
    }

    fn gamma(&mut self, shape_param: f64, scale: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.gamma(shape_param, scale, Shape::new(shape))),
            alive: true
        }
    }

    fn beta(&mut self, alpha: f64, beta_param: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.beta(alpha, beta_param, Shape::new(shape))),
            alive: true
        }
    }

    fn lognormal(&mut self, mu: f64, sigma: f64, shape: Vec<usize>) -> PyArray {
        PyArray {
            inner: ArrayData::Float(self.inner.lognormal(mu, sigma, Shape::new(shape))),
            alive: true
        }
    }
}

