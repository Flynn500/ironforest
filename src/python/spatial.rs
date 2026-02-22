use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::array::{NdArray, Shape};
use crate::spatial::{BallTree, KDTree, VPTree, AggTree, BruteForce, VantagePointSelection, DistanceMetric, KernelType, SpatialQuery};
use super::{PyArray, ArrayLike};

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBallTree>()?;
    m.add_class::<PyKDTree>()?;
    m.add_class::<PyVPTree>()?;
    m.add_class::<PyAggTree>()?;
    m.add_class::<PyBruteForce>()?;
    Ok(())
}

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
            match metric.to_lowercase().as_str() {
                "euclidean" => Ok(DistanceMetric::Euclidean),
                "manhattan" => Ok(DistanceMetric::Manhattan),
                "chebyshev" => Ok(DistanceMetric::Chebyshev),
                _ => Err(PyValueError::new_err(format!(
                    "Unknown distance metric '{}'. Valid options: 'euclidean', 'manhattan', 'chebyshev'",
                    metric
                ))),
            }
        }

        fn parse_kernel(kernel: &str) -> PyResult<KernelType> {
            match kernel.to_lowercase().as_str() {
                "gaussian" => Ok(KernelType::Gaussian),
                "epanechnikov" => Ok(KernelType::Epanechnikov),
                "uniform" => Ok(KernelType::Uniform),
                "triangular" => Ok(KernelType::Triangular),
                _ => Err(PyValueError::new_err(format!(
                    "Unknown kernel type '{}'. Valid options: 'gaussian', 'epanechnikov', 'uniform', 'triangular'",
                    kernel
                ))),
            }
        }

        fn parse_vantage_selection(selection: &str) -> PyResult<VantagePointSelection> {
            match selection.to_lowercase().as_str() {
                "first" => Ok(VantagePointSelection::First),
                "random" => Ok(VantagePointSelection::Random),
                _ => Err(PyValueError::new_err(format!(
                    "Unknown vantage point selection method '{}'. Valid options: 'first', 'random'",
                    selection
                ))),
            }
        }

        #[pyclass(name = "BallTree")]
        pub struct PyBallTree {
            inner: BallTree,
        }

        #[pymethods]
        impl PyBallTree {
            #[staticmethod]
            #[pyo3(signature = (array, leaf_size=20, metric="euclidean"))]
            fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>) -> PyResult<Self> {
                let leaf_size = leaf_size.unwrap_or(20);
                let metric_str = metric.unwrap_or("euclidean");
                let metric = parse_metric(metric_str)?;
                let tree = BallTree::from_ndarray(&array.inner, leaf_size, metric);
                Ok(PyBallTree { inner: tree })
            }

            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_radius(&query_vec, radius))
            }

            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_knn(&query_vec, k))
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<ArrayLike>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                normalize: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let normalize = normalize.unwrap_or(false);

                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(self.inner.dim)?
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }

        #[pyclass(name = "KDTree")]
        pub struct PyKDTree {
            inner: KDTree,
        }

        #[pymethods]
        impl PyKDTree {
            #[staticmethod]
            #[pyo3(signature = (array, leaf_size=20, metric="euclidean"))]
            fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>) -> PyResult<Self> {
                let leaf_size = leaf_size.unwrap_or(20);
                let metric_str = metric.unwrap_or("euclidean");
                let metric = parse_metric(metric_str)?;
                let tree = KDTree::from_ndarray(&array.inner, leaf_size, metric);
                Ok(PyKDTree { inner: tree })
            }

            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_radius(&query_vec, radius))
            }

            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_knn(&query_vec, k))
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<ArrayLike>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                normalize: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let normalize = normalize.unwrap_or(false);

                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(self.inner.dim)?
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }

        #[pyclass(name = "VPTree")]
        pub struct PyVPTree {
            inner: VPTree,
        }

        #[pymethods]
        impl PyVPTree {
            #[staticmethod]
            #[pyo3(signature = (array, leaf_size=20, metric="euclidean", selection="first"))]
            fn from_array(array: &PyArray, leaf_size: Option<usize>, metric: Option<&str>, selection: Option<&str>) -> PyResult<Self> {
                let leaf_size = leaf_size.unwrap_or(20);
                let metric_str = metric.unwrap_or("euclidean");
                let metric = parse_metric(metric_str)?;
                let selection_str = selection.unwrap_or("first");
                let selection_method = parse_vantage_selection(selection_str)?;
                let tree = VPTree::from_ndarray(&array.inner, leaf_size, metric, selection_method);
                Ok(PyVPTree { inner: tree })
            }

            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_radius(&query_vec, radius))
            }

            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_knn(&query_vec, k))
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<ArrayLike>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                normalize: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let normalize = normalize.unwrap_or(false);

                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(self.inner.dim)?
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }

        #[pyclass(name = "AggTree")]
        pub struct PyAggTree {
            inner: AggTree,
        }

        #[pymethods]
        impl PyAggTree {
            #[staticmethod]
            #[pyo3(signature = (array, leaf_size=20, metric="euclidean", kernel="gaussian", bandwidth=1.0, atol=0.01))]
            fn from_array(
                array: &PyArray,
                leaf_size: Option<usize>,
                metric: Option<&str>,
                kernel: Option<&str>,
                bandwidth: Option<f64>,
                atol: Option<f64>,
            ) -> PyResult<Self> {
                let leaf_size = leaf_size.unwrap_or(20);
                let metric = parse_metric(metric.unwrap_or("euclidean"))?;
                let kernel = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let bandwidth = bandwidth.unwrap_or(1.0);
                let atol = atol.unwrap_or(0.01);
                let tree = AggTree::from_ndarray(&array.inner, leaf_size, metric, kernel, bandwidth, atol);
                Ok(PyAggTree { inner: tree })
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<ArrayLike>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                normalize: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let normalize = normalize.unwrap_or(false);

                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(self.inner.dim)?
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }

        #[pyclass(name = "BruteForce")]
        pub struct PyBruteForce {
            inner: BruteForce,
        }

        #[pymethods]
        impl PyBruteForce {
            #[staticmethod]
            #[pyo3(signature = (array, metric="euclidean"))]
            fn from_array(array: &PyArray, metric: Option<&str>) -> PyResult<Self> {
                let metric = parse_metric(metric.unwrap_or("euclidean"))?;
                let tree = BruteForce::from_ndarray(&array.inner, metric);
                Ok(PyBruteForce { inner: tree })
            }

            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_radius(&query_vec, radius))
            }

            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = query.into_vec_with_dim(self.inner.dim)?;
                Ok(self.inner.query_knn(&query_vec, k))
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", normalize=true))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<ArrayLike>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                normalize: Option<bool>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
                let normalize = normalize.unwrap_or(false);

                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(self.inner.dim)?
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }
