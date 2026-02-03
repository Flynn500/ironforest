use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::array::{NdArray, Shape};
use crate::spatial::{BallTree, KDTree, VPTree, VantagePointSelection, DistanceMetric, KernelType, ApproxCriterion};
use super::PyArray;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBallTree>()?;
    m.add_class::<PyKDTree>()?;
    m.add_class::<PyVPTree>()?;
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

            fn query_radius(&self, query: &Bound<'_, PyAny>, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let indices = self.inner.query_radius(&query_vec, radius);

                Ok(indices)
            }

            fn query_knn(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let results = self.inner.query_knn(&query_vec, k);

                Ok(results)
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian"))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", criterion="none", min_samples=None, max_span=None))]
            fn kernel_density_approx(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                criterion: Option<&str>,
                min_samples: Option<usize>,
                max_span: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let approx_criterion = match criterion.unwrap_or("none").to_lowercase().as_str() {
                    "none" => ApproxCriterion::None,
                    "min_samples" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'min_samples' criterion")
                        )?;
                        ApproxCriterion::MinSamples(samples)
                    },
                    "max_span" => {
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'max_span' criterion")
                        )?;
                        ApproxCriterion::MaxSpan(span)
                    },
                    "combined" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'combined' criterion")
                        )?;
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'combined' criterion")
                        )?;
                        ApproxCriterion::Combined(samples, span)
                    },
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown approximation criterion '{}'. Valid options: 'none', 'min_samples', 'max_span', 'combined'",
                        criterion.unwrap_or("none")
                    ))),
                };

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density_approx(&queries_arr, bandwidth, kernel_type, &approx_criterion);

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

            fn query_radius(&self, query: &Bound<'_, PyAny>, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let indices = self.inner.query_radius(&query_vec, radius);

                Ok(indices)
            }

            fn query_knn(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let results = self.inner.query_knn(&query_vec, k);

                Ok(results)
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian"))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", criterion="none", min_samples=None, max_span=None))]
            fn kernel_density_approx(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                criterion: Option<&str>,
                min_samples: Option<usize>,
                max_span: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let approx_criterion = match criterion.unwrap_or("none").to_lowercase().as_str() {
                    "none" => ApproxCriterion::None,
                    "min_samples" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'min_samples' criterion")
                        )?;
                        ApproxCriterion::MinSamples(samples)
                    },
                    "max_span" => {
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'max_span' criterion")
                        )?;
                        ApproxCriterion::MaxSpan(span)
                    },
                    "combined" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'combined' criterion")
                        )?;
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'combined' criterion")
                        )?;
                        ApproxCriterion::Combined(samples, span)
                    },
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown approximation criterion '{}'. Valid options: 'none', 'min_samples', 'max_span', 'combined'",
                        criterion.unwrap_or("none")
                    ))),
                };

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density_approx(&queries_arr, bandwidth, kernel_type, &approx_criterion);

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

            fn query_radius(&self, query: &Bound<'_, PyAny>, radius: f64) -> PyResult<Vec<usize>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let indices = self.inner.query_radius(&query_vec, radius);

                Ok(indices)
            }

            fn query_knn(&self, query: &Bound<'_, PyAny>, k: usize) -> PyResult<Vec<(usize, f64)>> {
                let query_vec = if let Ok(scalar) = query.extract::<f64>() {
                    vec![scalar]
                } else if let Ok(vec_data) = query.extract::<Vec<f64>>() {
                    vec_data
                } else if let Ok(arr) = query.extract::<PyArray>() {
                    arr.inner.as_slice().to_vec()
                } else {
                    return Err(PyValueError::new_err("query must be a scalar, list, or Array"));
                };

                let results = self.inner.query_knn(&query_vec, k);

                Ok(results)
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian"))]
            fn kernel_density(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density(&queries_arr, bandwidth, kernel_type);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }

            #[pyo3(signature = (queries=None, bandwidth=1.0, kernel="gaussian", criterion="none", min_samples=None, max_span=None))]
            fn kernel_density_approx(
                &self,
                py: Python<'_>,
                queries: Option<&Bound<'_, PyAny>>,
                bandwidth: Option<f64>,
                kernel: Option<&str>,
                criterion: Option<&str>,
                min_samples: Option<usize>,
                max_span: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                let bandwidth = bandwidth.unwrap_or(1.0);
                let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;

                let approx_criterion = match criterion.unwrap_or("none").to_lowercase().as_str() {
                    "none" => ApproxCriterion::None,
                    "min_samples" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'min_samples' criterion")
                        )?;
                        ApproxCriterion::MinSamples(samples)
                    },
                    "max_span" => {
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'max_span' criterion")
                        )?;
                        ApproxCriterion::MaxSpan(span)
                    },
                    "combined" => {
                        let samples = min_samples.ok_or_else(||
                            PyValueError::new_err("min_samples parameter required for 'combined' criterion")
                        )?;
                        let span = max_span.ok_or_else(||
                            PyValueError::new_err("max_span parameter required for 'combined' criterion")
                        )?;
                        ApproxCriterion::Combined(samples, span)
                    },
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown approximation criterion '{}'. Valid options: 'none', 'min_samples', 'max_span', 'combined'",
                        criterion.unwrap_or("none")
                    ))),
                };

                let queries_arr = if let Some(q) = queries {
                    if let Ok(scalar) = q.extract::<f64>() {
                        NdArray::from_vec(Shape::new(vec![1, 1]), vec![scalar])
                    } else if let Ok(arr) = q.extract::<PyArray>() {
                        let shape = arr.inner.shape().dims();
                        if shape.len() == 1 {
                            if shape[0] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[0], self.inner.dim
                                )));
                            }
                            NdArray::from_vec(
                                Shape::new(vec![1, shape[0]]),
                                arr.inner.as_slice().to_vec()
                            )
                        } else if shape.len() == 2 {
                            if shape[1] != self.inner.dim {
                                return Err(PyValueError::new_err(format!(
                                    "Query dimension {} doesn't match tree dimension {}",
                                    shape[1], self.inner.dim
                                )));
                            }
                            arr.inner.clone()
                        } else {
                            return Err(PyValueError::new_err(
                                "queries must be 1D or 2D array"
                            ));
                        }
                    } else if let Ok(vec_data) = q.extract::<Vec<f64>>() {
                        let n = vec_data.len();
                        if n == self.inner.dim {
                            NdArray::from_vec(Shape::new(vec![1, n]), vec_data)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Query vector length {} doesn't match tree dimension {}",
                                n, self.inner.dim
                            )));
                        }
                    } else {
                        return Err(PyValueError::new_err(
                            "queries must be a scalar, list, or Array"
                        ));
                    }
                } else {
                    NdArray::from_vec(
                        Shape::new(vec![self.inner.n_points, self.inner.dim]),
                        self.inner.data.clone()
                    )
                };

                let result = self.inner.kernel_density_approx(&queries_arr, bandwidth, kernel_type, approx_criterion);

                if result.shape().dims()[0] == 1 {
                    Ok(result.as_slice()[0].into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(PyArray { inner: result }.into_pyobject(py)?.into_any().unbind())
                }
            }
        }
