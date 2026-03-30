use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;

use crate::array::{NdArray, Shape};
use crate::spatial::DistanceMetric;
use crate::spatial::spatial_index::{SpatialIndex, TreeType, QueryInput, QueryResult};
use super::{PyArray, ArrayData, ArrayLike};
use super::spatial::{
    PySpatialResult,
    parse_metric, parse_kernel, parse_vantage_selection, parse_projection_type,
};

// =============================================================================
// TreeType Enum
// =============================================================================

#[pyclass(name = "TreeType", module = "ironforest._core.spatial", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PyTreeType {
    Auto = 0,
    KDTree = 1,
    BallTree = 2,
    VPTree = 3,
    RPTree = 4,
    BruteForce = 5,
}

fn parse_tree_type(s: &str) -> PyResult<PyTreeType> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(PyTreeType::Auto),
        "kd" | "kdtree" | "kd_tree" => Ok(PyTreeType::KDTree),
        "ball" | "balltree" | "ball_tree" => Ok(PyTreeType::BallTree),
        "vp" | "vptree" | "vp_tree" => Ok(PyTreeType::VPTree),
        "rp" | "rptree" | "rp_tree" => Ok(PyTreeType::RPTree),
        "brute" | "brute_force" | "bruteforce" => Ok(PyTreeType::BruteForce),
        _ => Err(PyValueError::new_err(format!(
            "Unknown tree type '{}'. Valid options: 'auto', 'kd', 'ball', 'vp', 'rp', 'brute_force'",
            s
        ))),
    }
}

fn resolve_tree_type(tt: PyTreeType) -> TreeType {
    match tt {
        PyTreeType::Auto | PyTreeType::KDTree => TreeType::KDTree,
        PyTreeType::BallTree => TreeType::BallTree,
        PyTreeType::VPTree => TreeType::VPTree,
        PyTreeType::RPTree => TreeType::RPTree,
        PyTreeType::BruteForce => TreeType::BruteForce,
    }
}

fn tree_type_to_py(tt: TreeType) -> PyTreeType {
    match tt {
        TreeType::KDTree => PyTreeType::KDTree,
        TreeType::BallTree => PyTreeType::BallTree,
        TreeType::VPTree => PyTreeType::VPTree,
        TreeType::RPTree => PyTreeType::RPTree,
        TreeType::BruteForce => PyTreeType::BruteForce,
    }
}

fn to_py_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn query_result_to_py(qr: QueryResult) -> PySpatialResult {
    match (qr.counts, qr.k) {
        (Some(counts), _) => PySpatialResult::from_batch_radius(qr.indices, qr.distances, counts),
        (None, Some(k)) => PySpatialResult::from_batch_knn(qr.indices, qr.distances, qr.n_queries, k),
        (None, None) => PySpatialResult::from_single(qr.indices, qr.distances),
    }
}

// =============================================================================
// PySpatialIndex
// =============================================================================

#[pyclass(name = "SpatialIndex", module = "ironforest._core.spatial")]
pub struct PySpatialIndex {
    inner: SpatialIndex,
}

#[pymethods]
impl PySpatialIndex {
    #[new]
    #[pyo3(signature = (
        data,
        tree_type = "auto",
        leaf_size = 20,
        metric = "euclidean",
        rebuild_threshold = 1000,
        seed = 0,
        projection = "gaussian",
        selection = "variance",
        copy = true,
    ))]
    fn __init__(
        data: ArrayLike,
        tree_type: &str,
        leaf_size: usize,
        metric: &str,
        rebuild_threshold: usize,
        seed: u64,
        projection: &str,
        selection: &str,
        copy: bool,
    ) -> PyResult<Self> {
        let parsed_type = resolve_tree_type(parse_tree_type(tree_type)?);
        let metric = parse_metric(metric)?;
        let vp_selection = parse_vantage_selection(selection)?;
        let use_f32 = data.is_f32();

        if use_f32 {
            let arr = if copy { data.into_f32_ndarray()?.to_contiguous() } else { data.into_f32_ndarray()? };
            let dim = arr.shape().dims()[1];
            let projection_type = parse_projection_type(projection, 1.0 / f64::sqrt(dim as f64))?;
            Ok(PySpatialIndex {
                inner: SpatialIndex::new_f32(arr, parsed_type, leaf_size, metric, rebuild_threshold, seed, projection_type, vp_selection),
            })
        } else {
            let arr = if copy { data.into_ndarray()?.to_contiguous() } else { data.into_ndarray()? };
            let dim = arr.shape().dims()[1];
            let projection_type = parse_projection_type(projection, 1.0 / f64::sqrt(dim as f64))?;
            Ok(PySpatialIndex {
                inner: SpatialIndex::new_f64(arr, parsed_type, leaf_size, metric, rebuild_threshold, seed, projection_type, vp_selection),
            })
        }
    }

    // =========================================================================
    // Dynamic Insertion
    // =========================================================================

    fn insert(&mut self, points: ArrayLike) -> PyResult<()> {
        let ndim = points.ndim();
        if self.inner.use_f32() {
            let arr = points.into_f32_ndarray()?;
            let shape = arr.shape().dims().to_vec();
            let point_dim = if ndim == 1 { shape[0] } else { shape[1] };
            self.inner.insert_f32(arr.as_slice_unchecked(), point_dim).map_err(to_py_err)
        } else {
            let arr = points.into_ndarray()?;
            let shape = arr.shape().dims().to_vec();
            let point_dim = if ndim == 1 { shape[0] } else { shape[1] };
            self.inner.insert_f64(arr.as_slice_unchecked(), point_dim).map_err(to_py_err)
        }
    }

    fn flush(&mut self) -> PyResult<()> {
        self.inner.flush().map_err(to_py_err)
    }

    // =========================================================================
    // Properties
    // =========================================================================

    #[getter]
    fn tree_type(&self) -> PyResult<String> {
        match self.inner.tree_type(){
            TreeType::KDTree => Ok("kd_tree".to_owned()),
            TreeType::BallTree => Ok("ball_tree".to_owned()),
            TreeType::VPTree => Ok("vp_tree".to_owned()),
            TreeType::RPTree => Ok("rp_tree".to_owned()),
            TreeType::BruteForce => Ok("brute_force".to_owned()),
        }
    }

    #[getter]
    fn dtype(&self) -> &str {
        if self.inner.use_f32() { "float32" } else { "float64" }
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_points(&self) -> PyResult<usize> {
        self.inner.n_points().map_err(to_py_err)
    }

    #[getter]
    fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    #[getter]
    fn metric(&self) -> &str {
        match self.inner.metric() {
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::Manhattan => "manhattan",
            DistanceMetric::Chebyshev => "chebyshev",
            DistanceMetric::Cosine => "cosine",
        }
    }

    #[getter]
    fn get_rebuild_threshold(&self) -> usize {
        self.inner.rebuild_threshold()
    }

    #[setter]
    fn set_rebuild_threshold(&mut self, value: usize) {
        self.inner.set_rebuild_threshold(value);
    }

    // =========================================================================
    // Queries
    // =========================================================================

    #[pyo3(signature = (query, k))]
    fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<PySpatialResult> {
        let is_batch = query.ndim() == 2;
        let dim = self.inner.dim();
        if self.inner.use_f32() {
            let q = query.into_f32_spatial_query_ndarray(dim)?;
            let result = self.inner.query_knn(QueryInput::F32(&q), is_batch, k).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        } else {
            let q = query.into_spatial_query_ndarray(dim)?;
            let result = self.inner.query_knn(QueryInput::F64(&q), is_batch, k).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        }
    }

    #[pyo3(signature = (query, k, n_candidates=None, n_probes=None))]
    fn query_ann(
        &self,
        query: ArrayLike,
        k: usize,
        n_candidates: Option<usize>,
        n_probes: Option<usize>,
    ) -> PyResult<PySpatialResult> {
        let n_candidates = n_candidates.unwrap_or(k * 2);
        let is_batch = query.ndim() == 2;
        let dim = self.inner.dim();
        if self.inner.use_f32() {
            let q = query.into_f32_spatial_query_ndarray(dim)?;
            let result = self.inner.query_ann(QueryInput::F32(&q), is_batch, k, n_candidates, n_probes).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        } else {
            let q = query.into_spatial_query_ndarray(dim)?;
            let result = self.inner.query_ann(QueryInput::F64(&q), is_batch, k, n_candidates, n_probes).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        }
    }

    #[pyo3(signature = (query, radius))]
    fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<PySpatialResult> {
        let is_batch = query.ndim() == 2;
        let dim = self.inner.dim();
        if self.inner.use_f32() {
            let q = query.into_f32_spatial_query_ndarray(dim)?;
            let result = self.inner.query_radius(QueryInput::F32(&q), is_batch, radius).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        } else {
            let q = query.into_spatial_query_ndarray(dim)?;
            let result = self.inner.query_radius(QueryInput::F64(&q), is_batch, radius).map_err(to_py_err)?;
            Ok(query_result_to_py(result))
        }
    }

    #[pyo3(signature = (queries=None, bandwidth=None, kernel=None, normalize=None))]
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
        let do_normalize = normalize.unwrap_or(false);
        let dim = self.inner.dim();

        let result = if self.inner.use_f32() {
            match queries {
                Some(q) => {
                    let q_arr = q.into_f32_spatial_query_ndarray(dim)?;
                    self.inner.kernel_density(Some(QueryInput::F32(&q_arr)), bandwidth, kernel_type, do_normalize).map_err(to_py_err)?
                }
                None => {
                    self.inner.kernel_density(None, bandwidth, kernel_type, do_normalize).map_err(to_py_err)?
                }
            }
        } else {
            match queries {
                Some(q) => {
                    let q_arr = q.into_spatial_query_ndarray(dim)?;
                    self.inner.kernel_density(Some(QueryInput::F64(&q_arr)), bandwidth, kernel_type, do_normalize).map_err(to_py_err)?
                }
                None => {
                    self.inner.kernel_density(None, bandwidth, kernel_type, do_normalize).map_err(to_py_err)?
                }
            }
        };

        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice_unchecked()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result), alive: true }.into_pyobject(py)?.into_any().unbind())
        }
    }

    #[pyo3(signature = (indices=None))]
    fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
        let idx_vec: Option<Vec<i64>> = match indices {
            Some(idx) => {
                let idx_arr = idx.into_i64_ndarray()?;
                Some(idx_arr.as_slice_unchecked().to_vec())
            }
            None => None,
        };

        let (data, n_rows, n_cols) = self.inner.data(idx_vec.as_deref()).map_err(to_py_err)?;
        Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_rows, n_cols]), data)),
            alive: true,
        })
    }
}
