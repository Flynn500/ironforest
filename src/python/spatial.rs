use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;
use crate::Generator;
use crate::array::{NdArray, Shape};
use crate::projection::{ProjectionReducer, ProjectionType};
use crate::spatial::trees::{AggTree, BallTree, BruteForce, KDTree, RPTree, VPTree, VantagePointSelection,
    KDTree32, BallTree32, VPTree32, BruteForce32, RPTree32, AggTree32};
use crate::spatial::{DistanceMetric, KernelType, SpatialTree};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery, AnnQuery};
use super::{PyArray, ArrayData, ArrayLike};
use pyo3::types::PyBytes;
use rmp_serde;
use std::io::{Write, Read};
use num_traits::{ToPrimitive, NumCast};

// =============================================================================
// Result Type
// =============================================================================
//
// Spatial Result object returns spatial queries in a more ergonomic format. Can
// immediately get statistics, centroid etc. and split batch queries without the
// headache of dealing with counts (for radius mainly). 

#[pyclass(name = "SpatialResult")]
pub struct PySpatialResult {
    #[pyo3(get)]
    indices: PyArray,
    #[pyo3(get)]
    distances: PyArray,
    #[pyo3(get)]
    counts: Option<PyArray>,
    n_queries: usize,
    k: Option<usize>,
}

impl PySpatialResult {
    pub fn from_single(indices: Vec<i64>, distances: Vec<f64>) -> Self {
        let n = indices.len();
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n), indices)), alive: true },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), distances)), alive: true },
            counts: None,
            n_queries: 1,
            k: None,
        }
    }

    pub fn from_batch_knn(indices: Vec<i64>, distances: Vec<f64>, n_queries: usize, k: usize) -> Self {
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::new(vec![n_queries, k]), indices)), alive: true },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, k]), distances)), alive: true },
            counts: None,
            n_queries,
            k: Some(k),
        }
    }

    pub fn from_batch_radius(indices: Vec<i64>, distances: Vec<f64>, counts: Vec<i64>) -> Self {
        let n_queries = counts.len();
        let total = indices.len();
        PySpatialResult {
            indices: PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(total), indices)), alive: true },
            distances: PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(total), distances)), alive: true },
            counts: Some(PyArray { inner: ArrayData::Int(NdArray::from_vec(Shape::d1(n_queries), counts)), alive: true }),
            n_queries,
            k: None,
        }
    }

    /// Returns (offset, length) pairs for each query's result slice.
    fn query_result_ranges(&self) -> Vec<(usize, usize)> {
        if self.n_queries == 1 {
            let total = self.distances.as_float().unwrap().as_slice_unchecked().len();
            vec![(0, total)]
        } else if let Some(k) = self.k {
            (0..self.n_queries).map(|i| (i * k, k)).collect()
        } else {
            let counts = self.counts.as_ref().unwrap();
            let counts_slice = counts.as_int().unwrap().as_slice_unchecked();
            let mut offset = 0;
            counts_slice.iter().map(|&c| {
                let c = c as usize;
                let range = (offset, c);
                offset += c;
                range
            }).collect()
        }
    }

    fn per_query_distances(&self) -> Vec<&[f64]> {
        let dist_slice = self.distances.as_float().unwrap().as_slice_unchecked();
        self.query_result_ranges().into_iter()
            .map(|(off, len)| &dist_slice[off..off + len])
            .collect()
    }

    fn per_query_indices(&self) -> Vec<&[i64]> {
        let idx_slice = self.indices.as_int().unwrap().as_slice_unchecked();
        self.query_result_ranges().into_iter()
            .map(|(off, len)| &idx_slice[off..off + len])
            .collect()
    }

    fn aggregate_per_query<F: Fn(&NdArray<f64>) -> f64>(&self, f: F) -> Vec<f64> {
        self.per_query_distances().iter().map(|d| {
            if d.is_empty() { return f64::NAN; }
            f(&NdArray::from_vec(Shape::d1(d.len()), d.to_vec()))
        }).collect()
    }
}

#[pymethods]
impl PySpatialResult {
    fn count(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let counts: Vec<f64> = self.per_query_distances().iter()
            .map(|d| d.len() as f64)
            .collect();
        scalar_or_array(py, counts, self.n_queries == 1)
    }

    fn split(&self) -> Vec<PySpatialResult> {
        let idx_chunks = self.per_query_indices();
        let dist_chunks = self.per_query_distances();

        idx_chunks.into_iter().zip(dist_chunks)
            .map(|(idx, dist)| {
                PySpatialResult::from_single(idx.to_vec(), dist.to_vec())
            })
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.indices.as_int().unwrap().as_slice_unchecked().is_empty()
    }

    fn min(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.min()), self.n_queries == 1)
    }

    fn max(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.max()), self.n_queries == 1)
    }

    fn radius(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.max(py)
    }
    
    fn mean(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.mean()), self.n_queries == 1)
    }

    fn median(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.median()), self.n_queries == 1)
    }

    fn var(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.var()), self.n_queries == 1)
    }

    fn std(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        scalar_or_array(py, self.aggregate_per_query(|a| a.std()), self.n_queries == 1)
    }

    fn quantile(&self, py: Python<'_>, q: f64) -> PyResult<Py<PyAny>> {
        if !(0.0..=1.0).contains(&q) {
            return Err(pyo3::exceptions::PyValueError::new_err("quantile must be between 0 and 1"));
        }
        scalar_or_array(py, self.aggregate_per_query(|a| a.quantile(q)), self.n_queries == 1)
    }

    fn centroid(&self, data: &PyArray) -> PyResult<PyArray> {
        let data_arr = data.as_float()?;
        let data_slice = data_arr.as_slice_unchecked();
        let dim = data_arr.shape().dims()[1];
        let chunks = self.per_query_indices();

        let mut result = Vec::with_capacity(chunks.len() * dim);
        for indices in &chunks {
            if indices.is_empty() {
                result.extend(std::iter::repeat(f64::NAN).take(dim));
                continue;
            }
            let mut centroid = vec![0.0f64; dim];
            for &idx in *indices {
                let row = &data_slice[idx as usize * dim..(idx as usize + 1) * dim];
                for (c, &v) in centroid.iter_mut().zip(row) {
                    *c += v;
                }
            }
            let n = indices.len() as f64;
            for c in &mut centroid {
                *c /= n;
            }
            result.extend(centroid);
        }

        let n_queries = chunks.len();
        if self.n_queries == 1 {
            Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::d1(dim), result)), alive: true })
        } else {
            Ok(PyArray { inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_queries, dim]), result)), alive: true })
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

//macro for accessing tree inner. Has been changed to an option for serialization
macro_rules! tree {
    ($self:expr) => {
        $self.inner.as_ref().ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?
    };
}

enum SpatialInner<T64, T32> {
    F64(T64),
    F32(T32),
}

/// Returns a Python scalar for single queries, or a 1-D `PyArray` for batch queries.
fn scalar_or_array(py: Python<'_>, values: Vec<f64>, is_single: bool) -> PyResult<Py<PyAny>> {
    if is_single {
        Ok(values[0].into_pyobject(py)?.into_any().unbind())
    } else {
        let n = values.len();
        Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(Shape::d1(n), values)),
            alive: true
        }.into_pyobject(py)?.into_any().unbind())
    }
}

/// Recovers the original-order point matrix from a tree's internal (permuted) storage.
///
/// Trees reorder their input data during construction for cache efficiency. This function
/// undoes that permutation so `data()` always returns rows in the order the user inserted them.
fn get_tree_data(
    tree_indices: &[usize],
    raw_data: &[f64],
    n_points: usize,
    dim: usize,
    indices: Option<ArrayLike>,
) -> PyResult<PyArray> {
    

    let mut original_data = vec![0.0f64; n_points * dim];
    for (tree_pos, &orig_idx) in tree_indices.iter().enumerate() {
        original_data[orig_idx * dim..(orig_idx + 1) * dim]
            .copy_from_slice(&raw_data[tree_pos * dim..(tree_pos + 1) * dim]);
    }

    match indices {
        None => Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(
                Shape::new(vec![n_points, dim]),
                original_data,
            )),
            alive: true
        }),
        Some(idx) => {
            let idx_arr = idx.into_i64_ndarray()?;
            let k = idx_arr.len();
            let mut result = Vec::with_capacity(k * dim);
            for &orig_idx in idx_arr.as_slice_unchecked() {
                let i = orig_idx as usize;
                if i >= n_points {
                    return Err(PyValueError::new_err(format!(
                        "Index {} out of bounds for tree with {} points",
                        orig_idx, n_points
                    )));
                }
                result.extend_from_slice(&original_data[i * dim..(i + 1) * dim]);
            }
            Ok(PyArray {
                inner: ArrayData::Float(NdArray::from_vec(
                    Shape::new(vec![k, dim]),
                    result,
                )),
                alive: true
            })
        }
    }
}

// =============================================================================
// Parsing
// =============================================================================

fn parse_metric(metric: &str) -> PyResult<DistanceMetric> {
    match metric.to_lowercase().as_str() {
        "euclidean" => Ok(DistanceMetric::Euclidean),
        "manhattan" => Ok(DistanceMetric::Manhattan),
        "chebyshev" => Ok(DistanceMetric::Chebyshev),
        "cosine" => Ok(DistanceMetric::Cosine),
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
        "variance" => Ok(VantagePointSelection::Variance { sample_size: 10 }), //need to expose param or set smarter default
        _ => Err(PyValueError::new_err(format!(
            "Unknown vantage point selection method '{}'. Valid options: 'first', 'random'",
            selection
        ))),
    }
}

fn parse_projection_type(projection: &str, density: f64) -> PyResult<ProjectionType> {
    match projection.to_lowercase().as_str() {
        "gaussian" => Ok(ProjectionType::Gaussian),
        "sparse" => Ok(ProjectionType::Sparse(density)),
        // "achlioptas" => Ok(ProjectionType::Achlioptas),
        _ => Err(PyValueError::new_err(format!(
            "Unknown projection type '{}'. Valid options: 'gaussian'",
            projection
        ))),
    }
}

// =============================================================================
// Query Macros
// =============================================================================

// Generates query_radius, query_knn, kernel_density, and data methods for any
// spatial index type whose inner field implements SpatialQuery. All four tree
// types (BallTree, KDTree, VPTree, BruteForce) use this macro; AggTree is
// excluded because its kernel_density signature differs. M tree is excluded
// because underlying data is managed differently so KDE without params becomes
// difficult.
macro_rules! knn_body {
    ($tree:expr, $queries_arr:expr, $is_batch:expr, $k:expr) => {{
        let n_queries = $queries_arr.shape().dims()[0];
        if $is_batch {
            let results = $tree.query_knn_batch(&$queries_arr, $k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap()))
                .unzip();
            Ok(PySpatialResult::from_batch_knn(indices, distances, n_queries, $k))
        } else {
            let query_slice = &$queries_arr.as_slice_unchecked()[..$tree.dim];
            let results = $tree.query_knn(query_slice, $k);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
            Ok(PySpatialResult::from_single(indices, distances))
        }
    }};
}

macro_rules! ann_body {
    ($tree:expr, $queries_arr:expr, $is_batch:expr, $k:expr, $n_candidates:expr, $n_probes:expr) => {{
        let n_queries = $queries_arr.shape().dims()[0];
        if $is_batch {
            let results = match $n_probes {
                Some(n_probes) => $tree.query_ann_stochastic_batch(&$queries_arr, $k, $n_candidates, n_probes),
                None => $tree.query_ann_batch(&$queries_arr, $k, $n_candidates),
            };
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .flatten()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap()))
                .unzip();
            Ok(PySpatialResult::from_batch_knn(indices, distances, n_queries, $k))
        } else {
            let query_slice = &$queries_arr.as_slice_unchecked()[..$tree.dim];
            let results = match $n_probes {
                Some(n_probes) => $tree.query_ann_stochastic(query_slice, $k, $n_candidates, n_probes),
                None => $tree.query_ann(query_slice, $k, $n_candidates),
            };
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
            Ok(PySpatialResult::from_single(indices, distances))
        }
    }};
}

macro_rules! radius_body {
    ($tree:expr, $queries_arr:expr, $is_batch:expr, $rad:expr) => {{
        if $is_batch {
            let results = $tree.query_radius_batch(&$queries_arr, $rad);
            let mut all_indices = Vec::new();
            let mut all_distances = Vec::new();
            let mut counts = Vec::with_capacity(results.len());
            for batch in results {
                counts.push(batch.len() as i64);
                for (i, d) in batch {
                    all_indices.push(i as i64);
                    all_distances.push(d.to_f64().unwrap());
                }
            }
            Ok(PySpatialResult::from_batch_radius(all_indices, all_distances, counts))
        } else {
            let query_slice = &$queries_arr.as_slice_unchecked()[..$tree.dim];
            let results = $tree.query_radius(query_slice, $rad);
            let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
            Ok(PySpatialResult::from_single(indices, distances))
        }
    }};
}

macro_rules! kde_body {
    ($tree:expr, $queries_arr:expr, $bandwidth:expr, $kernel_type:expr, $normalize:expr, $py:expr) => {{
        let result = $tree.kernel_density(&$queries_arr, $bandwidth, $kernel_type, $normalize);
        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice_unchecked()[0].into_pyobject($py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result), alive: true }.into_pyobject($py)?.into_any().unbind())
        }
    }};
}

macro_rules! impl_knn_query {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            #[pyo3(signature = (query, k))]
            fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<PySpatialResult> {
                let is_batch = query.ndim() == 2;
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                match inner {
                    SpatialInner::F64(tree) => {
                        let q = query.into_spatial_query_ndarray(tree.dim)?;
                        knn_body!(tree, q, is_batch, k)
                    }
                    SpatialInner::F32(tree) => {
                        let q = query.into_f32_spatial_query_ndarray(tree.dim)?;
                        knn_body!(tree, q, is_batch, k)
                    }
                }
            }
        }
    };
}

macro_rules! impl_ann_query {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
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
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                match inner {
                    SpatialInner::F64(tree) => {
                        let q = query.into_spatial_query_ndarray(tree.dim)?;
                        ann_body!(tree, q, is_batch, k, n_candidates, n_probes)
                    }
                    SpatialInner::F32(tree) => {
                        let q = query.into_f32_spatial_query_ndarray(tree.dim)?;
                        ann_body!(tree, q, is_batch, k, n_candidates, n_probes)
                    }
                }
            }
        }
    };
}

macro_rules! impl_radius_query {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            #[pyo3(signature = (query, radius))]
            fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<PySpatialResult> {
                let is_batch = query.ndim() == 2;
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                match inner {
                    SpatialInner::F64(tree) => {
                        let q = query.into_spatial_query_ndarray(tree.dim)?;
                        radius_body!(tree, q, is_batch, radius)
                    }
                    SpatialInner::F32(tree) => {
                        let q = query.into_f32_spatial_query_ndarray(tree.dim)?;
                        let rad: f32 = <f32 as NumCast>::from(radius).unwrap();
                        radius_body!(tree, q, is_batch, rad)
                    }
                }
            }
        }
    };
}

macro_rules! impl_kde_query {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
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
                let normalize = normalize.unwrap_or(false);
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                match inner {
                    SpatialInner::F64(tree) => {
                        let queries_arr = if let Some(q) = queries {
                            q.into_spatial_query_ndarray(tree.dim)?
                        } else {
                            NdArray::from_vec(
                                Shape::new(vec![tree.n_points, tree.dim]),
                                tree.data().to_vec()
                            )
                        };
                        kde_body!(tree, queries_arr, bandwidth, kernel_type, normalize, py)
                    }
                    SpatialInner::F32(tree) => {
                        let queries_arr = if let Some(q) = queries {
                            q.into_f32_spatial_query_ndarray(tree.dim)?
                        } else {
                            NdArray::from_vec(
                                Shape::new(vec![tree.n_points, tree.dim]),
                                tree.data().to_vec()
                            )
                        };
                        kde_body!(tree, queries_arr, bandwidth, kernel_type, normalize, py)
                    }
                }
            }
        }
    };
}

macro_rules! impl_data_query {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            #[pyo3(signature = (indices=None))]
            fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                match inner {
                    SpatialInner::F64(tree) => get_tree_data(tree.indices(), tree.data(), tree.n_points, tree.dim, indices),
                    SpatialInner::F32(tree) => get_tree_data_f32(tree.indices(), tree.data(), tree.n_points, tree.dim, indices),
                }
            }
        }
    };
}

macro_rules! impl_dtype_getter {
    ($py_type:ty) => {
        #[pymethods]
        impl $py_type {
            #[getter]
            fn dtype(&self) -> &str {
                match self.inner.as_ref() {
                    Some(SpatialInner::F64(_)) => "float64",
                    Some(SpatialInner::F32(_)) => "float32",
                    None => "uninitialized",
                }
            }
        }
    };
}

impl_data_query!(PyBallTree);
impl_data_query!(PyKDTree);
impl_data_query!(PyVPTree);
impl_data_query!(PyBruteForce);
impl_data_query!(PyRPTree);

impl_ann_query!(PyBallTree);
impl_ann_query!(PyKDTree);
impl_ann_query!(PyRPTree);

impl_knn_query!(PyBallTree);
impl_knn_query!(PyKDTree);
impl_knn_query!(PyVPTree);
impl_knn_query!(PyBruteForce);
impl_knn_query!(PyRPTree);

impl_radius_query!(PyBallTree);
impl_radius_query!(PyKDTree);
impl_radius_query!(PyVPTree);
impl_radius_query!(PyBruteForce);
impl_radius_query!(PyRPTree);

impl_kde_query!(PyBallTree);
impl_kde_query!(PyKDTree);
impl_kde_query!(PyVPTree);
impl_kde_query!(PyBruteForce);

impl_dtype_getter!(PyBallTree);
impl_dtype_getter!(PyKDTree);
impl_dtype_getter!(PyVPTree);
impl_dtype_getter!(PyBruteForce);
impl_dtype_getter!(PyRPTree);
impl_dtype_getter!(PyAggTree);


// =============================================================================
// Serialization Macro
// =============================================================================
//
// Generates __get_state__ and __set_state__ methods for pickle compatibility and
// a custom save and load method for direct saving and loading of spatial trees.


macro_rules! impl_spatial_serialization {
    ($py_type:ty, $t64:ty, $t32:ty, $constructor:ident) => {
        #[pymethods]
        impl $py_type {
            fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                let mut bytes = Vec::new();
                match inner {
                    SpatialInner::F64(tree) => {
                        bytes.push(0u8);
                        bytes.extend(rmp_serde::to_vec(tree).map_err(|e| PyValueError::new_err(e.to_string()))?);
                    }
                    SpatialInner::F32(tree) => {
                        bytes.push(1u8);
                        bytes.extend(rmp_serde::to_vec(tree).map_err(|e| PyValueError::new_err(e.to_string()))?);
                    }
                }
                Ok(PyBytes::new(py, &bytes))
            }

            fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
                let bytes = state.as_bytes();
                if bytes.is_empty() {
                    return Err(PyValueError::new_err("Empty state"));
                }
                let tag = bytes[0];
                let payload = &bytes[1..];
                self.inner = Some(match tag {
                    0 => SpatialInner::F64(rmp_serde::from_slice::<$t64>(payload).map_err(|e| PyValueError::new_err(e.to_string()))?),
                    1 => SpatialInner::F32(rmp_serde::from_slice::<$t32>(payload).map_err(|e| PyValueError::new_err(e.to_string()))?),
                    _ => return Err(PyValueError::new_err(format!("Unknown dtype tag: {}", tag))),
                });
                Ok(())
            }

            fn save(&self, path: &str) -> PyResult<()> {
                let inner = self.inner.as_ref()
                    .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
                let mut bytes = Vec::new();
                match inner {
                    SpatialInner::F64(tree) => {
                        bytes.push(0u8);
                        bytes.extend(rmp_serde::to_vec(tree).map_err(|e| PyValueError::new_err(e.to_string()))?);
                    }
                    SpatialInner::F32(tree) => {
                        bytes.push(1u8);
                        bytes.extend(rmp_serde::to_vec(tree).map_err(|e| PyValueError::new_err(e.to_string()))?);
                    }
                }
                std::fs::File::create(path)
                    .and_then(|mut f| f.write_all(&bytes))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            #[staticmethod]
            fn load(path: &str) -> PyResult<$py_type> {
                let mut bytes = Vec::new();
                std::fs::File::open(path)
                    .and_then(|mut f| f.read_to_end(&mut bytes))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                if bytes.is_empty() {
                    return Err(PyValueError::new_err("Empty file"));
                }
                let tag = bytes[0];
                let payload = &bytes[1..];
                let inner = match tag {
                    0 => SpatialInner::F64(rmp_serde::from_slice::<$t64>(payload).map_err(|e| PyValueError::new_err(e.to_string()))?),
                    1 => SpatialInner::F32(rmp_serde::from_slice::<$t32>(payload).map_err(|e| PyValueError::new_err(e.to_string()))?),
                    _ => return Err(PyValueError::new_err(format!("Unknown dtype tag: {}", tag))),
                };
                Ok($constructor { inner: Some(inner) })
            }
        }
    };
}

macro_rules! impl_simple_serialization {
    ($py_type:ty, $inner_type:ty, $constructor:ident) => {
        #[pymethods]
        impl $py_type {
            fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                let bytes = rmp_serde::to_vec(tree!(self))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyBytes::new(py, &bytes))
            }

            fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
                self.inner = Some(
                    rmp_serde::from_slice(state.as_bytes())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                );
                Ok(())
            }

            fn save(&self, path: &str) -> PyResult<()> {
                let bytes = rmp_serde::to_vec(tree!(self))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                std::fs::File::create(path)
                    .and_then(|mut f| f.write_all(&bytes))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            }

            #[staticmethod]
            fn load(path: &str) -> PyResult<$py_type> {
                let mut bytes = Vec::new();
                std::fs::File::open(path)
                    .and_then(|mut f| f.read_to_end(&mut bytes))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let inner = rmp_serde::from_slice(&bytes)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok($constructor { inner: Some(inner) })
            }
        }
    };
}

impl_spatial_serialization!(PyBallTree, BallTree, BallTree32, PyBallTree);
impl_spatial_serialization!(PyKDTree, KDTree, KDTree32, PyKDTree);
impl_spatial_serialization!(PyVPTree, VPTree, VPTree32, PyVPTree);
impl_spatial_serialization!(PyBruteForce, BruteForce, BruteForce32, PyBruteForce);
impl_spatial_serialization!(PyAggTree, AggTree, AggTree32, PyAggTree);
impl_spatial_serialization!(PyRPTree, RPTree, RPTree32, PyRPTree);
impl_simple_serialization!(PyProjectionReducer, ProjectionReducer, PyProjectionReducer);

// =============================================================================
// Tree Types
// =============================================================================

#[pyclass(name = "BallTree")]
pub struct PyBallTree {
    inner: Option<SpatialInner<BallTree, BallTree32>>,
}

#[pymethods]
impl PyBallTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", preserve_array=true))]
    fn from_array(mut array: PyRefMut<'_, PyArray>, leaf_size: Option<usize>, metric: Option<&str>, preserve_array: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyBallTree { inner: Some(SpatialInner::F64(BallTree::new(data, leaf_size, metric))) })
    }

    #[new]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", copy=true))]
    fn __init__(array: ArrayLike, leaf_size: Option<usize>, metric: Option<&str>, copy: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            Ok(PyBallTree { inner: Some(SpatialInner::F32(BallTree32::new(data, leaf_size, metric))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            Ok(PyBallTree { inner: Some(SpatialInner::F64(BallTree::new(data, leaf_size, metric))) })
        }
    }
}

#[pyclass(name = "KDTree")]
pub struct PyKDTree {
    inner: Option<SpatialInner<KDTree, KDTree32>>,
}

#[pymethods]
impl PyKDTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", preserve_array=true))]
    fn from_array(mut array: PyRefMut<'_, PyArray>, leaf_size: Option<usize>, metric: Option<&str>, preserve_array: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyKDTree { inner: Some(SpatialInner::F64(KDTree::new(data, leaf_size, metric))) })
    }

    #[new]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", copy=true))]
    fn __init__(array: ArrayLike, leaf_size: Option<usize>, metric: Option<&str>, copy: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            Ok(PyKDTree { inner: Some(SpatialInner::F32(KDTree32::new(data, leaf_size, metric))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            Ok(PyKDTree { inner: Some(SpatialInner::F64(KDTree::new(data, leaf_size, metric))) })
        }
    }
}

#[pyclass(name = "VPTree")]
pub struct PyVPTree {
    inner: Option<SpatialInner<VPTree, VPTree32>>,
}

#[pymethods]
impl PyVPTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", selection="variance", preserve_array=true))]
    fn from_array(mut array: PyRefMut<'_, PyArray>, leaf_size: Option<usize>, metric: Option<&str>, selection: Option<&str>, preserve_array: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let selection_method = parse_vantage_selection(selection.unwrap_or("random"))?;
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyVPTree { inner: Some(SpatialInner::F64(VPTree::new(data, leaf_size, metric, selection_method))) })
    }

    #[new]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", selection="variance", copy=true))]
    fn __init__(array: ArrayLike, leaf_size: Option<usize>, metric: Option<&str>, selection: Option<&str>, copy: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let selection_method = parse_vantage_selection(selection.unwrap_or("random"))?;
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            Ok(PyVPTree { inner: Some(SpatialInner::F32(VPTree32::new(data, leaf_size, metric, selection_method))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            Ok(PyVPTree { inner: Some(SpatialInner::F64(VPTree::new(data, leaf_size, metric, selection_method))) })
        }
    }
}

#[pyclass(name = "RPTree")]
pub struct PyRPTree {
    inner: Option<SpatialInner<RPTree, RPTree32>>,
}

#[pymethods]
impl PyRPTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", projection="gaussian", seed=0, preserve_array=true))]
    fn from_array(mut array: PyRefMut<'_, PyArray>, leaf_size: Option<usize>, metric: Option<&str>, projection: Option<&str>, seed: u64, preserve_array: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let ndim = array.as_float()?.ndim() as f64;
        let projection_method = parse_projection_type(projection.unwrap_or("gaussian"), 1.0 / ndim.sqrt())?;
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyRPTree { inner: Some(SpatialInner::F64(RPTree::new(data, leaf_size, metric, projection_method, seed))) })
    }

    #[new]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", projection="gaussian", seed=0, copy=true))]
    fn __init__(array: ArrayLike, leaf_size: Option<usize>, metric: Option<&str>, projection: Option<&str>, seed: u64, copy: bool) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            let ndim = data.shape().dims()[1] as f64;
            let projection_method = parse_projection_type(projection.unwrap_or("gaussian"), 1.0 / ndim.sqrt())?;
            Ok(PyRPTree { inner: Some(SpatialInner::F32(RPTree32::new(data, leaf_size, metric, projection_method, seed))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            let projection_method = parse_projection_type(projection.unwrap_or("gaussian"), 1.0 / f64::sqrt(data.ndim() as f64))?;
            Ok(PyRPTree { inner: Some(SpatialInner::F64(RPTree::new(data, leaf_size, metric, projection_method, seed))) })
        }
    }
}

#[pyclass(name = "BruteForce")]
pub struct PyBruteForce {
    inner: Option<SpatialInner<BruteForce, BruteForce32>>,
}

#[pymethods]
impl PyBruteForce {
    #[staticmethod]
    #[pyo3(signature = (array, metric="euclidean", preserve_array=true))]
    fn from_array(mut array: PyRefMut<'_, PyArray>, metric: Option<&str>, preserve_array: bool) -> PyResult<Self> {
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyBruteForce { inner: Some(SpatialInner::F64(BruteForce::new(data, metric))) })
    }

    #[new]
    #[pyo3(signature = (array, metric="euclidean", copy=true))]
    fn __init__(array: ArrayLike, metric: Option<&str>, copy: bool) -> PyResult<Self> {
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            Ok(PyBruteForce { inner: Some(SpatialInner::F32(BruteForce32::new(data, metric))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            Ok(PyBruteForce { inner: Some(SpatialInner::F64(BruteForce::new(data, metric))) })
        }
    }
}

#[pyclass(name = "AggTree")]
pub struct PyAggTree {
    inner: Option<SpatialInner<AggTree, AggTree32>>,
}

#[pymethods]
impl PyAggTree {
    #[staticmethod]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", kernel="gaussian", bandwidth=1.0, atol=0.01, preserve_array=true))]
    fn from_array(
        mut array: PyRefMut<'_, PyArray>,
        leaf_size: Option<usize>,
        metric: Option<&str>,
        kernel: Option<&str>,
        bandwidth: Option<f64>,
        atol: Option<f64>,
        preserve_array: bool,
    ) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let kernel = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let bandwidth = bandwidth.unwrap_or(1.0);
        let atol = atol.unwrap_or(0.01);
        let data = if preserve_array {
            array.as_view_float()?
        } else {
            array.take_float()?.to_contiguous()
        };
        Ok(PyAggTree { inner: Some(SpatialInner::F64(AggTree::new(data, leaf_size, metric, kernel, bandwidth, atol))) })
    }

    #[new]
    #[pyo3(signature = (array, leaf_size=20, metric="euclidean", kernel="gaussian", bandwidth=1.0, atol=0.01, copy=true))]
    fn __init__(
        array: ArrayLike,
        leaf_size: Option<usize>,
        metric: Option<&str>,
        kernel: Option<&str>,
        bandwidth: Option<f64>,
        atol: Option<f64>,
        copy: bool,
    ) -> PyResult<Self> {
        let leaf_size = leaf_size.unwrap_or(20);
        let metric = parse_metric(metric.unwrap_or("euclidean"))?;
        let kernel = parse_kernel(kernel.unwrap_or("gaussian"))?;
        let bandwidth = bandwidth.unwrap_or(1.0);
        let atol = atol.unwrap_or(0.01);
        let use_f32 = array.is_f32();
        if use_f32 {
            let data = if copy { array.into_f32_ndarray()?.to_contiguous() } else { array.into_f32_ndarray()? };
            Ok(PyAggTree { inner: Some(SpatialInner::F32(AggTree32::new(data, leaf_size, metric, kernel, bandwidth, atol))) })
        } else {
            let data = if copy { array.into_ndarray()?.to_contiguous() } else { array.into_ndarray()? };
            Ok(PyAggTree { inner: Some(SpatialInner::F64(AggTree::new(data, leaf_size, metric, kernel, bandwidth, atol))) })
        }
    }

    #[pyo3(signature = (queries=None, normalize=true))]
    fn kernel_density(
        &self,
        py: Python<'_>,
        queries: Option<ArrayLike>,
        normalize: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let normalize = normalize.unwrap_or(false);
        let inner = self.inner.as_ref()
            .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
        let result = match inner {
            SpatialInner::F64(tree) => {
                let queries_arr = if let Some(q) = queries {
                    q.into_spatial_query_ndarray(tree.dim)?
                } else {
                    tree.data.clone()
                };
                tree.kernel_density(&queries_arr, normalize)
            }
            SpatialInner::F32(tree) => {
                let queries_arr = if let Some(q) = queries {
                    q.into_f32_spatial_query_ndarray(tree.dim)?
                } else {
                    tree.data.clone()
                };
                tree.kernel_density(&queries_arr, normalize)
            }
        };
        if result.shape().dims()[0] == 1 {
            Ok(result.as_slice_unchecked()[0].into_pyobject(py)?.into_any().unbind())
        } else {
            Ok(PyArray { inner: ArrayData::Float(result), alive: true }.into_pyobject(py)?.into_any().unbind())
        }
    }
}

// #[pyclass(name = "MTree")]
// pub struct PyMTree {
//     inner: Option<MTree>,
// }

// #[pymethods] //ide error
// impl PyMTree {
//     #[staticmethod]
//     #[pyo3(signature = (array, capacity=50, metric="euclidean"))]
//     fn from_array(array: &PyArray, capacity: Option<usize>, metric: Option<&str>) -> PyResult<Self> {
//         let capacity = capacity.unwrap_or(50);
//         let metric = parse_metric(metric.unwrap_or("euclidean"))?;
//         let tree = MTree::from_ndarray(array.as_float()?, capacity, metric);
//         Ok(PyMTree { inner: Some(tree) })
//     }

//     #[new]
//     #[pyo3(signature = (array, capacity=20, metric="euclidean"))]
//     fn __init__(
//         array: ArrayLike,
//         capacity: Option<usize>,
//         metric: Option<&str>,
//     ) -> PyResult<Self> {
//         let capacity = capacity.unwrap_or(20);
//         let metric = parse_metric(metric.unwrap_or("euclidean"))?;
//         let data = array.into_ndarray()?;
//         let tree = MTree::from_ndarray(&data, capacity, metric);
//         Ok(PyMTree { inner: Some(tree) })
//     }

//     fn insert(&mut self, point: ArrayLike) -> PyResult<()> {
//         let tree = self.inner.as_mut()
//             .ok_or_else(|| PyValueError::new_err("Tree is uninitialized"))?;
//         let arr = point.into_spatial_query_ndarray(tree.dim)?;
//         let point_idx = tree.n_points;
//         tree.insert(arr.as_slice_unchecked().to_vec(), point_idx);
//         Ok(())
//     }

//     #[pyo3(signature = (indices=None))]
//     fn data(&self, indices: Option<ArrayLike>) -> PyResult<PyArray> {
//         let tree = tree!(self);
//         let data = tree.collect_data();

//         match indices {
//             None => Ok(PyArray {
//                 inner: ArrayData::Float(NdArray::from_vec(
//                     Shape::new(vec![tree.n_points, tree.dim]),
//                     data,
//                 )),
//                 alive: true
//             }),
//             Some(idx) => {
//                 let idx_arr = idx.into_i64_ndarray()?;
//                 let k = idx_arr.len();
//                 let mut result = Vec::with_capacity(k * tree.dim);
//                 for &i in idx_arr.as_slice_unchecked() {
//                     let i = i as usize;
//                     if i >= tree.n_points {
//                         return Err(PyValueError::new_err(format!(
//                             "Index {} out of bounds for tree with {} points", i, tree.n_points
//                         )));
//                     }
//                     result.extend_from_slice(&data[i * tree.dim..(i + 1) * tree.dim]);
//                 }
//                 Ok(PyArray {
//                     inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![k, tree.dim]), result)),
//                     alive: true
//                 })
//             }
//         }
//     }

//     fn query_radius(&self, query: ArrayLike, radius: f64) -> PyResult<PySpatialResult> {
//         let tree = tree!(self);
//         let is_batch = query.ndim() == 2;
//         let queries_arr = query.into_spatial_query_ndarray(tree.dim)?;
//         if is_batch {
//             let results = tree.query_radius_batch(&queries_arr, radius);
//             let mut all_indices = Vec::new();
//             let mut all_distances = Vec::new();
//             let mut counts = Vec::with_capacity(results.len());
//             for batch in results {
//                 counts.push(batch.len() as i64);
//                 for (i, d) in batch {
//                     all_indices.push(i as i64);
//                     all_distances.push(d);
//                 }
//             }
//             Ok(PySpatialResult::from_batch_radius(all_indices, all_distances, counts))
//         } else {
//             let query_slice = &queries_arr.as_slice_unchecked()[..tree.dim];
//             let results = tree.query_radius(query_slice, radius);
//             let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
//                 .map(|(i, d)| (i as i64, d)).unzip();
//             Ok(PySpatialResult::from_single(indices, distances))
//         }
//     }

//     fn query_knn(&self, query: ArrayLike, k: usize) -> PyResult<PySpatialResult> {
//         let tree = tree!(self);
//         let is_batch = query.ndim() == 2;
//         let queries_arr = query.into_spatial_query_ndarray(tree.dim)?;
//         let n_queries = queries_arr.shape().dims()[0];
//         if is_batch {
//             let results = tree.query_knn_batch(&queries_arr, k);
//             let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
//                 .flatten()
//                 .map(|(i, d)| (i as i64, d))
//                 .unzip();
//             Ok(PySpatialResult::from_batch_knn(indices, distances, n_queries, k))
//         } else {
//             let query_slice = &queries_arr.as_slice_unchecked()[..tree.dim];
//             let results = tree.query_knn(query_slice, k);
//             let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
//                 .map(|(i, d)| (i as i64, d)).unzip();
//             Ok(PySpatialResult::from_single(indices, distances))
//         }
//     }

//     fn kernel_density(
//         &self,
//         py: Python<'_>,
//         queries: Option<ArrayLike>,
//         bandwidth: Option<f64>,
//         kernel: Option<&str>,
//         normalize: Option<bool>,
//     ) -> PyResult<Py<PyAny>> {
//         let tree = tree!(self);
//         let bandwidth = bandwidth.unwrap_or(1.0);
//         let kernel_type = parse_kernel(kernel.unwrap_or("gaussian"))?;
//         let normalize = normalize.unwrap_or(false);

//         let queries_arr = if let Some(q) = queries {
//             q.into_spatial_query_ndarray(tree.dim)?
//         } else {
//             NdArray::from_vec(
//                 Shape::new(vec![tree.n_points, tree.dim]),
//                 tree.collect_data(),
//             )
//         };

//         let result = tree.kernel_density(&queries_arr, bandwidth, kernel_type, normalize);

//         if result.shape().dims()[0] == 1 {
//             Ok(result.as_slice_unchecked()[0].into_pyobject(py)?.into_any().unbind())
//         } else {
//             Ok(PyArray { inner: ArrayData::Float(result), alive: true }.into_pyobject(py)?.into_any().unbind())
//         }
//     }
// }

// =============================================================================
// Misc Types
// =============================================================================
#[pyclass(name = "ProjectionReducer")]
pub struct PyProjectionReducer {
    inner: Option<ProjectionReducer>,
}

#[pymethods]
impl PyProjectionReducer {
    #[new]
    #[pyo3(signature = (input_dim, output_dim, projection_type="gaussian", density=0.1, seed=None))]
    pub fn __init__(
        input_dim: usize,
        output_dim: usize,
        projection_type: &str,
        density: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let p_type = parse_projection_type(projection_type, density)?;
        let mut rng = match seed {
            Some(s) => Generator::from_seed(s),
            None => Generator::new(),
        };

        let reducer = ProjectionReducer::fit(input_dim, output_dim, p_type, &mut rng);
        Ok(PyProjectionReducer { inner: Some(reducer) })
    }

    #[staticmethod]
    #[pyo3(signature = (data, output_dim, projection_type="gaussian", density=0.1, seed=None))]
    pub fn fit_transform(
        data: ArrayLike,
        output_dim: usize,
        projection_type: &str,
        density: f64,
        seed: Option<u64>,
    ) -> PyResult<(Self, PyArray)> {
        let p_type = parse_projection_type(projection_type, density)?;
        let mut rng = match seed {
            Some(s) => Generator::from_seed(s),
            None => Generator::new(),
        };
        let input_ndarray = data.into_ndarray()?;

        let (reducer, transformed) = ProjectionReducer::fit_transform(
            &input_ndarray,
            output_dim,
            p_type,
            &mut rng
        );

        Ok((
            PyProjectionReducer { inner: Some(reducer) },
            PyArray { inner: ArrayData::Float(transformed), alive: true }
        ))
    }

    pub fn transform(&self, data: ArrayLike) -> PyResult<PyArray> {
        let reducer = tree!(self); //tree macro just gets from option, maybe rename?
        let mut input_ndarray = data.into_ndarray()?;
        let current_dims = input_ndarray.shape().dims();
        
        //allows for users to input 1d arrays instead of ones of "correct" shape
        let was_1d = current_dims.len() == 1;
        if was_1d {
            let n_features = current_dims[0];
            input_ndarray = input_ndarray.reshape(vec![1, n_features]);
        }

        if input_ndarray.shape().dims()[1] != reducer.input_dim() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch: expected {}, got {}",
                reducer.input_dim(),
                input_ndarray.shape().dims()[1]
            )));
        }
        let transformed = reducer.transform(&input_ndarray);
        if was_1d {
            let len = transformed.as_slice_unchecked().len();
            return Ok(PyArray { inner: ArrayData::Float(transformed.reshape(vec![len])), alive: true });
        }

        Ok(PyArray { inner: ArrayData::Float(transformed), alive: true })
    }

    #[getter]
    pub fn input_dim(&self) -> PyResult<usize> {
        self.inner.as_ref()
            .map(|r| r.input_dim())
            .ok_or_else(|| PyValueError::new_err("Reducer not initialized"))
    }

    #[getter]
    pub fn output_dim(&self) -> PyResult<usize> {
        self.inner.as_ref()
            .map(|r| r.output_dim())
            .ok_or_else(|| PyValueError::new_err("Reducer not initialized"))
    }
}

// =============================================================================
// f32 Tree Support
// =============================================================================

/// Like `get_tree_data` but for f32-backed trees; upcasts to f64 for output since python is f64
fn get_tree_data_f32(
    tree_indices: &[usize],
    raw_data: &[f32],
    n_points: usize,
    dim: usize,
    indices: Option<ArrayLike>,
) -> PyResult<PyArray> {
    let mut original_data = vec![0.0f64; n_points * dim];
    for (tree_pos, &orig_idx) in tree_indices.iter().enumerate() {
        for d in 0..dim {
            original_data[orig_idx * dim + d] = raw_data[tree_pos * dim + d] as f64;
        }
    }

    match indices {
        None => Ok(PyArray {
            inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![n_points, dim]), original_data)),
            alive: true,
        }),
        Some(idx) => {
            let idx_arr = idx.into_i64_ndarray()?;
            let k = idx_arr.len();
            let mut result = Vec::with_capacity(k * dim);
            for &orig_idx in idx_arr.as_slice_unchecked() {
                let i = orig_idx as usize;
                if i >= n_points {
                    return Err(PyValueError::new_err(format!(
                        "Index {} out of bounds for tree with {} points", orig_idx, n_points
                    )));
                }
                result.extend_from_slice(&original_data[i * dim..(i + 1) * dim]);
            }
            Ok(PyArray {
                inner: ArrayData::Float(NdArray::from_vec(Shape::new(vec![k, dim]), result)),
                alive: true,
            })
        }
    }
}



// =============================================================================
// Module Registration
// =============================================================================

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpatialResult>()?;

    m.add_class::<PyProjectionReducer>()?;
    m.add_class::<PyBallTree>()?;
    m.add_class::<PyKDTree>()?;
    m.add_class::<PyVPTree>()?;
    //m.add_class::<PyMTree>()?;
    m.add_class::<PyAggTree>()?;
    m.add_class::<PyBruteForce>()?;
    m.add_class::<PyRPTree>()?;
    Ok(())
}
