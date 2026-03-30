use num_traits::{ToPrimitive, NumCast};

use crate::array::{NdArray, Shape};
use crate::projection::ProjectionType;
use crate::spatial::trees::{
    BallTree, BruteForce, KDTree, RPTree, VPTree, VantagePointSelection,
    BallTree32, BruteForce32, KDTree32, RPTree32, VPTree32,
};
use crate::spatial::common::IronFloat;
use crate::spatial::{DistanceMetric, KernelType, SpatialTree};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery, AnnQuery};


#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TreeType {
    KDTree,
    BallTree,
    VPTree,
    RPTree,
    BruteForce,
}

// =============================================================================
// Inner Tree Enum
// =============================================================================

pub(crate) enum TreeInner {
    KDTreeF64(KDTree),
    KDTreeF32(KDTree32),
    BallTreeF64(BallTree),
    BallTreeF32(BallTree32),
    VPTreeF64(VPTree),
    VPTreeF32(VPTree32),
    RPTreeF64(RPTree),
    RPTreeF32(RPTree32),
    BruteForceF64(BruteForce),
    BruteForceF32(BruteForce32),
}

macro_rules! dispatch_typed {
    ($tree_inner:expr, f64 |$t64:ident| $body64:expr, f32 |$t32:ident| $body32:expr) => {
        match $tree_inner {
            TreeInner::KDTreeF64($t64)     => { $body64 }
            TreeInner::KDTreeF32($t32)     => { $body32 }
            TreeInner::BallTreeF64($t64)   => { $body64 }
            TreeInner::BallTreeF32($t32)   => { $body32 }
            TreeInner::VPTreeF64($t64)     => { $body64 }
            TreeInner::VPTreeF32($t32)     => { $body32 }
            TreeInner::RPTreeF64($t64)     => { $body64 }
            TreeInner::RPTreeF32($t32)     => { $body32 }
            TreeInner::BruteForceF64($t64) => { $body64 }
            TreeInner::BruteForceF32($t32) => { $body32 }
        }
    };
}

pub struct QueryResult {
    pub indices: Vec<i64>,
    pub distances: Vec<f64>,
    pub counts: Option<Vec<i64>>,
    pub n_queries: usize,
    pub k: Option<usize>,
}

pub enum QueryInput<'a> {
    F64(&'a NdArray<f64>),
    F32(&'a NdArray<f32>),
}

pub struct SpatialIndex {
    tree: Option<TreeInner>,
    buffer_f64: Vec<f64>,
    buffer_f32: Vec<f32>,

    // Config
    tree_type: TreeType,
    dim: usize,
    use_f32: bool,
    leaf_size: usize,
    metric: DistanceMetric,
    rebuild_threshold: usize,

    // RPTree-specific
    seed: u64,
    projection_type: ProjectionType,

    // VPTree-specific
    vp_selection: VantagePointSelection,
}

impl SpatialIndex {
    pub fn new_f64(
        data: NdArray<f64>,
        tree_type: TreeType,
        leaf_size: usize,
        metric: DistanceMetric,
        rebuild_threshold: usize,
        seed: u64,
        projection_type: ProjectionType,
        vp_selection: VantagePointSelection,
    ) -> Self {
        let dim = data.shape().dims()[1];
        let mut idx = SpatialIndex {
            tree: None,
            buffer_f64: Vec::new(),
            buffer_f32: Vec::new(),
            tree_type,
            dim,
            use_f32: false,
            leaf_size,
            metric,
            rebuild_threshold,
            seed,
            projection_type,
            vp_selection,
        };
        idx.tree = Some(idx.build_tree_f64(data));
        idx
    }

    pub fn new_f32(
        data: NdArray<f32>,
        tree_type: TreeType,
        leaf_size: usize,
        metric: DistanceMetric,
        rebuild_threshold: usize,
        seed: u64,
        projection_type: ProjectionType,
        vp_selection: VantagePointSelection,
    ) -> Self {
        let dim = data.shape().dims()[1];
        let mut idx = SpatialIndex {
            tree: None,
            buffer_f64: Vec::new(),
            buffer_f32: Vec::new(),
            tree_type,
            dim,
            use_f32: true,
            leaf_size,
            metric,
            rebuild_threshold,
            seed,
            projection_type,
            vp_selection,
        };
        idx.tree = Some(idx.build_tree_f32(data));
        idx
    }

    fn tree_ref(&self) -> Result<&TreeInner, String> {
        self.tree.as_ref().ok_or_else(|| "SpatialIndex is uninitialized".to_string())
    }

    pub fn buffer_count(&self) -> usize {
        if self.dim == 0 { return 0; }
        if self.use_f32 {
            self.buffer_f32.len() / self.dim
        } else {
            self.buffer_f64.len() / self.dim
        }
    }

    pub fn rebuild(&mut self) -> Result<(), String> {
        let tree_ref = self.tree.as_ref()
            .ok_or_else(|| "SpatialIndex is uninitialized".to_string())?;

        if self.use_f32 {
            let mut combined = extract_data_f32(tree_ref, self.dim);
            combined.extend_from_slice(&self.buffer_f32);
            self.buffer_f32.clear();
            let n = combined.len() / self.dim;
            let arr = NdArray::from_vec(Shape::new(vec![n, self.dim]), combined);
            self.tree = Some(self.build_tree_f32(arr));
        } else {
            let mut combined = extract_data_f64(tree_ref, self.dim);
            combined.extend_from_slice(&self.buffer_f64);
            self.buffer_f64.clear();
            let n = combined.len() / self.dim;
            let arr = NdArray::from_vec(Shape::new(vec![n, self.dim]), combined);
            self.tree = Some(self.build_tree_f64(arr));
        }
        Ok(())
    }

    fn build_tree_f64(&self, data: NdArray<f64>) -> TreeInner {
        match self.tree_type {
            TreeType::KDTree => {
                TreeInner::KDTreeF64(KDTree::new(data, self.leaf_size, self.metric))
            }
            TreeType::BallTree => {
                TreeInner::BallTreeF64(BallTree::new(data, self.leaf_size, self.metric))
            }
            TreeType::VPTree => {
                TreeInner::VPTreeF64(VPTree::new(data, self.leaf_size, self.metric, self.vp_selection))
            }
            TreeType::RPTree => {
                TreeInner::RPTreeF64(RPTree::new(data, self.leaf_size, self.metric, self.projection_type, self.seed))
            }
            TreeType::BruteForce => {
                TreeInner::BruteForceF64(BruteForce::new(data, self.metric))
            }
        }
    }

    fn build_tree_f32(&self, data: NdArray<f32>) -> TreeInner {
        match self.tree_type {
            TreeType::KDTree => {
                TreeInner::KDTreeF32(KDTree32::new(data, self.leaf_size, self.metric))
            }
            TreeType::BallTree => {
                TreeInner::BallTreeF32(BallTree32::new(data, self.leaf_size, self.metric))
            }
            TreeType::VPTree => {
                TreeInner::VPTreeF32(VPTree32::new(data, self.leaf_size, self.metric, self.vp_selection))
            }
            TreeType::RPTree => {
                TreeInner::RPTreeF32(RPTree32::new(data, self.leaf_size, self.metric, self.projection_type, self.seed))
            }
            TreeType::BruteForce => {
                TreeInner::BruteForceF32(BruteForce32::new(data, self.metric))
            }
        }
    }


    pub fn insert_f64(&mut self, flat_data: &[f64], point_dim: usize) -> Result<(), String> {
        if point_dim != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, point_dim));
        }
        self.buffer_f64.extend_from_slice(flat_data);
        if self.buffer_count() >= self.rebuild_threshold {
            self.rebuild()?;
        }
        Ok(())
    }

    pub fn insert_f32(&mut self, flat_data: &[f32], point_dim: usize) -> Result<(), String> {
        if point_dim != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, point_dim));
        }
        self.buffer_f32.extend_from_slice(flat_data);
        if self.buffer_count() >= self.rebuild_threshold {
            self.rebuild()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), String> {
        if self.buffer_count() > 0 {
            self.rebuild()
        } else {
            Ok(())
        }
    }

    pub fn tree_type(&self) -> TreeType {
        self.tree_type
    }

    pub fn use_f32(&self) -> bool {
        self.use_f32
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_points(&self) -> Result<usize, String> {
        let tree_ref = self.tree_ref()?;
        let tree_points = dispatch_typed!(tree_ref,
            f64 |t| t.n_points(),
            f32 |t| t.n_points()
        );
        Ok(tree_points + self.buffer_count())
    }

    pub fn pending_count(&self) -> usize {
        self.buffer_count()
    }

    pub fn metric(&self) -> &DistanceMetric {
        &self.metric
    }

    pub fn rebuild_threshold(&self) -> usize {
        self.rebuild_threshold
    }

    pub fn set_rebuild_threshold(&mut self, value: usize) {
        self.rebuild_threshold = value;
    }

    // =========================================================================
    // Queries
    // =========================================================================

    pub fn query_knn(&self, query: QueryInput<'_>, is_batch: bool, k: usize) -> Result<QueryResult, String> {
        let tree_ref = self.tree_ref()?;
        match query {
            QueryInput::F64(q) => {
                dispatch_typed!(tree_ref,
                    f64 |t| knn_impl(t, q, is_batch, k, &self.buffer_f64, self.dim, &self.metric),
                    f32 |_t| Err("f64 query provided for f32 tree".to_string())
                )
            }
            QueryInput::F32(q) => {
                dispatch_typed!(tree_ref,
                    f64 |_t| Err("f32 query provided for f64 tree".to_string()),
                    f32 |t| knn_impl(t, q, is_batch, k, &self.buffer_f32, self.dim, &self.metric)
                )
            }
        }
    }

    pub fn query_ann(
        &self,
        query: QueryInput<'_>,
        is_batch: bool,
        k: usize,
        n_candidates: usize,
        n_probes: Option<usize>,
    ) -> Result<QueryResult, String> {
        let tree_ref = self.tree_ref()?;
        match query {
            QueryInput::F64(q) => {
                dispatch_typed!(tree_ref,
                    f64 |t| ann_impl(t, q, is_batch, k, n_candidates, n_probes, &self.buffer_f64, self.dim, &self.metric),
                    f32 |_t| Err("f64 query provided for f32 tree".to_string())
                )
            }
            QueryInput::F32(q) => {
                dispatch_typed!(tree_ref,
                    f64 |_t| Err("f32 query provided for f64 tree".to_string()),
                    f32 |t| ann_impl(t, q, is_batch, k, n_candidates, n_probes, &self.buffer_f32, self.dim, &self.metric)
                )
            }
        }
    }

    pub fn query_radius(&self, query: QueryInput<'_>, is_batch: bool, radius: f64) -> Result<QueryResult, String> {
        let tree_ref = self.tree_ref()?;
        match query {
            QueryInput::F64(q) => {
                dispatch_typed!(tree_ref,
                    f64 |t| radius_impl(t, q, is_batch, radius, &self.buffer_f64, self.dim, &self.metric),
                    f32 |_t| Err("f64 query provided for f32 tree".to_string())
                )
            }
            QueryInput::F32(q) => {
                dispatch_typed!(tree_ref,
                    f64 |_t| Err("f32 query provided for f64 tree".to_string()),
                    f32 |t| {
                        let rad: f32 = <f32 as NumCast>::from(radius).unwrap();
                        radius_impl(t, q, is_batch, rad, &self.buffer_f32, self.dim, &self.metric)
                    }
                )
            }
        }
    }

    pub fn kernel_density(
        &self,
        queries: Option<QueryInput<'_>>,
        bandwidth: f64,
        kernel: KernelType,
        normalize: bool,
    ) -> Result<NdArray<f64>, String> {
        let tree_ref = self.tree_ref()?;
        match queries {
            Some(QueryInput::F64(q)) => {
                dispatch_typed!(tree_ref,
                    f64 |t| {
                        let mut result = t.kernel_density(q, bandwidth, kernel, normalize);
                        if !self.buffer_f64.is_empty() {
                            add_buffer_kde(&mut result, q, &self.buffer_f64, self.dim, &self.metric, bandwidth, kernel, normalize);
                        }
                        Ok(result)
                    },
                    f32 |_t| Err("f64 query provided for f32 tree".to_string())
                )
            }
            Some(QueryInput::F32(q)) => {
                dispatch_typed!(tree_ref,
                    f64 |_t| Err("f32 query provided for f64 tree".to_string()),
                    f32 |t| {
                        let mut result = t.kernel_density(q, bandwidth, kernel, normalize);
                        if !self.buffer_f32.is_empty() {
                            add_buffer_kde(&mut result, q, &self.buffer_f32, self.dim, &self.metric, bandwidth, kernel, normalize);
                        }
                        Ok(result)
                    }
                )
            }
            None => {
                dispatch_typed!(tree_ref,
                    f64 |t| {
                        let queries_arr = NdArray::from_vec(
                            Shape::new(vec![t.n_points(), t.dim]),
                            t.data().to_vec()
                        );
                        let mut result = t.kernel_density(&queries_arr, bandwidth, kernel, normalize);
                        if !self.buffer_f64.is_empty() {
                            add_buffer_kde(&mut result, &queries_arr, &self.buffer_f64, self.dim, &self.metric, bandwidth, kernel, normalize);
                        }
                        Ok(result)
                    },
                    f32 |t| {
                        let queries_arr = NdArray::from_vec(
                            Shape::new(vec![t.n_points(), t.dim]),
                            t.data().to_vec()
                        );
                        let mut result = t.kernel_density(&queries_arr, bandwidth, kernel, normalize);
                        if !self.buffer_f32.is_empty() {
                            add_buffer_kde(&mut result, &queries_arr, &self.buffer_f32, self.dim, &self.metric, bandwidth, kernel, normalize);
                        }
                        Ok(result)
                    }
                )
            }
        }
    }

    pub fn data(&self, indices: Option<&[i64]>) -> Result<(Vec<f64>, usize, usize), String> {
        let tree_ref = self.tree_ref()?;

        let tree_n = dispatch_typed!(tree_ref, f64 |t| t.n_points(), f32 |t| t.n_points());
        let buf_n = self.buffer_count();
        let total_n = tree_n + buf_n;

        let tree_data = extract_data_f64(tree_ref, self.dim);

        if buf_n == 0 {
            match indices {
                None => Ok((tree_data, tree_n, self.dim)),
                Some(idx) => {
                    let mut result = Vec::with_capacity(idx.len() * self.dim);
                    for &orig_idx in idx {
                        let i = orig_idx as usize;
                        if i >= tree_n {
                            return Err(format!(
                                "Index {} out of bounds for index with {} points", orig_idx, tree_n
                            ));
                        }
                        result.extend_from_slice(&tree_data[i * self.dim..(i + 1) * self.dim]);
                    }
                    Ok((result, idx.len(), self.dim))
                }
            }
        } else {
            //need to build a new vec with the buffered data included
            let mut full = tree_data;
            if self.use_f32 {
                full.extend(self.buffer_f32.iter().map(|&v| v as f64));
            } else {
                full.extend_from_slice(&self.buffer_f64);
            }

            match indices {
                None => Ok((full, total_n, self.dim)),
                Some(idx) => {
                    let mut result = Vec::with_capacity(idx.len() * self.dim);
                    for &orig_idx in idx {
                        let i = orig_idx as usize;
                        if i >= total_n {
                            return Err(format!(
                                "Index {} out of bounds for index with {} points", orig_idx, total_n
                            ));
                        }
                        result.extend_from_slice(&full[i * self.dim..(i + 1) * self.dim]);
                    }
                    Ok((result, idx.len(), self.dim))
                }
            }
        }
    }
}

fn extract_data_f64(tree: &TreeInner, dim: usize) -> Vec<f64> {
    dispatch_typed!(tree,
        f64 |t| {
            let mut data = vec![0.0f64; t.n_points() * dim];
            for (tree_pos, &orig_idx) in t.indices().iter().enumerate() {
                for d in 0..dim {
                    data[orig_idx * dim + d] = t.data()[tree_pos * dim + d].to_f64().unwrap();
                }
            }
            data
        },
        f32 |t| {
            let mut data = vec![0.0f64; t.n_points() * dim];
            for (tree_pos, &orig_idx) in t.indices().iter().enumerate() {
                for d in 0..dim {
                    data[orig_idx * dim + d] = t.data()[tree_pos * dim + d].to_f64().unwrap();
                }
            }
            data
        }
    )
}

fn extract_data_f32(tree: &TreeInner, dim: usize) -> Vec<f32> {
    dispatch_typed!(tree,
        f64 |t| {
            let mut data = vec![0.0f32; t.n_points() * dim];
            for (tree_pos, &orig_idx) in t.indices().iter().enumerate() {
                for d in 0..dim {
                    data[orig_idx * dim + d] = NumCast::from(t.data()[tree_pos * dim + d]).unwrap();
                }
            }
            data
        },
        f32 |t| {
            let mut data = vec![0.0f32; t.n_points() * dim];
            for (tree_pos, &orig_idx) in t.indices().iter().enumerate() {
                for d in 0..dim {
                    data[orig_idx * dim + d] = NumCast::from(t.data()[tree_pos * dim + d]).unwrap();
                }
            }
            data
        }
    )
}

// =============================================================================
// Buffer-aware query helpers
// =============================================================================

#[inline]
fn buffer_distance<T: IronFloat>(metric: &DistanceMetric, query: &[T], point: &[T]) -> T {
    let q = metric.pre_transform(query);
    let p = metric.pre_transform(point);
    let rd = metric.reduced_distance(&q, &p);
    metric.post_transform(rd)
}

fn merge_buffer_topk<T: IronFloat>(
    results: &mut Vec<(usize, T)>,
    buffer: &[T],
    dim: usize,
    metric: &DistanceMetric,
    query: &[T],
    k: usize,
    offset: usize,
) {
    if buffer.is_empty() { return; }
    let n_buf = buffer.len() / dim;
    for i in 0..n_buf {
        let point = &buffer[i * dim..(i + 1) * dim];
        let dist = buffer_distance(metric, query, point);
        if results.len() < k {
            results.push((offset + i, dist));
        } else if dist < results.last().unwrap().1 {
            results.pop();
            results.push((offset + i, dist));
        } else {
            continue;
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    }
}

fn merge_buffer_radius<T: IronFloat>(
    results: &mut Vec<(usize, T)>,
    buffer: &[T],
    dim: usize,
    metric: &DistanceMetric,
    query: &[T],
    radius: T,
    offset: usize,
) {
    if buffer.is_empty() { return; }
    let n_buf = buffer.len() / dim;
    for i in 0..n_buf {
        let point = &buffer[i * dim..(i + 1) * dim];
        let dist = buffer_distance(metric, query, point);
        if dist <= radius {
            results.push((offset + i, dist));
        }
    }
}

fn add_buffer_kde<T: IronFloat>(
    densities: &mut NdArray<f64>,
    queries: &NdArray<T>,
    buffer: &[T],
    dim: usize,
    metric: &DistanceMetric,
    bandwidth: f64,
    kernel: KernelType,
    normalized: bool,
) {
    let n_queries = densities.shape().dims()[0];
    let n_buf = buffer.len() / dim;
    if n_buf == 0 { return; }

    let h = T::from(bandwidth).unwrap();
    let queries_slice = queries.as_contiguous_slice();
    let density_slice = densities.as_mut_slice().expect("KDE result should be owned");

    for qi in 0..n_queries {
        let q = &queries_slice[qi * dim..(qi + 1) * dim];
        let mut contrib = 0.0f64;
        for bi in 0..n_buf {
            let p = &buffer[bi * dim..(bi + 1) * dim];
            let dist = buffer_distance(metric, q, p);
            contrib += kernel.evaluate(dist, h).to_f64().unwrap();
        }
        if normalized {
            let h_d = bandwidth.powi(dim as i32);
            let c_k = kernel.normalization_constant(dim);
            contrib /= h_d * c_k;
        }
        density_slice[qi] += contrib;
    }
}

// =============================================================================
// Query implementations
// =============================================================================

fn knn_impl<T, F>(
    tree: &T,
    queries: &NdArray<F>,
    is_batch: bool,
    k: usize,
    buffer: &[F],
    dim: usize,
    metric: &DistanceMetric,
) -> Result<QueryResult, String>
where
    T: SpatialTree<Float = F> + KnnQuery,
    F: IronFloat,
{
    let offset = tree.n_points();
    let n_queries = queries.shape().dims()[0];
    if is_batch {
        let mut results = tree.query_knn_batch(queries, k);
        if !buffer.is_empty() {
            let qs = queries.as_contiguous_slice();
            for (qi, res) in results.iter_mut().enumerate() {
                merge_buffer_topk(res, buffer, dim, metric, &qs[qi * dim..(qi + 1) * dim], k, offset);
            }
        }
        let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
            .flatten()
            .map(|(i, d)| (i as i64, d.to_f64().unwrap()))
            .unzip();
        if n_queries == 1 {
            Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
        } else {
            Ok(QueryResult { indices, distances, counts: None, n_queries, k: Some(k) })
        }
    } else {
        let query_slice = &queries.as_slice_unchecked()[..dim];
        let mut results = tree.query_knn(query_slice, k);
        merge_buffer_topk(&mut results, buffer, dim, metric, query_slice, k, offset);
        let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
            .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
        Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
    }
}

fn ann_impl<T, F>(
    tree: &T,
    queries: &NdArray<F>,
    is_batch: bool,
    k: usize,
    n_candidates: usize,
    n_probes: Option<usize>,
    buffer: &[F],
    dim: usize,
    metric: &DistanceMetric,
) -> Result<QueryResult, String>
where
    T: SpatialTree<Float = F> + AnnQuery,
    F: IronFloat,
{
    let offset = tree.n_points();
    let n_queries = queries.shape().dims()[0];
    if is_batch {
        let mut results = match n_probes {
            Some(np) => tree.query_ann_stochastic_batch(queries, k, n_candidates, np),
            None => tree.query_ann_batch(queries, k, n_candidates),
        };
        if !buffer.is_empty() {
            let qs = queries.as_contiguous_slice();
            for (qi, res) in results.iter_mut().enumerate() {
                merge_buffer_topk(res, buffer, dim, metric, &qs[qi * dim..(qi + 1) * dim], k, offset);
            }
        }
        let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
            .flatten()
            .map(|(i, d)| (i as i64, d.to_f64().unwrap()))
            .unzip();
        if n_queries == 1 {
            Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
        } else {
            Ok(QueryResult { indices, distances, counts: None, n_queries, k: Some(k) })
        }
    } else {
        let query_slice = &queries.as_slice_unchecked()[..dim];
        let mut results = match n_probes {
            Some(np) => tree.query_ann_stochastic(query_slice, k, n_candidates, np),
            None => tree.query_ann(query_slice, k, n_candidates),
        };
        merge_buffer_topk(&mut results, buffer, dim, metric, query_slice, k, offset);
        let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
            .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
        Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
    }
}

fn radius_impl<T, F>(
    tree: &T,
    queries: &NdArray<F>,
    is_batch: bool,
    radius: F,
    buffer: &[F],
    dim: usize,
    metric: &DistanceMetric,
) -> Result<QueryResult, String>
where
    T: SpatialTree<Float = F> + RadiusQuery,
    F: IronFloat,
{
    let offset = tree.n_points();
    if is_batch {
        let n_queries = queries.shape().dims()[0];
        let mut results = tree.query_radius_batch(queries, radius);
        if !buffer.is_empty() {
            let qs = queries.as_contiguous_slice();
            for (qi, res) in results.iter_mut().enumerate() {
                merge_buffer_radius(res, buffer, dim, metric, &qs[qi * dim..(qi + 1) * dim], radius, offset);
            }
        }
        if n_queries == 1 {
            let batch = results.into_iter().next().unwrap_or_default();
            let (indices, distances): (Vec<i64>, Vec<f64>) = batch.into_iter()
                .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
            Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
        } else {
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
            Ok(QueryResult { indices: all_indices, distances: all_distances, counts: Some(counts), n_queries, k: None })
        }
    } else {
        let query_slice = &queries.as_slice_unchecked()[..dim];
        let mut results = tree.query_radius(query_slice, radius);
        merge_buffer_radius(&mut results, buffer, dim, metric, query_slice, radius, offset);
        let (indices, distances): (Vec<i64>, Vec<f64>) = results.into_iter()
            .map(|(i, d)| (i as i64, d.to_f64().unwrap())).unzip();
        Ok(QueryResult { indices, distances, counts: None, n_queries: 1, k: None })
    }
}
