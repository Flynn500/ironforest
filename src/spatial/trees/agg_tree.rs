use crate::{KernelType, Shape, array::NdArray, spatial::common::{DistanceMetric, IronFloat}};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const KDE_PAR_THRESHOLD: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct AggNode<T> {
    pub center: Vec<T>,
    pub radius: T,
    // Higher-order moments stored in f64 for kernel computation precision
    pub variance: f64,
    pub moment3: f64,
    pub moment4: f64,
    pub max_abs_error: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct AggTree<T: IronFloat> {
    pub nodes: Vec<AggNode<T>>,
    pub indices: Vec<usize>,
    pub data: NdArray<T>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub kernel: KernelType,
    pub bandwidth: f64,
    pub atol: f64,
}

impl<T: IronFloat> AggTree<T> {
    pub fn new(
        mut data: NdArray<T>, leaf_size: usize, metric: DistanceMetric,
        kernel: KernelType, bandwidth: f64, atol: f64,
    ) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        if !data.is_owned() {
            data = data.to_contiguous();
        }
        if matches!(metric, DistanceMetric::Cosine) {
            for i in 0..n_points {
                let normed = metric.pre_transform(data.row(i)).into_owned();
                data.set_row(i, &normed);
            }
        }

        let mut tree = AggTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size,
            metric,
            kernel,
            bandwidth,
            atol,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();

        let mut live = Vec::new();
        tree.collect_live_ranges(0, &mut live);
        let remap = tree.compact_data(&live);
        tree.remap_nodes(&remap);

        tree
    }

    fn reorder_data(&mut self) {
        let mut new_data = vec![T::zero(); self.data.len()];

        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(self.data.row(old_idx));
        }

        self.data = NdArray::from_vec(Shape::new(vec![self.n_points, self.dim]), new_data);
    }

    fn collect_live_ranges(&self, node_idx: usize, live: &mut Vec<(usize, usize)>) {
        let node = &self.nodes[node_idx];

        if node.max_abs_error < self.atol {
            return;
        }

        match (node.left, node.right) {
            (None, _) => live.push((node.start, node.end)),
            (Some(left), Some(right)) => {
                self.collect_live_ranges(left, live);
                self.collect_live_ranges(right, live);
            }
            _ => unreachable!()
        }
    }

    fn compact_data(&mut self, live: &[(usize, usize)]) -> Vec<usize> {
        let mut new_data: Vec<T> = Vec::new();
        let mut remap = vec![usize::MAX; self.n_points];
        let mut new_idx = 0;

        for &(start, end) in live {
            for i in start..end {
                new_data.extend_from_slice(self.data.row(i));
                remap[i] = new_idx;
                new_idx += 1;
            }
        }

        self.data = NdArray::from_vec(Shape::new(vec![new_idx, self.dim]), new_data);
        remap
    }

    fn remap_nodes(&mut self, remap: &[usize]) {
        let atol = self.atol;

        for node in &mut self.nodes {
            let is_live_leaf = node.left.is_none() && !(node.max_abs_error < atol);

            if is_live_leaf {
                node.start = remap[node.start];
                node.end = remap[node.end - 1] + 1;
            }
        }
    }

    fn init_node(&self, start: usize, end: usize) -> (Vec<T>, T, f64, f64, f64) {
        let n: T = T::from(end - start).unwrap();
        let n_f64 = (end - start) as f64;
        let mut centroid = vec![T::zero(); self.dim];

        for i in start..end {
            let p = self.data.row(i);
            for (j, &x) in p.iter().enumerate() {
                centroid[j] = centroid[j] + x;
            }
        }

        for c in &mut centroid {
            *c = *c / n;
        }

        let mut max_dist = T::zero();
        let mut variance = 0.0f64;
        let mut moment3 = 0.0f64;
        let mut moment4 = 0.0f64;

        for i in start..end {
            let p = self.data.row(i);
            let dist: T = self.metric.post_transform(self.metric.reduced_distance(p, &centroid));
            if dist > max_dist { max_dist = dist; }
            let dist_f64 = dist.to_f64().unwrap();
            let d2 = dist_f64 * dist_f64;
            variance += d2;
            moment3 += d2 * dist_f64;
            moment4 += d2 * d2;
        }

        variance /= n_f64;
        moment3 /= n_f64;
        moment4 /= n_f64;

        (centroid, max_dist, variance, moment3, moment4)
    }


    fn furthest_from(&self, query: &[T], start: usize, end: usize) -> usize {
        (start..end)
            .max_by(|&a, &b| {
                let da = self.metric.post_transform(self.metric.reduced_distance(query, self.data.row(a)));
                let db = self.metric.post_transform(self.metric.reduced_distance(query, self.data.row(b)));
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }

    fn pivot_partition(&mut self, start: usize, end: usize, centroid: &[T]) -> usize {
        let p1_slot = self.furthest_from(centroid, start, end);
        let p1 = self.data.row(p1_slot).to_vec();

        let p2_slot = self.furthest_from(&p1, start, end);
        let p2 = self.data.row(p2_slot).to_vec();

        let axis: Vec<T> = p2.iter().zip(&p1).map(|(a, b)| *a - *b).collect();

        let mut projections: Vec<(T, usize)> = (start..end)
            .map(|i| {
                let p = self.data.row(i);
                let proj: T = p.iter().zip(&axis).map(|(x, a)| *x * *a).sum();
                (proj, self.indices[i])
            })
            .collect();

        let mid_offset = (end - start) / 2;
        projections.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        self.indices[start..end].copy_from_slice(
            &projections.iter().map(|&(_, idx)| idx).collect::<Vec<_>>()
        );

        start + mid_offset
    }


    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (center, radius, variance, moment3, moment4) = self.init_node(start, end);
        let n = (end - start) as f64;
        let radius_f64 = radius.to_f64().unwrap();

        let max_abs_error = self.kernel.node_error_bound(n, radius_f64, self.bandwidth);
        let mut mid = self.pivot_partition(start, end, &center);

        let node_idx = self.nodes.len();
        self.nodes.push(AggNode {
            center,
            radius,
            variance,
            moment3,
            moment4,
            max_abs_error,
            start,
            end,
            left: None,
            right: None,
        });

        let count = end - start;
        if count <= self.leaf_size || max_abs_error < self.atol {
            return node_idx;
        }

        if mid == start { mid = start + 1; }
        else if mid == end { mid = end - 1; }

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn min_distance_to_node_inner(&self, node_idx: usize, query: &[T]) -> f64 {
        let node = &self.nodes[node_idx];
        let d = self.metric.post_transform(self.metric.reduced_distance(query, &node.center));
        (d - node.radius).max(T::zero()).to_f64().unwrap()
    }

    fn approx_kde_for_node(&self, query: &[T], node: &AggNode<T>, h: f64, kernel: KernelType) -> f64 {
        let n = (node.end - node.start) as f64;
        let r_c: f64 = self.metric.post_transform(self.metric.reduced_distance(query, &node.center)).to_f64().unwrap();

        let k0 = kernel.evaluate(r_c, h);
        let k2 = kernel.evaluate_second_derivative(r_c, h);
        let k3 = kernel.third_derivative(r_c, h);
        let k4 = kernel.fourth_derivative(r_c, h);

        n * (k0 + 0.5 * k2 * node.variance + (1.0 / 6.0) * k3 * node.moment3 + (1.0 / 24.0) * k4 * node.moment4)
    }

    fn kde_recursive(&self, node_idx: usize, query: &[T], h: f64, density: &mut f64, kernel: KernelType) {
        let node = &self.nodes[node_idx];

        if node.left.is_none() {
            if node.max_abs_error < self.atol {
                *density += self.approx_kde_for_node(query, node, h, kernel);
            } else {
                for i in node.start..node.end {
                    let dist: f64 = self.metric.post_transform(self.metric.reduced_distance(query, self.data.row(i))).to_f64().unwrap();
                    *density += kernel.evaluate(dist, h);
                }
            }
            return;
        }
        let n = (self.nodes[node_idx].end - self.nodes[node_idx].start) as f64;
        let transformed_dist = self.min_distance_to_node_inner(node_idx, query);
        if kernel.evaluate(transformed_dist, h) * n < 1e-10 {
            return;
        }

        if let Some(left) = node.left {
            self.kde_recursive(left, query, h, density, kernel);
        }
        if let Some(right) = node.right {
            self.kde_recursive(right, query, h, density, kernel);
        }
    }

    fn seq_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[T], n_queries: usize, dim: usize) -> Vec<f64> {
        let mut results = vec![0.0; n_queries];

        for i in 0..n_queries {
            let query = &queries[i * dim..(i + 1) * dim];
            let mut density = 0.0;
            self.kde_recursive(0, query, bandwidth, &mut density, kernel);
            results[i] = density;
        }
        results
    }

    fn par_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[T], n_queries: usize, dim: usize) -> Vec<f64> {
        let results: Vec<f64> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                let mut density = 0.0;
                self.kde_recursive(0, query, bandwidth, &mut density, kernel);
                density
            })
            .collect();
        results
    }

    pub fn kernel_density(&self, queries: &NdArray<T>, normalize: bool) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");

        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim, "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[T] = &queries_cow;
        let mut results = if n_queries >= KDE_PAR_THRESHOLD {
            self.par_kde_recursion(self.kernel, self.bandwidth, queries_slice, n_queries, dim)
        } else {
            self.seq_kde_recursion(self.kernel, self.bandwidth, queries_slice, n_queries, dim)
        };

        if normalize {
            let h_d = self.bandwidth.powi(dim as i32);
            let c_k = self.kernel.normalization_constant(dim);
            let norm = h_d * c_k;
            for val in &mut results {
                *val /= norm;
            }
        }

        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }
}
