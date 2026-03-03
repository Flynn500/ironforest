use std::collections::BinaryHeap;
use crate::{array::{NdArray, Shape}, spatial::common::{DistanceMetric, HeapItem, KernelType}};
use rayon::prelude::*;

//Maybe this should be a single default? will have to run comparisons to see 
//if query type changes optimal threshold
const KDE_PAR_THRESHOLD: usize = 512;
const KNN_PAR_THRESHOLD: usize = 512;
const RAD_PAR_THRESHOLD: usize = 512;

pub trait SpatialQuery: Sync {
    type Node;

    fn nodes(&self) -> &[Self::Node];
    fn indices(&self) -> &[usize];
    fn data(&self) -> &[f64];
    fn dim(&self) -> usize;
    fn metric(&self) -> &DistanceMetric;

    fn node_start(&self, idx: usize) -> usize;
    fn node_end(&self, idx: usize) -> usize;
    fn node_left(&self, idx: usize) -> Option<usize>;
    fn node_right(&self, idx: usize) -> Option<usize>;

    fn root(&self) -> usize { 0 }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64;

    fn knn_child_order(&self, node_idx: usize, _query: &[f64]) -> (usize, usize) {
        (self.node_left(node_idx).unwrap(), self.node_right(node_idx).unwrap())
    }

    //combination of min dist to node and knn order - we do this to avoid calling dot 
    //2x for RPTrees, other trees need only implement child order and min dist to node
    fn node_projection(&self, node_idx: usize, query: &[f64]) -> (usize, usize, f64) {
        let dist = self.min_distance_to_node(node_idx, query);
        let (first, second) = self.knn_child_order(node_idx, query);
        (first, second, dist)
    }

    fn get_point(&self, i: usize) -> &[f64] {
        let dim = self.dim();
        &self.data()[i * dim..(i + 1) * dim]
    }

    fn n_points(&self) -> usize;

    fn query_knn(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.n_points() == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k);
        self.query_knn_recursive(self.root(), query, &mut heap, k);
        heap.into_sorted_vec()
            .into_iter()
            .map(|item| (item.index, self.metric().post_transform(item.distance)))
            .collect()
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize) {
        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = self.metric().reduced_distance(query, self.get_point(i));
                if heap.len() < k {
                    heap.push(HeapItem { distance: dist, index: self.indices()[i] });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(HeapItem { distance: dist, index: self.indices()[i] });
                }
            }
            return;
        }

        let (first, second, node_dist) = self.node_projection(node_idx, query);

        if heap.len() == k && node_dist > heap.peek().unwrap().distance {
            return;
        }

        self.query_knn_recursive(first, query, heap, k);
        self.query_knn_recursive(second, query, heap, k);
    }

    fn seq_knn_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, k: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries.as_slice()[i * dim..(i + 1) * dim];
                self.query_knn(query, k)
            })
            .collect()
    }

    fn par_knn_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, k: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries.as_slice()[i * dim..(i + 1) * dim];
                self.query_knn(query, k)
            })
            .collect()
    }

    fn query_knn_batch(&self, queries: &NdArray<f64>, k: usize) -> Vec<Vec<(usize, f64)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        if n_queries >= KNN_PAR_THRESHOLD {
            self.par_knn_batch(queries, n_queries, dim, k)
        } else {
            self.seq_knn_batch(queries, n_queries, dim, k)
        }
    }

    fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<(usize, f64)>) {
        if self.min_distance_to_node(node_idx, query) > radius {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = self.metric().reduced_distance(query, self.get_point(i));
                if dist <= radius {
                    results.push((self.indices()[i], dist));
                }
            }
            return;
        }

        if let Some(left) = self.node_left(node_idx) {
            self.query_radius_recursive(left, query, radius, results);
        }
        if let Some(right) = self.node_right(node_idx) {
            self.query_radius_recursive(right, query, radius, results);
        }
    }

    fn query_radius(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        self.query_radius_recursive(self.root(), query, self.metric().pre_transform_radius(radius), &mut results);
        results
    }

    fn seq_radius_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, radius: f64) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries.as_slice()[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }

    fn par_radius_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, radius: f64) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries.as_slice()[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }

    fn query_radius_batch(&self, queries: &NdArray<f64>, radius: f64) -> Vec<Vec<(usize, f64)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        if n_queries >= RAD_PAR_THRESHOLD {
            self.par_radius_batch(queries, n_queries, dim, self.metric().pre_transform_radius(radius))
        } else {
            self.seq_radius_batch(queries, n_queries, dim, self.metric().pre_transform_radius(radius))
        }
    }

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType) {
        let n = (self.node_end(node_idx) - self.node_start(node_idx)) as f64;
        if kernel.evaluate(self.min_distance_to_node(node_idx, query), h) * n < 1e-10 {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = self.metric().reduced_distance(query, self.get_point(i));
                *density += kernel.evaluate(dist, h);
            }
            return;
        }

        if let Some(left) = self.node_left(node_idx) {
            self.kde_recursive(left, query, h, density, kernel);
        }
        if let Some(right) = self.node_right(node_idx) {
            self.kde_recursive(right, query, h, density, kernel);
        }
    }

    fn seq_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &NdArray<f64>, n_queries: usize, dim: usize) -> Vec<f64>{
        let mut results = vec![0.0; n_queries];

        for i in 0..n_queries {
            let query = &queries.as_slice()[i * dim..(i + 1) * dim];
            let mut density = 0.0;
            self.kde_recursive(self.root(), query, bandwidth, &mut density, kernel);
            results[i] = density;
        }
        results
    }

    fn par_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &NdArray<f64>, n_queries: usize, dim: usize) -> Vec<f64>{
        let results: Vec<f64> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries.as_slice()[i * dim..(i + 1) * dim];
                let mut density = 0.0;
                self.kde_recursive(self.root(), query, bandwidth, &mut density, kernel);
                density
            })
            .collect();
        results
    }

    fn kernel_density(&self, queries: &NdArray<f64>, bandwidth: f64, kernel: KernelType, normalize: bool) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");

        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let mut results = if n_queries >= KDE_PAR_THRESHOLD {
            self.par_kde_recursion(kernel, bandwidth, queries, n_queries, dim)
        } else {
            self.seq_kde_recursion(kernel, bandwidth, queries, n_queries, dim)
        };

        if normalize {
            let h_d = bandwidth.powi(dim as i32);
            let c_k = kernel.normalization_constant(dim);
            let norm = h_d * c_k;
            for val in &mut results {
                *val /= norm;
            }
        }
        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }
}