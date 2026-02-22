use std::collections::BinaryHeap;
use crate::{array::{NdArray, Shape}, spatial::common::{DistanceMetric, HeapItem, KernelType}};
use rayon::prelude::*;

const KDE_PAR_THRESHOLD: usize = 512;

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


    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64;

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        (self.node_left(node_idx).unwrap(), self.node_right(node_idx).unwrap())
    }

    fn get_point(&self, i: usize) -> &[f64] {
        let dim = self.dim();
        &self.data()[i * dim..(i + 1) * dim]
    }

    fn n_points(&self) -> usize {
        self.data().len() / self.dim()
    }

    fn query_radius(&self, query: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_radius_recursive(0, query, radius, &mut results);
        results
    }

    fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<usize>) {
        if self.min_distance_to_node(node_idx, query) > radius {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                if self.metric().distance(query, self.get_point(i)) <= radius {
                    results.push(self.indices()[i]);
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

    fn query_knn(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.indices().is_empty() {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k);
        self.query_knn_recursive(0, query, &mut heap, k);
        heap.into_sorted_vec()
            .into_iter()
            .map(|item| (item.index, item.distance))
            .collect()
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize) {
        if heap.len() == k && self.min_distance_to_node(node_idx, query) > heap.peek().unwrap().distance {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = self.metric().distance(query, self.get_point(i));
                if heap.len() < k {
                    heap.push(HeapItem { distance: dist, index: self.indices()[i] });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(HeapItem { distance: dist, index: self.indices()[i] });
                }
            }
            return;
        }

        let (first, second) = self.knn_child_order(node_idx, query);
        self.query_knn_recursive(first, query, heap, k);
        self.query_knn_recursive(second, query, heap, k);
    }

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType) {
        if kernel.evaluate(self.min_distance_to_node(node_idx, query), h) < 1e-10 {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = self.metric().distance(query, self.get_point(i));
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
            self.kde_recursive(0, query, bandwidth, &mut density, kernel);
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
                self.kde_recursive(0, query, bandwidth, &mut density, kernel);
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