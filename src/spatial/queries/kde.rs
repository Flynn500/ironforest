use crate::{array::{NdArray, Shape}, spatial::common::KernelType};
use rayon::prelude::*;
use crate::spatial::SpatialTree;

const KDE_PAR_THRESHOLD: usize = 512;

pub trait KdeQuery: SpatialTree {
    fn kernel_density(&self, queries: &NdArray<f64>, bandwidth: f64, kernel: KernelType, normalize: bool) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[f64] = &queries_cow;
        let mut results = if n_queries >= KDE_PAR_THRESHOLD {
            self.par_kde_recursion(kernel, bandwidth, queries_slice, n_queries, dim)
        } else {
            self.seq_kde_recursion(kernel, bandwidth, queries_slice, n_queries, dim)
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

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType) {
        let n = (self.node_end(node_idx) - self.node_start(node_idx)) as f64;
        if kernel.evaluate(self.min_distance_to_node(node_idx, query), h) * n < 1e-10 {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = match Self::REDUCED {
                    true => self.metric().reduced_distance(query, self.get_point(i)),
                    false => self.metric().distance(query, self.get_point(i)),
                };
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

    fn seq_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[f64], n_queries: usize, dim: usize) -> Vec<f64> {
        let mut results = vec![0.0; n_queries];
        for i in 0..n_queries {
            let query = &queries[i * dim..(i + 1) * dim];
            self.kde_recursive(self.root(), query, bandwidth, &mut results[i], kernel);
        }
        results
    }

    fn par_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[f64], n_queries: usize, dim: usize) -> Vec<f64> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                let mut density = 0.0;
                self.kde_recursive(self.root(), query, bandwidth, &mut density, kernel);
                density
            })
            .collect()
    }
}