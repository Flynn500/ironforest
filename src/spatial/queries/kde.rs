use crate::{array::{NdArray, Shape}, spatial::common::KernelType};
use rayon::prelude::*;
use crate::spatial::SpatialTree;
use num_traits::ToPrimitive;

const KDE_PAR_THRESHOLD: usize = 512;

pub trait KdeQuery: SpatialTree {
    fn kernel_density(&self, queries: &NdArray<Self::Float>, bandwidth: f64, kernel: KernelType, normalize: bool) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[Self::Float] = &queries_cow;
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

    fn kde_recursive(&self, node_idx: usize, query: &[Self::Float], h: f64, density: &mut f64, kernel: KernelType) {
        if self.is_leaf(node_idx) {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist: f64 = match Self::REDUCED {
                    true => self.metric().reduced_distance(query, self.get_point(i)).to_f64().unwrap(),
                    false => self.metric().distance(query, self.get_point(i)).to_f64().unwrap(),
                };
                *density += kernel.evaluate(dist, h);
            }
            return;
        }

        let plan = self.plan_traversal(node_idx, query);

        let first_n = (self.node_end(plan.first.child_idx) - self.node_start(plan.first.child_idx)) as f64;
        if kernel.evaluate(plan.first.lower_bound.to_f64().unwrap(), h) * first_n >= 1e-10 {
            self.kde_recursive(plan.first.child_idx, query, h, density, kernel);
        }

        let second_n = (self.node_end(plan.second.child_idx) - self.node_start(plan.second.child_idx)) as f64;
        if kernel.evaluate(plan.second.lower_bound.to_f64().unwrap(), h) * second_n >= 1e-10 {
            self.kde_recursive(plan.second.child_idx, query, h, density, kernel);
        }
    }

    fn seq_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[Self::Float], n_queries: usize, dim: usize) -> Vec<f64> {
        let mut results = vec![0.0; n_queries];
        for i in 0..n_queries {
            let query = &queries[i * dim..(i + 1) * dim];
            self.kde_recursive(self.root(), query, bandwidth, &mut results[i], kernel);
        }
        results
    }

    fn par_kde_recursion(&self, kernel: KernelType, bandwidth: f64, queries: &[Self::Float], n_queries: usize, dim: usize) -> Vec<f64> {
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
