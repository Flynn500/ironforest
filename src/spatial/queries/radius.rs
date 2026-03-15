use crate::array::NdArray;
use rayon::prelude::*;
use crate::spatial::SpatialTree;

const RAD_PAR_THRESHOLD: usize = 512;

pub trait RadiusQuery: SpatialTree {
    fn query_radius(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        let rad = match Self::REDUCED {
            true => self.metric().to_reduced(radius),
            false => radius,
        };
        self.query_radius_recursive(self.root(), query, rad, &mut results);
        results
    }

    fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<(usize, f64)>) {
        if self.min_distance_to_node(node_idx, query) > radius {
            return;
        }

        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = match Self::REDUCED {
                    true => self.metric().reduced_distance(query, self.get_point(i)),
                    false => self.metric().distance(query, self.get_point(i)),
                };
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

    fn query_radius_batch(&self, queries: &NdArray<f64>, radius: f64) -> Vec<Vec<(usize, f64)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        if n_queries >= RAD_PAR_THRESHOLD {
            self.par_radius_batch(queries, n_queries, dim, radius)
        } else {
            self.seq_radius_batch(queries, n_queries, dim, radius)
        }
    }

    fn seq_radius_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, radius: f64) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries.as_slice_unchecked()[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }

    fn par_radius_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, radius: f64) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries.as_slice_unchecked()[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }
}