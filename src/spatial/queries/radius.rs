use crate::array::NdArray;
use rayon::prelude::*;
use crate::spatial::SpatialTree;

const RAD_PAR_THRESHOLD: usize = 512;

pub trait RadiusQuery: SpatialTree {
    fn query_radius(&self, query: &[Self::Float], radius: Self::Float) -> Vec<(usize, Self::Float)> {
        let mut results = Vec::new();
        let rad = match Self::REDUCED {
            true => self.metric().to_reduced(radius),
            false => radius,
        };
        self.query_radius_recursive(self.root(), query, rad, &mut results);
        results
    }

    fn query_radius_recursive(
        &self,
        node_idx: usize,
        query: &[Self::Float],
        radius: Self::Float,
        results: &mut Vec<(usize, Self::Float)>,
    ) {
        if self.is_leaf(node_idx) {
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

        let plan = self.plan_traversal(node_idx, query);

        if plan.first.lower_bound <= radius {
            self.query_radius_recursive(plan.first.child_idx, query, radius, results);
        }

        if plan.second.lower_bound <= radius {
            self.query_radius_recursive(plan.second.child_idx, query, radius, results);
        }
    }

    fn query_radius_batch(&self, queries: &NdArray<Self::Float>, radius: Self::Float) -> Vec<Vec<(usize, Self::Float)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[Self::Float] = &queries_cow;
        if n_queries >= RAD_PAR_THRESHOLD {
            self.par_radius_batch(queries_slice, n_queries, dim, radius)
        } else {
            self.seq_radius_batch(queries_slice, n_queries, dim, radius)
        }
    }

    fn seq_radius_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, radius: Self::Float) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }

    fn par_radius_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, radius: Self::Float) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_radius(query, radius)
            })
            .collect()
    }
}
