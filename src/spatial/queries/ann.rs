use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::{array::NdArray, spatial::HeapItem};
use rayon::prelude::*;
use crate::spatial::SpatialTree;

const ANN_PAR_THRESHOLD: usize = 512;

pub trait AnnQuery: SpatialTree {
    fn query_ann(&self, query: &[f64], k: usize, n_candidates: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.n_points() == 0 {
            return Vec::new();
        }
        self.ann_candidates_inner(query, k, n_candidates.max(k))
    }

    fn ann_candidates_inner(&self, query: &[f64], k: usize, n_candidates: usize) -> Vec<(usize, f64)> {
        let mut queue: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<HeapItem> = BinaryHeap::new();

        queue.push(Reverse(HeapItem { distance: 0.0, index: self.root() }));

        while let Some(Reverse(HeapItem { distance: node_dist, index: node_idx })) = queue.pop() {
            if candidates.len() >= k {
                if node_dist > candidates.peek().unwrap().distance {
                    break;
                }
            }

            if self.node_left(node_idx).is_none() {
                for i in self.node_start(node_idx)..self.node_end(node_idx) {
                    let dist = match Self::REDUCED {
                        true => self.metric().reduced_distance(query, self.get_point(i)),
                        false => self.metric().distance(query, self.get_point(i)),
                    };
                    if candidates.len() < n_candidates {
                        candidates.push(HeapItem { distance: dist, index: self.indices()[i] });
                    } else if dist < candidates.peek().unwrap().distance {
                        candidates.pop();
                        candidates.push(HeapItem { distance: dist, index: self.indices()[i] });
                    }
                }
                if candidates.len() >= n_candidates {
                    break;
                }
            } else {
                let (first, second, margin) = self.node_projection(node_idx, query);
                queue.push(Reverse(HeapItem { distance: 0.0, index: first }));
                queue.push(Reverse(HeapItem { distance: margin, index: second }));
            }
        }
        let mut results: Vec<(usize, f64)> = candidates.into_iter()
            .map(|item| {
                let dist = if Self::REDUCED {
                    self.metric().post_transform(item.distance)
                } else {
                    item.distance
                };
                (item.index, dist)
            })
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn query_ann_batch(&self, queries: &NdArray<f64>, k: usize, n_candidates: usize) -> Vec<Vec<(usize, f64)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        if n_queries >= ANN_PAR_THRESHOLD {
            self.par_ann_batch(queries, n_queries, dim, k, n_candidates)
        } else {
            self.seq_ann_batch(queries, n_queries, dim, k, n_candidates)
        }
    }

    fn seq_ann_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries.as_slice_unchecked()[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }

    fn par_ann_batch(&self, queries: &NdArray<f64>, n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries.as_slice_unchecked()[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }
}