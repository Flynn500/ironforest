use std::collections::BinaryHeap;
use crate::{array::NdArray, spatial::HeapItem};
use rayon::prelude::*;
use crate::spatial::SpatialTree;
use num_traits::float::Float as _;

const KNN_PAR_THRESHOLD: usize = 512;


pub trait KnnQuery: SpatialTree {
    fn query_knn(&self, query: &[Self::Float], k: usize) -> Vec<(usize, Self::Float)> {
        if k == 0 || self.n_points() == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k);
        self.query_knn_recursive(self.root(), query, &mut heap, k);
        heap.into_sorted_vec()
            .into_iter()
            .map(|item| {
                let dist = if Self::REDUCED {
                    self.metric().post_transform(item.distance)
                } else {
                    item.distance
                };
                (item.index, dist)
            })
            .collect()
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[Self::Float], heap: &mut BinaryHeap<HeapItem<Self::Float>>, k: usize) {
        if self.node_left(node_idx).is_none() {
            for i in self.node_start(node_idx)..self.node_end(node_idx) {
                let dist = match Self::REDUCED {
                    true => self.metric().reduced_distance(query, self.get_point(i)),
                    false => self.metric().distance(query, self.get_point(i)),
                };

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

        let threshold = heap.peek().map(|t| t.distance).unwrap_or(Self::Float::infinity());

        if heap.len() == k && node_dist > threshold {
            return;
        }

        let first_dist = self.min_distance_to_node(first, query);
        if heap.len() < k || first_dist <= threshold {
            self.query_knn_recursive(first, query, heap, k);
        }

        let second_dist = self.min_distance_to_node(second, query);
        if heap.len() < k || second_dist <= threshold {
            self.query_knn_recursive(second, query, heap, k);
        }
    }

    fn query_knn_batch(&self, queries: &NdArray<Self::Float>, k: usize) -> Vec<Vec<(usize, Self::Float)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[Self::Float] = &queries_cow;
        if n_queries >= KNN_PAR_THRESHOLD {
            self.par_knn_batch(queries_slice, n_queries, dim, k)
        } else {
            self.seq_knn_batch(queries_slice, n_queries, dim, k)
        }
    }

    fn seq_knn_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, k: usize) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_knn(query, k)
            })
            .collect()
    }

    fn par_knn_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, k: usize) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_knn(query, k)
            })
            .collect()
    }
}
