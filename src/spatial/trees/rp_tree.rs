use std::collections::BinaryHeap;

use crate::{Generator, array::{NdArray, Shape}, projection::{ProjectionType, RandomProjection, random_projection::ProjectionDirection}, spatial::{HeapItem, common::{DistanceMetric, IronFloat}}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, AnnQuery};
use crate::spatial::SpatialTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const KNN_PAR_THRESHOLD: usize = 512;

// RPNode stores the split value and projection direction in f64 regardless of
// the tree's data type. Projections are fundamentally f64 computations currently.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RPNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub direction: ProjectionDirection,
    pub split: f64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct RPTree<T: IronFloat> {
    pub nodes: Vec<RPNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<T>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub projection_type: ProjectionType,
    rng: Generator,
    pub data_is_reordered: bool,
}

impl<T: IronFloat> RPTree<T> {
    pub fn new(
        mut data: NdArray<T>,
        leaf_size: usize,
        metric: DistanceMetric,
        projection_type: ProjectionType,
        seed: u64,
    ) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        let rng = Generator::from_seed(seed);

        if (matches!(metric, DistanceMetric::Cosine) && !data.is_owned()) || !data.is_contiguous() {
            data = data.to_contiguous();
        }
        if matches!(metric, DistanceMetric::Cosine) {
            for i in 0..n_points {
                let normed = metric.pre_transform(data.row(i)).into_owned();
                data.set_row(i, &normed);
            }
        }

        let will_reorder = data.is_owned();
        let mut tree = RPTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size,
            metric,
            projection_type,
            rng,
            data_is_reordered: false,
        };

        tree.build_recursive(0, n_points);
        if will_reorder {
            tree.reorder_data();
            tree.data_is_reordered = true;
        }
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

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let node_idx = self.nodes.len();

        self.nodes.push(RPNode {
            start,
            end,
            left: None,
            right: None,
            direction: ProjectionDirection::Empty,
            split: 0.0,
        });

        let count = end - start;
        if count <= self.leaf_size {
            return node_idx;
        }

        let (direction, split, mid) = RandomProjection::rp_split(
            &self.data,
            &mut self.indices,
            start,
            end,
            self.dim,
            self.projection_type,
            &mut self.rng,
        );

        self.nodes[node_idx].direction = direction;
        self.nodes[node_idx].split = split;

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn knn_recursive_inner(
        &self,
        node_idx: usize,
        query: &[T],
        heap: &mut BinaryHeap<HeapItem<T>>,
        k: usize,
    ) {
        let node = &self.nodes[node_idx];

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.reduced_distance(query, self.get_point(i));
                if heap.len() < k {
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                }
            }
            return;
        }

        let (first, second, margin) = self.node_projection(node_idx, query);

        self.knn_recursive_inner(first, query, heap, k);

        let dominated = heap.len() == k
            && margin >= heap.peek().unwrap().distance;
        if !dominated {
            self.knn_recursive_inner(second, query, heap, k);
        }
    }

    fn ann_candidates_inner_rp(
        &self,
        query: &[T],
        k: usize,
        n_candidates: usize,
    ) -> Vec<(usize, T)> {
        let mut queue: BinaryHeap<std::cmp::Reverse<HeapItem<T>>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<HeapItem<T>> = BinaryHeap::new();

        queue.push(std::cmp::Reverse(HeapItem { distance: T::zero(), index: 0 }));

        while let Some(std::cmp::Reverse(HeapItem { distance: node_dist, index: node_idx })) = queue.pop() {
            if candidates.len() >= k {
                let worst_k = candidates.peek().unwrap().distance;
                if node_dist > worst_k {
                    break;
                }
            }

            let node = &self.nodes[node_idx];

            if node.left.is_none() {
                for i in node.start..node.end {
                    let dist = self.metric.reduced_distance(query, self.get_point(i));
                    if candidates.len() < n_candidates {
                        candidates.push(HeapItem { distance: dist, index: self.indices[i] });
                    } else if dist < candidates.peek().unwrap().distance {
                        candidates.pop();
                        candidates.push(HeapItem { distance: dist, index: self.indices[i] });
                    }
                }
                if candidates.len() >= n_candidates {
                    break;
                }
            } else {
                let (first, second, margin_sq) = self.node_projection(node_idx, query);
                queue.push(std::cmp::Reverse(HeapItem { distance: T::zero(), index: first }));
                queue.push(std::cmp::Reverse(HeapItem { distance: margin_sq, index: second }));
            }
        }

        let mut results: Vec<(usize, T)> = candidates.into_iter()
            .map(|item| (item.index, item.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn seq_ann_batch(&self, queries: &[T], n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, T)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }

    fn par_ann_batch(&self, queries: &[T], n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, T)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }

    pub fn query_ann_batch(&self, queries: &NdArray<T>, k: usize, n_candidates: usize) -> Vec<Vec<(usize, T)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[T] = &queries_cow;
        if n_queries >= KNN_PAR_THRESHOLD {
            self.par_ann_batch(queries_slice, n_queries, dim, k, n_candidates)
        } else {
            self.seq_ann_batch(queries_slice, n_queries, dim, k, n_candidates)
        }
    }

    pub fn query_ann(&self, query: &[T], k: usize, n_candidates: usize) -> Vec<(usize, T)> {
        self.ann_candidates_inner_rp(query, k, n_candidates.max(k))
    }

    fn radius_recursive_inner(
        &self,
        node_idx: usize,
        query: &[T],
        radius: T,
        results: &mut Vec<(usize, T)>,
    ) {
        let node = &self.nodes[node_idx];

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.reduced_distance(query, self.get_point(i));
                if dist <= radius {
                    results.push((self.indices[i], dist));
                }
            }
            return;
        }

        let (first, second, margin_sq) = self.node_projection(node_idx, query);

        self.radius_recursive_inner(first, query, radius, results);

        if margin_sq <= radius {
            self.radius_recursive_inner(second, query, radius, results);
        }
    }
}


impl<T: IronFloat> SpatialTree for RPTree<T> {
    type Node = RPNode;
    type Float = T;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[RPNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[T] { self.data.as_slice_unchecked() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize { self.n_points }
    fn data_is_reordered(&self) -> bool { self.data_is_reordered }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }

    fn min_distance_to_node(&self, node_idx: usize, query: &[T]) -> T {
        let node = &self.nodes[node_idx];
        let proj = node.direction.project_t(query);
        T::from((proj - node.split).abs()).unwrap()
    }

    fn knn_child_order(&self, _node_idx: usize, _query: &[T]) -> (usize, usize) { (0, 0) }

    fn node_projection(&self, node_idx: usize, query: &[T]) -> (usize, usize, T) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let proj = node.direction.project_t(query);
        let dist = (proj - node.split).abs();
        let (first, second) = if proj <= node.split { (l, r) } else { (r, l) };
        (first, second, T::from(dist * dist).unwrap())
    }
}

impl<T: IronFloat> KnnQuery for RPTree<T> {
    fn query_knn_recursive(&self, node_idx: usize, query: &[T], heap: &mut BinaryHeap<HeapItem<T>>, k: usize) {
        self.knn_recursive_inner(node_idx, query, heap, k)
    }
}

impl<T: IronFloat> RadiusQuery for RPTree<T> {
    fn query_radius_recursive(&self, node_idx: usize, query: &[T], radius: T, results: &mut Vec<(usize, T)>) {
        self.radius_recursive_inner(node_idx, query, radius, results);
    }
}

impl<T: IronFloat> AnnQuery for RPTree<T> {

}
