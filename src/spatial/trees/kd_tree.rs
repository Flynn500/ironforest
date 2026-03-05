use crate::{array::{NdArray, Shape}, spatial::common::DistanceMetric};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KDNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,

    pub axis: usize,
    pub split: f64,

    pub bbox_min: Vec<f64>,
    pub bbox_max: Vec<f64>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KDTree {
    pub nodes: Vec<KDNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl KDTree {
    pub fn new(data: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        let mut transformed_data = data.clone();
        if matches!(metric, DistanceMetric::Cosine) {
            for i in 0..n_points {
                let normed = metric.pre_transform(transformed_data.row(i)).into_owned();
                transformed_data.set_row(i, &normed);
            }
        }

        let mut tree = KDTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: transformed_data,
            n_points,
            dim,
            leaf_size,
            metric,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();
        tree
    }

    fn reorder_data(&mut self) {
        let mut new_data = vec![0.0; self.data.len()];

        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(self.data.row(old_idx));
        }

        self.data = NdArray::from_vec(Shape::new(vec![self.n_points, self.dim]), new_data);
    }

    fn init_node(&self, start: usize, end: usize) -> (Vec<f64>, Vec<f64>, usize) {
        let mut min = vec![f64::INFINITY; self.dim];
        let mut max = vec![f64::NEG_INFINITY; self.dim];

        for i in start..end {
            let p = self.data.row(self.indices[i]);
            for (j, &x) in p.iter().enumerate() {
                max[j] = max[j].max(x);
                min[j] = min[j].min(x);

            }
        }
        let mut best_dim = 0;
        let mut best_spread = 0.0;
        for d in 0..self.dim {
            let spread = max[d] - min[d];
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        (min, max, best_dim)        
    }

    fn partition(&mut self, start: usize, end: usize, dim: usize) -> usize {
        let mut slots: Vec<(f64, usize)> = (start..end)
            .map(|slot| (self.data.row(self.indices[slot])[dim], slot))
            .collect();

        let mid_offset = (end - start) / 2;

        slots.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        let new_order: Vec<usize> = slots
            .iter()
            .map(|&(_val, slot)| self.indices[slot])
            .collect();

        self.indices[start..end].copy_from_slice(&new_order);

        start + mid_offset
    }


    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (min, max, axis)  = self.init_node(start, end);
        let node_idx = self.nodes.len();

        self.nodes.push(KDNode { 
            start, 
            end, 
            left: None, 
            right: None, 
            axis, 
            split: 0.0, 
            bbox_min: min, 
            bbox_max: max,
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
        }

        let mut mid = self.partition(start, end, axis);
        let split = self.data.row(self.indices[mid])[axis];
        self.nodes[node_idx].split = split;

        if mid == start {
            mid = start + 1;
        } else if mid == end {
            mid = end - 1;
        }

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);
        
        node_idx
    }
}

impl SpatialTree for KDTree {
    type Node = KDNode;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[KDNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { self.data.as_slice() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }


    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        let clamped: Vec<f64> = query.iter().enumerate()
            .map(|(d, &q)| q.clamp(node.bbox_min[d], node.bbox_max[d]))
            .collect();
        self.metric.reduced_distance(&clamped, query)
    }

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        if query[node.axis] < node.split { (l, r) } else { (r, l) }
    }
}

impl KnnQuery for KDTree {

}

impl RadiusQuery for KDTree {

}

impl KdeQuery for KDTree {

}