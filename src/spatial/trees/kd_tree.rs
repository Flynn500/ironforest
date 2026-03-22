use crate::spatial::common::{DistanceMetric, IronFloat};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery, AnnQuery};
use crate::spatial::SpatialTree;
use crate::{NdArray, Shape};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct KDNode<T> {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,

    pub axis: usize,
    pub split: T,

    pub bbox_min: Vec<T>,
    pub bbox_max: Vec<T>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct KDTree<T: IronFloat> {
    pub nodes: Vec<KDNode<T>>,
    pub indices: Vec<usize>,
    pub data: NdArray<T>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub data_is_reordered: bool,
}

impl<T: IronFloat> KDTree<T> {
    pub fn new(mut data: NdArray<T>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

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
        let mut tree = KDTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size,
            metric,
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

    fn init_node(&self, start: usize, end: usize) -> (Vec<T>, Vec<T>, usize) {
        let mut min = vec![T::infinity(); self.dim];
        let mut max = vec![T::neg_infinity(); self.dim];

        for i in start..end {
            let p = self.data.row(self.indices[i]);
            for (j, &x) in p.iter().enumerate() {
                if x > max[j] { max[j] = x; }
                if x < min[j] { min[j] = x; }
            }
        }
        let mut best_dim = 0;
        let mut best_spread = T::zero();
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
        let mut slots: Vec<(T, usize)> = (start..end)
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
            split: T::zero(),
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

impl<T: IronFloat> SpatialTree for KDTree<T> {
    type Node = KDNode<T>;
    type Float = T;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[KDNode<T>] { &self.nodes }
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

    fn child_lower_bound(&self, child_idx: usize, query: &[T]) -> T {
        let node = &self.nodes[child_idx];
        let clamped: Vec<T> = query.iter().enumerate()
            .map(|(d, &q)| q.max(node.bbox_min[d]).min(node.bbox_max[d]))
            .collect();
        self.metric.reduced_distance(&clamped, query)
    }

    fn traversal_order(&self, node_idx: usize, query: &[T]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        if query[node.axis] < node.split { (l, r) } else { (r, l) }
    }
}

impl<T: IronFloat> KnnQuery for KDTree<T> {}
impl<T: IronFloat> RadiusQuery for KDTree<T> {}
impl<T: IronFloat> KdeQuery for KDTree<T> {}
impl<T: IronFloat> AnnQuery for KDTree<T> {}
