use crate::{Generator, array::{NdArray, Shape}, projection::{ProjectionType, RandomProjection, random_projection::ProjectionDirection}, spatial::{HeapItem, common::{DistanceMetric, IronFloat}, spatial_tree::{ChildTraversal, TraversalPlan}}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, AnnQuery, KdeQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};


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

    fn child_lower_bound(&self, _child_idx: usize, _query: &[T]) -> T { T::zero() }
    fn traversal_order(&self, node_idx: usize, query: &[T]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let proj = node.direction.project_t(query);
        if proj <= node.split { (l, r) } else { (r, l) }
    }

    fn plan_traversal(&self, node_idx: usize, query: &[T]) -> TraversalPlan<T> {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let proj = node.direction.project_t(query);
        let bound = T::from((proj - node.split).abs().powi(2)).unwrap();

        let (first, second) = if proj <= node.split { (l, r) } else { (r, l) };

        TraversalPlan {
            first: ChildTraversal { child_idx: first, lower_bound: T::zero() },
            second: ChildTraversal { child_idx: second, lower_bound: bound },
        }
    }
}

impl<T: IronFloat> KnnQuery for RPTree<T> {

}

impl<T: IronFloat> RadiusQuery for RPTree<T> {

}

impl<T: IronFloat> KdeQuery for RPTree<T> {}

impl<T: IronFloat> AnnQuery for RPTree<T> {

}
