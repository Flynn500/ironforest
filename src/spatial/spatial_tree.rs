use crate::spatial::common::{DistanceMetric, IronFloat};

pub struct ChildTraversal<F> {
    pub child_idx: usize,
    pub lower_bound: F,
}

pub struct TraversalPlan<F> {
    pub first: ChildTraversal<F>,
    pub second: ChildTraversal<F>,
}
pub trait SpatialTree: Sync {
    type Node;
    type Float: IronFloat;
    const REDUCED: bool;

    fn nodes(&self) -> &[Self::Node];
    fn indices(&self) -> &[usize];
    fn data(&self) -> &[Self::Float];
    fn dim(&self) -> usize;
    fn metric(&self) -> &DistanceMetric;

    fn node_start(&self, idx: usize) -> usize;
    fn node_end(&self, idx: usize) -> usize;
    fn node_left(&self, idx: usize) -> Option<usize>;
    fn node_right(&self, idx: usize) -> Option<usize>;

    fn root(&self) -> usize { 0 }

    fn is_leaf(&self, idx: usize) -> bool {
        self.node_left(idx).is_none()
    }

    fn child_lower_bound(&self, child_idx: usize, query: &[Self::Float]) -> Self::Float;

    fn traversal_order(&self, node_idx: usize, _query: &[Self::Float]) -> (usize, usize) {
        (self.node_left(node_idx).unwrap(), self.node_right(node_idx).unwrap())
    }

    fn plan_traversal(&self, node_idx: usize, query: &[Self::Float]) -> TraversalPlan<Self::Float> {
        let (first, second) = self.traversal_order(node_idx, query);
        TraversalPlan {
            first: ChildTraversal { child_idx: first, lower_bound: self.child_lower_bound(first, query) },
            second: ChildTraversal { child_idx: second, lower_bound: self.child_lower_bound(second, query) },
        }
    }

    fn data_is_reordered(&self) -> bool;

    fn get_point(&self, i: usize) -> &[Self::Float] {
        let dim = self.dim();
        let row = if self.data_is_reordered() { i } else { self.indices()[i] };
        &self.data()[row * dim..(row + 1) * dim]
    }

    fn n_points(&self) -> usize;
}
