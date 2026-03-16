use crate::spatial::common::DistanceMetric;

pub trait SpatialTree: Sync {
    type Node;
    const REDUCED: bool;

    fn nodes(&self) -> &[Self::Node];
    fn indices(&self) -> &[usize];
    fn data(&self) -> &[f64];
    fn dim(&self) -> usize;
    fn metric(&self) -> &DistanceMetric;

    fn node_start(&self, idx: usize) -> usize;
    fn node_end(&self, idx: usize) -> usize;
    fn node_left(&self, idx: usize) -> Option<usize>;
    fn node_right(&self, idx: usize) -> Option<usize>;

    fn root(&self) -> usize { 0 }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64;

    fn knn_child_order(&self, node_idx: usize, _query: &[f64]) -> (usize, usize) {
        (self.node_left(node_idx).unwrap(), self.node_right(node_idx).unwrap())
    }

    fn node_projection(&self, node_idx: usize, query: &[f64]) -> (usize, usize, f64) {
        let dist = self.min_distance_to_node(node_idx, query);
        let (first, second) = self.knn_child_order(node_idx, query);
        (first, second, dist)
    }

    fn data_is_reordered(&self) -> bool;

    fn get_point(&self, i: usize) -> &[f64] {
        let dim = self.dim();
        let row = if self.data_is_reordered() { i } else { self.indices()[i] };
        &self.data()[row * dim..(row + 1) * dim]
    }

    fn n_points(&self) -> usize;
}