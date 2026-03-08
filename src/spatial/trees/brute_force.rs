use crate::{array::NdArray, spatial::{common::DistanceMetric}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BFNode {
    pub start: usize,
    pub end: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BruteForce {
    pub nodes: Vec<BFNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl BruteForce {
    pub fn new(mut data: NdArray<f64>, metric: DistanceMetric) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        if matches!(metric, DistanceMetric::Cosine) {
            for i in 0..n_points {
                let normed = metric.pre_transform(data.row(i)).into_owned();
                data.set_row(i, &normed);
            }
        }
        
        let root = BFNode {
            start: 0,
            end: n_points,
        };
        BruteForce {
            nodes: vec![root],
            indices: (0..n_points).collect(),
            data: data,
            n_points,
            dim,
            leaf_size: n_points,
            metric,
        }
    }
}

impl SpatialTree for BruteForce {
    type Node = BFNode;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[BFNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { &self.data.as_slice() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, _idx: usize) -> Option<usize> { None }
    fn node_right(&self, _idx: usize) -> Option<usize> { None }

    fn min_distance_to_node(&self, _node_idx: usize, _query: &[f64]) -> f64 { 0.0 }

    fn knn_child_order(&self, _node_idx: usize, _query: &[f64]) -> (usize, usize) {
        unreachable!("BruteForce has no tree structure")
    }
}

impl KnnQuery for BruteForce {

}

impl RadiusQuery for BruteForce {

}

impl KdeQuery for BruteForce {

}