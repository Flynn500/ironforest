use crate::{array::NdArray, spatial::common::DistanceMetric};
use super::spatial_query::{SpatialQuery};

#[derive(Clone, Debug)]
pub struct BFNode {
    pub start: usize,
    pub end: usize,
}


#[derive(Debug, Clone)]
pub struct BruteForce {
    pub nodes: Vec<BFNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl BruteForce {
    pub fn new(points: &[f64], n_points: usize, dim: usize, metric: DistanceMetric) -> Self {
        let root = BFNode {
            start: 0,
            end: n_points,
        };
        BruteForce {
            nodes: vec![root],
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size: n_points,
            metric,
        }
    }

    pub fn from_ndarray(array: &NdArray<f64>, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, metric)
    }
}

impl SpatialQuery for BruteForce {
    type Node = BFNode;

    fn nodes(&self) -> &[BFNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { &self.data }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { Some(0) }
    fn node_right(&self, idx: usize) -> Option<usize> { Some(0) }

    fn min_distance_to_node(&self, _node_idx: usize, _query: &[f64]) -> f64 { 0.0 }

    fn knn_child_order(&self, _node_idx: usize, _query: &[f64]) -> (usize, usize) {
        unreachable!("BruteForce has no tree structure")
    }
}