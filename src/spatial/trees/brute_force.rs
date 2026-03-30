use crate::{array::NdArray, spatial::common::{DistanceMetric, IronFloat}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery, AnnQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BFNode {
    pub start: usize,
    pub end: usize,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct BruteForce<T: IronFloat> {
    pub nodes: Vec<BFNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<T>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    /// Always `true`: BruteForce never reorders data and `indices` is the
    /// identity permutation, so `data[i]` is the correct point at position `i`.
    pub data_is_reordered: bool,
}

impl<T: IronFloat> BruteForce<T> {
    pub fn new(mut data: NdArray<T>, metric: DistanceMetric) -> Self {
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

        let root = BFNode {
            start: 0,
            end: n_points,
        };
        BruteForce {
            nodes: vec![root],
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size: n_points,
            metric,
            data_is_reordered: true,
        }
    }
}

impl<T: IronFloat> SpatialTree for BruteForce<T> {
    type Node = BFNode;
    type Float = T;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[BFNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[T] { self.data.as_slice_unchecked() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize { self.n_points }
    fn data_is_reordered(&self) -> bool { self.data_is_reordered }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, _idx: usize) -> Option<usize> { None }
    fn node_right(&self, _idx: usize) -> Option<usize> { None }

    fn child_lower_bound(&self, _child_idx: usize, _query: &[Self::Float]) -> Self::Float { T::zero() }

    fn traversal_order(&self, _node_idx: usize, _query: &[Self::Float]) -> (usize, usize) {
        unreachable!("BruteForce has no tree structure")
    }
}

impl<T: IronFloat> KnnQuery for BruteForce<T> {}
impl<T: IronFloat> RadiusQuery for BruteForce<T> {}
impl<T: IronFloat> KdeQuery for BruteForce<T> {}
impl<T: IronFloat> AnnQuery for BruteForce<T> {}
