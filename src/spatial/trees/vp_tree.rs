use std::{cmp::Ordering};
use crate::{array::{NdArray, Shape}, spatial::{common::DistanceMetric}};
use crate::random::Generator;
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum VantagePointSelection {
    #[default]
    First,
    Random,
    Variance { sample_size: usize },
}

impl VantagePointSelection{
    fn select_vantage(
    &self,
    start: usize,
    end: usize,
    data: &NdArray<f64>,
    indices: &[usize],
    metric: &DistanceMetric,
) -> usize {
        match self {
            VantagePointSelection::First => start,
            VantagePointSelection::Random => {
                let mut rng = Generator::new();
                let i = rng.randint(start as i64, end as i64, Shape::scalar());
                i.item() as usize
            },
            VantagePointSelection::Variance { sample_size } => {
                let mut rng = Generator::new();
                let n = end - start;
                let k = (*sample_size).min(n);

                let candidates: Vec<usize> = (0..k)
                    .map(|_| rng.randint(start as i64, end as i64, Shape::scalar()).item() as usize)
                    .collect();

                candidates.iter().max_by(|&&a, &&b| {
                    let pa = data.row(indices[a]).to_vec();
                    let var_a = candidates.iter().map(|&j| {
                        let d = metric.distance(data.row(indices[j]), &pa);
                        d * d
                    }).sum::<f64>();

                    let pb = data.row(indices[b]).to_vec();
                    let var_b = candidates.iter().map(|&j| {
                        let d = metric.distance(data.row(indices[j]), &pb);
                        d * d
                    }).sum::<f64>();

                    var_a.partial_cmp(&var_b).unwrap_or(Ordering::Equal)
                }).copied().unwrap_or(start)
            },
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,

    pub radius: f64,
    pub center: Vec<f64>,
    pub bounding_radius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPTree {
    pub nodes: Vec<VPNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub selection_method: VantagePointSelection,
    pub data_is_reordered: bool,
}

impl VPTree {
    pub fn new(mut data: NdArray<f64>, leaf_size: usize, metric: DistanceMetric, selection_method: VantagePointSelection) -> Self{
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
        let mut tree = VPTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size,
            metric,
            selection_method,
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
        let mut new_data = vec![0.0; self.data.len()];

        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(self.data.row(old_idx));
        }

        self.data = NdArray::from_vec(Shape::new(vec![self.n_points, self.dim]), new_data);
    }

    fn init_node(&mut self, start: usize, end: usize) -> (f64, usize) {
        let vantage_idx = self.selection_method.select_vantage(
            start,
            end,
            &self.data,
            &self.indices,
            &self.metric,
        ); 
        self.indices.swap(start, vantage_idx);  

        let vantage_point = self.data.row(self.indices[start]).to_vec();

        let mut idx_dist: Vec<(usize, f64)> = (start + 1..end)
            .map(|i| {
                let p = self.data.row(self.indices[i]);
                let dist = self.metric.distance(p, &vantage_point);
                (i, dist)
            })
            .collect();

        let n = idx_dist.len();
        debug_assert!(n > 0, "init_node called with only vantage point");
        let mid = n / 2;

        idx_dist.select_nth_unstable_by(mid, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });

        let median_radius = if n % 2 == 1 {
            idx_dist[mid].1
        } else {
            let lower_max = idx_dist[..mid]
                .iter()
                .map(|(_, d)| *d)
                .fold(f64::NEG_INFINITY, f64::max);
            (lower_max + idx_dist[mid].1) * 0.5
        };

        let reordered: Vec<usize> = idx_dist.iter().map(|(i, _)| self.indices[*i]).collect();
        self.indices[start + 1..end].copy_from_slice(&reordered);

        (median_radius, start + 1 + mid)
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let count = end - start;
        let dim = self.dim;

        let mut center = vec![0.0; dim];
        for i in start..end {
            let p = self.data.row(self.indices[i]);
            for d in 0..dim {
                center[d] += p[d];
            }
        }
        for d in 0..dim {
            center[d] /= count as f64;
        }

        let bounding_radius = (start..end)
            .map(|i| self.metric.distance(self.data.row(self.indices[i]), &center))
            .fold(0.0f64, f64::max);

        if count <= self.leaf_size {
            let node_idx = self.nodes.len();
            self.nodes.push(VPNode {
                start,
                end,
                left: None,
                right: None,
                radius: 0.0,
                center,
                bounding_radius,
            });
            return node_idx;
        }

        let (radius, mid) = self.init_node(start, end);

        let node_idx = self.nodes.len();
        self.nodes.push(VPNode {
            start,
            end,
            left: None,
            right: None,
            radius,
            center,
            bounding_radius,
        });

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }
}

impl SpatialTree for VPTree {
    type Node = VPNode;
    const REDUCED: bool = false;

    fn nodes(&self) -> &[VPNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[f64] { self.data.as_slice_unchecked() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}
    fn data_is_reordered(&self) -> bool { self.data_is_reordered }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        let d = self.metric.distance(query, &node.center);
        (d - node.bounding_radius).max(0.0)
    }

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let d = self.metric.distance(query, self.get_point(node.start));
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        if d < node.radius { (l, r) } else { (r, l) }
    }
}

impl KnnQuery for VPTree {

}

impl RadiusQuery for VPTree {

}

impl KdeQuery for VPTree {

}