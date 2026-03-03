use crate::{Shape, array::NdArray, spatial::common::DistanceMetric};
use super::spatial_query::{SpatialQuery};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallNode {
    pub center: Vec<f64>,
    pub radius: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallTree {
    pub nodes: Vec<BallNode>,
    pub indices: Vec<usize>,
    pub data: NdArray<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl BallTree {
    pub fn new(data: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = data.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        let n_points = shape[0];
        let dim = shape[1];

        let mut tree = BallTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: data.clone(),
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

    fn init_node(&self, start: usize, end: usize) -> (Vec<f64>, f64) {
        let n = (end - start) as f64;
        let mut centroid = vec![0.0; self.dim];

        for i in start..end {
            let p = self.data.row(self.indices[i]);
            for (j, &x) in p.iter().enumerate() {
                centroid[j] += x;
            }
        }

        
        for c in &mut centroid {
            *c /= n;
        }

        let mut max_dist: f64 = 0.0;
        for i in start..end {
            let p = self.data.row(self.indices[i]);
            let dist = self.metric.reduced_distance(p, &centroid);

            if  dist > max_dist {
                max_dist = dist;
            }
        }
        (centroid, max_dist)
    }

    fn furthest_from(&self, query: &[f64], start: usize, end: usize) -> usize {
        (start..end)
            .max_by(|&a, &b| {
                let da = self.metric.reduced_distance(query, self.data.row(self.indices[a]));
                let db = self.metric.reduced_distance(query, self.data.row(self.indices[b]));
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }

    fn pivot_partition(&mut self, start: usize, end: usize) -> usize {
        let centroid = {
            let n = (end - start) as f64;
            let mut c = vec![0.0; self.dim];
            for i in start..end {
                let p = self.data.row(self.indices[i]);
                for (j, &x) in p.iter().enumerate() { c[j] += x / n; }
            }
            c
        };

        let p1_slot = self.furthest_from(&centroid, start, end);
        let p1 = self.data.row(self.indices[p1_slot]).to_vec();

        let p2_slot = self.furthest_from(&p1, start, end);
        let p2 = self.data.row(self.indices[p2_slot]).to_vec();

        let axis: Vec<f64> = p2.iter().zip(&p1).map(|(a, b)| a - b).collect();

        let mut projections: Vec<(f64, usize)> = (start..end)
            .map(|i| {
                let p = self.data.row(self.indices[i]);
                let proj = p.iter().zip(&axis).map(|(x, a)| x * a).sum::<f64>();
                (proj, self.indices[i])
            })
            .collect();

        let mid_offset = (end - start) / 2;
        projections.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        self.indices[start..end].copy_from_slice(
            &projections.iter().map(|&(_, idx)| idx).collect::<Vec<_>>()
        );

        start + mid_offset
    }


    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (center, radius) = self.init_node(start, end);
        
        let node_idx = self.nodes.len();

        self.nodes.push(BallNode {
            center,
            radius,
            start,
            end,
            left: None,
            right: None,
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
        }

        let mut mid = self.pivot_partition(start, end);

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

impl SpatialQuery for BallTree {
    type Node = BallNode;

    fn nodes(&self) -> &[BallNode] { &self.nodes }
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
        let d = self.metric.reduced_distance(query, &node.center);
        (d - node.radius).max(0.0)
    }

    fn knn_child_order(&self, node_idx: usize, query: &[f64]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let dl = self.metric.reduced_distance(query, &self.nodes[l].center);
        let dr = self.metric.reduced_distance(query, &self.nodes[r].center);
        if dl <= dr { (l, r) } else { (r, l) }
    }
}