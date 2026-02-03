use std::collections::BinaryHeap;
use crate::{array::{NdArray, Shape}, spatial::common::{ApproxCriterion, DistanceMetric, HeapItem, KernelType}};


#[derive(Clone, Debug)]
pub struct BallNode {
    pub center: Vec<f64>,
    pub radius: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct BallTree {
    pub nodes: Vec<BallNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
}

impl BallTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric) -> Self{
        let mut tree = BallTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
            metric,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric)
    }
    
    fn reorder_data(&mut self) {
        let mut new_data = vec![0.0; self.data.len()];
        
        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let src = old_idx * self.dim;
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(&self.data[src..src + self.dim]);
        }
        
        self.data = new_data;
    }

    fn get_point(&self, i: usize) -> &[f64] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn get_point_from_idx(&self, i: usize) -> &[f64] {
        let original_idx = self.indices[i];
        &self.data[original_idx * self.dim..(original_idx + 1) * self.dim]
    }

    fn init_node(&self, start: usize, end: usize) -> (Vec<f64>, f64) {
        let n = (end - start) as f64;
        let mut centroid = vec![0.0; self.dim];

        for i in start..end {
            let p = self.get_point_from_idx(i);
            for (j, &x) in p.iter().enumerate() {
                centroid[j] += x;
            }
        }

        
        for c in &mut centroid {
            *c /= n;
        }

        let mut max_dist: f64 = 0.0;
        for i in start..end {
            let p = self.get_point(i);
            let dist = self.metric.distance(p, &centroid);

            if  dist > max_dist {
                max_dist = dist;
            }
        }
        (centroid, max_dist)
    }

    fn select_split_dim(&self, start: usize, end: usize) -> usize {
        let mut best_dim = 0;
        let mut best_spread = 0.0;
        
        for d in 0..self.dim {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            
            for i in start..end {
                let val = self.get_point_from_idx(i)[d];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
            
            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }
        best_dim
    }

    fn partition(&mut self, start: usize, end: usize, dim: usize) -> usize {
        let mut slots_by_key: Vec<(f64, usize)> = (start..end)
            .map(|slot| (self.get_point_from_idx(slot)[dim], slot))
            .collect();

        let mid_offset = (end - start) / 2;

        slots_by_key.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        let new_order: Vec<usize> = slots_by_key
            .iter()
            .map(|&(_key, slot)| self.indices[slot])
            .collect();

        self.indices[start..end].copy_from_slice(&new_order);

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

        let dim = self.select_split_dim(start, end);
        let mut mid = self.partition(start, end, dim);

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

    pub fn query_radius(&self, query: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_radius_recursive(0, query, radius, &mut results);
        results
    }

    pub fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<usize>) {
        let node = &self.nodes[node_idx];

        let dist_to_centre = self.metric.distance(query, &node.center);
        if dist_to_centre - node.radius > radius {
            return;
        }

        if node.left.is_none() {
            for i in node.start..node.end {
                let p = self.get_point(i);
                if self.metric.distance(query, p) <= radius {
                    results.push(self.indices[i]);
                }
            }
        }

        if let Some(left) = node.left {
            self.query_radius_recursive(left, query, radius, results);
        }

        if let Some(right) = node.right {
            self.query_radius_recursive(right, query, radius, results);
        }
    }

    pub fn query_knn(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.n_points == 0 {
            return Vec::new();
        }

        let mut heap = BinaryHeap::with_capacity(k);
        self.query_knn_recursive(0, query, &mut heap, k);
        heap.into_sorted_vec().into_iter().map(|item| (item.index, item.distance)).collect()
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize,) {
        let node = &self.nodes[node_idx];

        let dist_to_centre = self.metric.distance(query, &node.center);

        if heap.len() == k {
            if dist_to_centre - node.radius > heap.peek().unwrap().distance {
                return;
            }
        }

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.distance(query, self.get_point(i));
                
                if heap.len() < k {
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                } else if dist < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(HeapItem { distance: dist, index: self.indices[i] });
                }
            }
            return;
        }
        
        let left_idx = node.left.unwrap();
        let right_idx = node.right.unwrap();

        let left_dist = self.metric.distance(query, &self.nodes[left_idx].center);
        let right_dist = self.metric.distance(query, &self.nodes[right_idx].center);

        if left_dist <= right_dist {
            self.query_knn_recursive(left_idx, query, heap, k);
            self.query_knn_recursive(right_idx, query, heap, k);
        } else {
            self.query_knn_recursive(right_idx, query, heap, k);
            self.query_knn_recursive(left_idx, query, heap, k);
        }

    }

    pub fn kernel_density(&self, queries: &NdArray<f64>, bandwidth: f64, kernel: KernelType) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");

        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim, "Query dimension must match tree dimension");

        let mut results = vec![0.0; n_queries];

        for i in 0..n_queries {
            let query = &queries.as_slice()[i * dim..(i + 1) * dim];
            let mut density = 0.0;
            self.kde_recursive(0, query, bandwidth, &mut density, kernel, &ApproxCriterion::None);
            results[i] = density;
        }

        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }

    pub fn kernel_density_approx(&self, queries: &NdArray<f64>, bandwidth: f64, kernel: KernelType, criterion: &ApproxCriterion) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");

        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim, "Query dimension must match tree dimension");

        let mut results = vec![0.0; n_queries];

        for i in 0..n_queries {
            let query = &queries.as_slice()[i * dim..(i + 1) * dim];
            let mut density = 0.0;
            self.kde_recursive(0, query, bandwidth, &mut density, kernel, criterion);
            results[i] = density;
        }

        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }

    fn approx_kde_for_node(&self, query: &[f64], node: &BallNode, h: f64, kernel: KernelType) -> f64 {
        let n_points = (node.end - node.start) as f64;

        let dist_to_center = self.metric.distance(query, &node.center);

        let min_possible = (dist_to_center - node.radius).max(0.0);
        let max_possible = dist_to_center + node.radius;
        
        let k_min = kernel.evaluate(min_possible, h);
        let k_max = kernel.evaluate(max_possible, h);
        let k_avg = (k_min + k_max) / 2.0;
        
        n_points * k_avg
    }

    fn should_approximate(&self, node: &BallNode, criterion: &ApproxCriterion) -> bool {
        if node.left.is_none() {
            return false;
        }
        
        match criterion {
            ApproxCriterion::None => false,
            
            ApproxCriterion::MinSamples(threshold) => {
                let node_count = node.end - node.start;
                node_count >= *threshold
            },
            
            ApproxCriterion::MaxSpan(threshold) => {
                let span = 2.0 * node.radius;
                span <= *threshold
            },
            
            ApproxCriterion::Combined(min_samples, max_span) => {
                let node_count = node.end - node.start;
                let span = 2.0 * node.radius;
                node_count >= *min_samples && span <= *max_span
            },
        }
    }

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType, criterion: &ApproxCriterion) {
        let node = &self.nodes[node_idx];
        let dist_to_center = self.metric.distance(query, &node.center);

        let min_dist = (dist_to_center - node.radius).max(0.0);
        if kernel.evaluate(min_dist, h) < 1e-10 {
            return;
        }

        if self.should_approximate(node, criterion) {
            *density += self.approx_kde_for_node(query, node, h, kernel);
            return;
        }

        if node.left.is_none() {
            for i in node.start..node.end {
                let dist = self.metric.distance(query, self.get_point(i));
                *density += kernel.evaluate(dist, h);
            }
            return;
        }

        if let Some(left) = node.left {
            self.kde_recursive(left, query, h, density, kernel, &criterion);
        }
        if let Some(right) = node.right {
            self.kde_recursive(right, query, h, density, kernel, &criterion);
        }
    }
}