use std::{cmp::Ordering, collections::BinaryHeap};
use crate::{array::{NdArray, Shape}, spatial::common::{DistanceMetric, KernelType, HeapItem, ApproxCriterion}};
use crate::random::Generator;
#[derive(Clone, Copy, Debug, Default)]
pub enum VantagePointSelection {
    #[default]
    First,
    Random,
}

impl VantagePointSelection{
    fn select_vantage(&self, start: usize, end: usize) -> usize {
        match self {
            VantagePointSelection::First => start,
            VantagePointSelection::Random => {
                let mut rng = Generator::new();
                let i = rng.randint(start as i64, end as i64, Shape::scalar());
                i.item() as usize
            },
        }
    }
}


#[derive(Debug, Clone)]
pub struct VPNode {
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>, 

    pub radius: f64,

    pub min_dist: f64,
    pub max_dist: f64,
}

#[derive(Debug, Clone)]
pub struct VPTree {
    pub nodes: Vec<VPNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub selection_method: VantagePointSelection,
}


impl VPTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric, selection_method: VantagePointSelection) -> Self{
        let mut tree = VPTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
            metric,
            selection_method,
        };

        tree.build_recursive(0, n_points);
        tree.reorder_data();
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric, selection_method: VantagePointSelection ) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric, selection_method)
    }

    fn get_point(&self, i: usize) -> &[f64] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn get_point_from_idx(&self, i: usize) -> &[f64] {
        let original_idx = self.indices[i];
        &self.data[original_idx * self.dim..(original_idx + 1) * self.dim]
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

    fn init_node(&mut self, start: usize, end: usize) -> (f64, f64, f64, usize) {
        let vantage_idx = self.selection_method.select_vantage(start, end); 
        self.indices.swap(start, vantage_idx);


        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        let vantage_point = self.get_point_from_idx(start).to_vec();

        let mut idx_dist: Vec<(usize, f64)> = (start + 1..end)
            .map(|i| {
                let p = self.get_point_from_idx(i);
                let dist = self.metric.distance(p, &vantage_point);
                min = min.min(dist);
                max = max.max(dist);
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

        (median_radius, min, max, start + 1 + mid)
    }

    fn min_dist_to_node(&self, query: &[f64], node: &VPNode) -> f64 {
        let vantage_point = self.get_point(node.start);
        let d_qv = self.metric.distance(query, vantage_point);

        let l1 = d_qv - node.max_dist;
        let l2 = node.min_dist - d_qv;

        l1.max(l2).max(0.0)
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (radius, min_dist, max_dist, mid) = self.init_node(start, end);
    
        let node_idx = self.nodes.len();

        self.nodes.push(VPNode { 
            start, 
            end, 
            left: None, 
            right: None, 
            radius, 
            min_dist, 
            max_dist, 
        });
        
        let count = end - start;

        if count <= self.leaf_size {
            return node_idx;
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

        let min_dist = self.min_dist_to_node(query, &node);
        if min_dist > radius {
            return;
        }

        if node.left.is_none() {
            for i in node.start..node.end {
                let p = self.get_point(i);
                if self.metric.distance(query, p) <= radius {
                    results.push(self.indices[i]);
                }
            }
            return;
        }

        if let Some(left) = node.left {
            self.query_radius_recursive(left, query, radius, results);
        }

        if let Some(right) = node.right {
            self.query_radius_recursive(right, query, radius, results);
        }
    }

        pub fn query_knn(&self, query: &[f64], k: usize) -> Vec<usize> {
        if k == 0 || self.n_points == 0 {
            return Vec::new();
        }
        
        let mut heap = BinaryHeap::with_capacity(k);
        self.query_knn_recursive(0, query, &mut heap, k);
        heap.into_sorted_vec().into_iter().map(|item| item.index).collect()
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize,) {
        let node = &self.nodes[node_idx];

        let min_dist = self.min_dist_to_node(query, &node);

        if heap.len() == k {
            if min_dist > heap.peek().unwrap().distance {
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

        let vantage_point = self.get_point(node.start);
        let dist_to_vantage = self.metric.distance(query, vantage_point);

        let (first, second) = if dist_to_vantage < node.radius {
            (left_idx, right_idx)
        } else {
            (right_idx, left_idx) 
        };

        self.query_knn_recursive(first, query, heap, k);
        self.query_knn_recursive(second, query, heap, k);
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

    pub fn kernel_density_approx(&self, queries: &NdArray<f64>, bandwidth: f64, kernel: KernelType, criterion: ApproxCriterion) -> NdArray<f64> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");

        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim, "Query dimension must match tree dimension");
        
        let mut results = vec![0.0; n_queries];
        
        for i in 0..n_queries {
            let query = &queries.as_slice()[i * dim..(i + 1) * dim];
            let mut density = 0.0;
            self.kde_recursive(0, query, bandwidth, &mut density, kernel, &criterion);
            results[i] = density;
        }
        
        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }

    fn approx_kde_for_node(&self, query: &[f64], node: &VPNode, h: f64, kernel: KernelType) -> f64 {
        let vantage_point = self.get_point(node.start);
        let dist_to_vantage = self.metric.distance(query, vantage_point);
        
        let n_points = (node.end - node.start) as f64;
        
        let min_possible = (dist_to_vantage - node.max_dist).max(0.0);
        let max_possible = dist_to_vantage + node.max_dist;



        let k_min = kernel.evaluate(min_possible, h);
        let k_max = kernel.evaluate(max_possible, h);
        let k_avg = (k_min + k_max) / 2.0;
        
        n_points * k_avg
    }

    fn should_approximate(&self, node: &VPNode, criterion: &ApproxCriterion) -> bool {
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
                let span = node.max_dist - node.min_dist;
                span <= *threshold
            },
            
            ApproxCriterion::Combined(min_samples, max_radius) => {
                let node_count = node.end - node.start;
                let span = node.max_dist - node.min_dist;
                node_count >= *min_samples && span <= *max_radius
            },
        }
    }

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType, criterion: &ApproxCriterion) {
        let node = &self.nodes[node_idx];
        let min_dist = self.min_dist_to_node(query, &node);

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