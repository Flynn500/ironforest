use std::{cmp::Ordering, collections::BinaryHeap};
use crate::{array::NdArray};

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

impl DistanceMetric {
    #[inline]
    pub fn distance(self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            DistanceMetric::Euclidean => euclidean(a, b),
            DistanceMetric::Manhattan => manhattan(a, b),
            DistanceMetric::Chebyshev => chebyshev(a, b),
        }
    }
}

#[inline]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[inline]
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean(a, b).sqrt()
}

#[inline]
fn manhattan(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[inline]
fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    let mut m: f64 = 0.0;
    for i in 0..a.len() {
        m = m.max((a[i] - b[i]).abs());
    }
    m
}




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

//for knn search
struct HeapItem {
        pub distance: f64,
        pub index: usize,
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
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
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric)
    }
    
    fn get_point(&self, i: usize) -> &[f64] {
        let idx = self.indices[i];
        &self.data[idx * self.dim..(idx + 1) * self.dim]
    }

    fn compute_bounding_ball(&self, start: usize, end: usize) -> (Vec<f64>, f64) {
        let n = (end - start) as f64;
        let mut centroid = vec![0.0; self.dim];

        for i in start..end {
            let p = self.get_point(i);
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
                let val = self.get_point(i)[d];
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
        let mut vals: Vec<(f64, usize)> = (start..end)
            .map(|i| (self.get_point(i)[dim], i))
            .collect();

        let mid_offset = (end - start) / 2;
        vals.select_nth_unstable_by(mid_offset, |a, b| a.0.partial_cmp(&b.0).unwrap());

        let new_order: Vec<usize> = vals.iter().map(|&(_, i)| self.indices[i]).collect();
        self.indices[start..end].copy_from_slice(&new_order);
        
        start + mid_offset
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        let (center, radius) = self.compute_bounding_ball(start, end);
        
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
}