use crate::{KernelType, array::NdArray, spatial::{HeapItem, common::DistanceMetric}};
use super::spatial_query::{SpatialQuery};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingEntry {
    pub object: Vec<f64>,
    pub covering_radius: f64,
    pub dist_to_parent: f64,
    pub child_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafEntry {
    pub object: Vec<f64>,
    pub dist_to_parent: f64,
    pub point_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MNode {
    Internal {
        parent_dist: f64,
        count: usize,
        entries: Vec<RoutingEntry>,
    },
    Leaf {
        parent_dist: f64,
        count: usize,
        entries: Vec<LeafEntry>,
    },
}

impl MNode{
    pub fn count(&self) -> usize {
        match self {
            MNode::Internal { count, .. } => *count,
            MNode::Leaf { count, .. } => *count,
        }
    }
}


//MTree does not use our NdArray internally. Internal structure differs enough were
//we just used vectors and overwrite query methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTree {
    pub nodes: Vec<MNode>,
    pub root: usize,
    pub dim: usize,
    pub n_points: usize,
    pub capacity: usize,
    pub metric: DistanceMetric,
}



impl MTree {
    pub fn new(dim: usize, capacity: usize, metric: DistanceMetric) -> Self {
        let root = MNode::Leaf {
            parent_dist: 0.0,
            count: 0,
            entries: Vec::new(),
        };
        MTree {
            nodes: vec![root],
            root: 0,
            dim,
            n_points: 0,
            capacity,
            metric,
        }
    }

    pub fn from_ndarray(array: &NdArray<f64>, capacity: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");

        let dim = shape[1];
        
        let mut tree = Self::new(dim, capacity, metric);

        for (i, point) in array.as_slice().chunks(dim).enumerate() {
            let transformed = metric.pre_transform(point);
            tree.insert(transformed.into_owned(), i);
        }
        tree
    }

    pub fn collect_data(&self) -> Vec<f64> {
        let mut data = vec![0.0f64; self.n_points * self.dim];
        for node in &self.nodes {
            if let MNode::Leaf { entries, .. } = node {
                for entry in entries {
                    let offset = entry.point_idx * self.dim;
                    data[offset..offset + self.dim].copy_from_slice(&entry.object);
                }
            }
        }
        data
    }

    fn insert_at(&mut self, node_idx: usize, point: Vec<f64>, point_idx: usize, dist_to_parent: f64) -> Option<(RoutingEntry, RoutingEntry)> {
        match &self.nodes[node_idx] {
            MNode::Leaf { .. } => {
                self.insert_into_leaf(node_idx, point, point_idx, dist_to_parent)
            }
            MNode::Internal { .. } => {
                 self.insert_into_internal(node_idx, point, point_idx)
            }
        }
    }

    pub fn insert(&mut self, point: Vec<f64>, point_idx: usize) {
        let point = self.metric.pre_transform(&point).into_owned();
        if let Some((mut left, mut right)) = self.insert_at(self.root, point, point_idx, 0.0) {
            left.dist_to_parent = 0.0;
            right.dist_to_parent = 0.0;
            
            let count = self.nodes[left.child_idx].count() + self.nodes[right.child_idx].count();

            let new_root = MNode::Internal {
                parent_dist: 0.0,
                count: count,
                entries: vec![left, right],
            };

            let new_root_idx = self.nodes.len();
            self.nodes.push(new_root);
            self.root = new_root_idx;
        }
        self.n_points += 1;
    }

    fn insert_into_leaf(&mut self, node_idx: usize, point: Vec<f64>, point_idx: usize, dist_to_parent: f64) -> Option<(RoutingEntry, RoutingEntry)> {
        let MNode::Leaf { 
            entries, 
            count, .. 
        } = &mut self.nodes[node_idx] else { unreachable!() };

        entries.push(LeafEntry { 
            object: point, 
            dist_to_parent, 
            point_idx 
        });
        
        *count += 1;

        if entries.len() > self.capacity {
            Some(self.split_leaf(node_idx))
        } else {
            None
        }
    }

    fn pick_best_child(&self, node_idx: usize, point: &[f64]) -> usize {
        let MNode::Internal { entries, .. } = &self.nodes[node_idx] else { unreachable!() };

        let mut best_idx = 0;
        let mut best_cost = f64::MAX;

        for (i, entry) in entries.iter().enumerate() {
            let d = self.metric.reduced_distance(point, &entry.object);
            let cost = if d <= entry.covering_radius {
                d
            } else {
                d - entry.covering_radius
            };

            if cost < best_cost {
                best_cost = cost;
                best_idx = i;
            }
        }

        best_idx
    }

    fn insert_into_internal(&mut self, node_idx: usize, point: Vec<f64>, point_idx: usize) -> Option<(RoutingEntry, RoutingEntry)> {
        let best_entry_idx = self.pick_best_child(node_idx, &point);

        let MNode::Internal { entries, .. } = &self.nodes[node_idx] else { unreachable!() };
        let child_idx = entries[best_entry_idx].child_idx;
        let parent_obj = entries[best_entry_idx].object.clone();
        let dist_to_child = self.metric.reduced_distance(&point, &parent_obj);

        let split = self.insert_at(child_idx, point, point_idx, dist_to_child);

        let MNode::Internal { entries, count, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        *count += 1;
        entries[best_entry_idx].covering_radius = entries[best_entry_idx].covering_radius.max(dist_to_child);

        if let Some((mut left, mut right)) = split {
            left.dist_to_parent = self.metric.reduced_distance(&left.object, &parent_obj);
            right.dist_to_parent = self.metric.reduced_distance(&right.object, &parent_obj);

            entries[best_entry_idx] = left;
            entries.push(right);

            if entries.len() > self.capacity {
                return Some(self.split_internal(node_idx));
            }
        }

        None
    }

    fn promote(&self, objects: &[Vec<f64>]) -> (usize, usize) {
        let mut best = (0, 1);
        let mut best_dist = 0.0;

        for i in 0..objects.len() {
            for j in i+1..objects.len() {
                let d = self.metric.reduced_distance(&objects[i], &objects[j]);
                if d > best_dist {
                    best_dist = d;
                    best = (i, j);
                }
            }
        }
        best
    }

    fn split_leaf(&mut self, node_idx: usize) -> (RoutingEntry, RoutingEntry) {
        let MNode::Leaf { entries, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        let entries = std::mem::take(entries);

        let (p1_idx, p2_idx) = self.promote(&entries.iter().map(|e| e.object.clone()).collect::<Vec<_>>());

        let p1 = entries[p1_idx].object.clone();
        let p2 = entries[p2_idx].object.clone();

        let mut left_entries: Vec<LeafEntry> = Vec::new();
        let mut right_entries: Vec<LeafEntry> = Vec::new();

        for entry in entries {
            let d1 = self.metric.reduced_distance(&entry.object, &p1);
            let d2 = self.metric.reduced_distance(&entry.object, &p2);
            if d1 <= d2 {
                left_entries.push(LeafEntry { dist_to_parent: d1, ..entry });
            } else {
                right_entries.push(LeafEntry { dist_to_parent: d2, ..entry });
            }
        }

        let left_radius = left_entries.iter().map(|e| e.dist_to_parent).fold(0.0_f64, f64::max);
        let right_radius = right_entries.iter().map(|e| e.dist_to_parent).fold(0.0_f64, f64::max);

        let left_count = left_entries.len();
        let right_count = right_entries.len();

        self.nodes[node_idx] = MNode::Leaf { 
            parent_dist: 0.0,
            count: left_count,
            entries: 
            left_entries };
        

        self.nodes.push(MNode::Leaf { 
            parent_dist: 0.0, 
            count: right_count,
            entries: right_entries 
        });

        let right_idx = self.nodes.len() - 1;

        let left_entry = RoutingEntry { object: p1, covering_radius: left_radius, dist_to_parent: 0.0, child_idx: node_idx };
        let right_entry = RoutingEntry { object: p2, covering_radius: right_radius, dist_to_parent: 0.0, child_idx: right_idx };

        (left_entry, right_entry)
    }

    fn split_internal(&mut self, node_idx: usize) -> (RoutingEntry, RoutingEntry) {
        let MNode::Internal { entries, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        let entries = std::mem::take(entries);

        let objects: Vec<Vec<f64>> = entries.iter().map(|e| e.object.clone()).collect();
        let (p1_idx, p2_idx) = self.promote(&objects);

        let p1 = entries[p1_idx].object.clone();
        let p2 = entries[p2_idx].object.clone();

        let mut left_entries: Vec<RoutingEntry> = Vec::new();
        let mut right_entries: Vec<RoutingEntry> = Vec::new();

        for mut entry in entries {
            let d1 = self.metric.reduced_distance(&entry.object, &p1);
            let d2 = self.metric.reduced_distance(&entry.object, &p2);
            if d1 <= d2 {
                entry.dist_to_parent = d1;
                left_entries.push(entry);
            } else {
                entry.dist_to_parent = d2;
                right_entries.push(entry);
            }
        }

        let left_radius = left_entries.iter().map(|e| e.dist_to_parent + e.covering_radius).fold(0.0_f64, f64::max);
        let right_radius = right_entries.iter().map(|e| e.dist_to_parent + e.covering_radius).fold(0.0_f64, f64::max);

        let left_count: usize = left_entries.iter().map(|e| self.nodes[e.child_idx].count()).sum();
        let right_count: usize = right_entries.iter().map(|e| self.nodes[e.child_idx].count()).sum();

        self.nodes[node_idx] = MNode::Internal { 
            parent_dist: 0.0, 
            count: left_count,
            entries: 
            left_entries 
        };
        
        self.nodes.push(MNode::Internal { 
            parent_dist: 0.0, 
            count: right_count,
            entries: right_entries 
        });

        let right_idx = self.nodes.len() - 1;
        let left_entry = RoutingEntry { object: p1, covering_radius: left_radius, dist_to_parent: 0.0, child_idx: node_idx };
        let right_entry = RoutingEntry { object: p2, covering_radius: right_radius, dist_to_parent: 0.0, child_idx: right_idx };

        (left_entry, right_entry)
    }

    //Have to recreate query methods manually as M-tree differs substantially from the others
    fn knn_recursive_inner(
        &self,
        node_idx: usize,
        query: &[f64],
        d_query_parent: f64,
        heap: &mut BinaryHeap<HeapItem>,
        k: usize,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };
                    let best = heap.peek().map(|h| h.distance).unwrap_or(f64::MAX);
                    if heap.len() == k && lb > best {
                        continue;
                    }

                    let dist = self.metric.reduced_distance(query, &entry.object);
                    if heap.len() < k {
                        heap.push(HeapItem { distance: dist, index: entry.point_idx });
                    } else if dist < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(HeapItem { distance: dist, index: entry.point_idx });
                    }
                }
            }
            MNode::Internal { entries, .. } => {
                let mut children: Vec<(f64, f64, usize)> = Vec::with_capacity(entries.len());

                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };
                    let best = heap.peek().map(|h| h.distance).unwrap_or(f64::MAX);
                    if heap.len() == k && (lb - entry.covering_radius) > best {
                        continue;
                    }

                    let d = self.metric.reduced_distance(query, &entry.object);
                    let min_dist = (d - entry.covering_radius).max(0.0);

                    let best = heap.peek().map(|h| h.distance).unwrap_or(f64::MAX);
                    if heap.len() == k && min_dist > best {
                        continue;
                    }

                    children.push((min_dist, d, entry.child_idx));
                }

                children.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for (min_dist, d_to_routing, child_idx) in children {
                    let best = heap.peek().map(|h| h.distance).unwrap_or(f64::MAX);
                    if heap.len() == k && min_dist > best {
                        break;
                    }

                    self.knn_recursive_inner(child_idx, query, d_to_routing, heap, k);
                }
            }
        }
    }

    fn radius_recursive_inner(
        &self,
        node_idx: usize,
        query: &[f64],
        d_query_parent: f64,
        radius: f64,
        results: &mut Vec<(usize, f64)>,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };
                    if lb > radius {
                        continue;
                    }

                    let dist = self.metric.reduced_distance(query, &entry.object);
                    if dist <= radius {
                        results.push((entry.point_idx, dist));
                    }
                }
            }
            MNode::Internal { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };

                    if (lb - entry.covering_radius) > radius {
                        continue;
                    }

                    let d = self.metric.reduced_distance(query, &entry.object);
                    if (d - entry.covering_radius).max(0.0) > radius {
                        continue;
                    }

                    self.radius_recursive_inner(entry.child_idx, query, d, radius, results);
                }
            }
        }
    }

    fn kde_recursive_inner(
        &self,
        node_idx: usize,
        query: &[f64],
        d_query_parent: f64,
        h: f64,
        density: &mut f64,
        kernel: KernelType,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };
                    
                    //single point so don't need to multiply by n
                    let transformed_lb = self.metric.post_transform(lb);
                    if kernel.evaluate(transformed_lb, h) < 1e-10 {
                        continue;
                    }

                    let transformed_dist = self.metric.post_transform(self.metric.reduced_distance(query, &entry.object));
                    *density += kernel.evaluate(transformed_dist, h);
                }
            }
            MNode::Internal { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        0.0
                    };
                    let min_dist_lb = (lb - entry.covering_radius).max(0.0);
                    let n = self.nodes[entry.child_idx].count() as f64;
                    
                    let transformed_dist_lb = self.metric.post_transform(min_dist_lb);
                    if kernel.evaluate(transformed_dist_lb, h) * n < 1e-10 {
                        continue;
                    }

                    let d = self.metric.reduced_distance(query, &entry.object);
                    let min_dist = (d - entry.covering_radius).max(0.0);
                    
                    let transformed_dist = self.metric.post_transform(min_dist);
                    if kernel.evaluate(transformed_dist, h) * n < 1e-10 {
                        continue;
                    }

                    self.kde_recursive_inner(entry.child_idx, query, d, h, density, kernel);
                }
            }
        }
    }
}

impl SpatialQuery for MTree {
    type Node = MNode;

    //most stuff is overriden so most helpers are uneeded
    fn nodes(&self) -> &[MNode] { &self.nodes }
    fn indices(&self) -> &[usize] { &[] }
    fn data(&self) -> &[f64] { &[] }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize {self.n_points}

    fn root(&self) -> usize { self.root }

    fn node_start(&self, _idx: usize) -> usize { 0 }
    fn node_end(&self, _idx: usize) -> usize { 0 }
    fn node_left(&self, idx: usize) -> Option<usize> {
        match &self.nodes[idx] {
            MNode::Internal { entries, .. } => entries.first().map(|e| e.child_idx),
            MNode::Leaf { .. } => None,
        }
    }
    fn node_right(&self, idx: usize) -> Option<usize> {
        match &self.nodes[idx] {
            MNode::Internal { entries, .. } => entries.last().map(|e| e.child_idx),
            MNode::Leaf { .. } => None,
        }
    }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        match &self.nodes[node_idx] {
            MNode::Internal { entries, .. } => {
                entries.iter().map(|e| {
                    let d = self.metric.reduced_distance(query, &e.object);
                    (d - e.covering_radius).max(0.0)
                }).fold(f64::MAX, f64::min)
            }
            MNode::Leaf { entries, .. } => {
                entries.iter().map(|e| self.metric.reduced_distance(query, &e.object))
                    .fold(f64::MAX, f64::min)
            }
        }
    }

    fn query_knn_recursive(&self, node_idx: usize, query: &[f64], heap: &mut BinaryHeap<HeapItem>, k: usize) {
        self.knn_recursive_inner(node_idx, query, f64::INFINITY, heap, k);
    }

    fn query_radius_recursive(&self, node_idx: usize, query: &[f64], radius: f64, results: &mut Vec<(usize, f64)>) {
        self.radius_recursive_inner(node_idx, query, f64::INFINITY, radius, results);
    }

    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType) {
        self.kde_recursive_inner(node_idx, query, f64::INFINITY, h, density, kernel);
    }
}