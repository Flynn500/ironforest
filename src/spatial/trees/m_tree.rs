use crate::{KernelType, array::NdArray, spatial::{HeapItem, common::{DistanceMetric, IronFloat}}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct RoutingEntry<T> {
    pub object: Vec<T>,
    pub covering_radius: T,
    pub dist_to_parent: T,
    pub child_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct LeafEntry<T> {
    pub object: Vec<T>,
    pub dist_to_parent: T,
    pub point_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub enum MNode<T> {
    Internal {
        parent_dist: T,
        count: usize,
        entries: Vec<RoutingEntry<T>>,
    },
    Leaf {
        parent_dist: T,
        count: usize,
        entries: Vec<LeafEntry<T>>,
    },
}

impl<T: IronFloat> MNode<T> {
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
#[serde(bound = "T: IronFloat")]
pub struct MTree<T: IronFloat> {
    pub nodes: Vec<MNode<T>>,
    pub root: usize,
    pub dim: usize,
    pub n_points: usize,
    pub capacity: usize,
    pub metric: DistanceMetric,
}



impl<T: IronFloat> MTree<T> {
    pub fn new(dim: usize, capacity: usize, metric: DistanceMetric) -> Self {
        let root = MNode::Leaf {
            parent_dist: T::zero(),
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

    pub fn from_ndarray(array: &NdArray<T>, capacity: usize, metric: DistanceMetric) -> Self {
        let shape = array.shape().dims();

        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");

        let dim = shape[1];

        let mut tree = Self::new(dim, capacity, metric);

        let array_contig;
        let array_slice = if array.is_contiguous() {
            array.as_slice().unwrap()
        } else {
            array_contig = array.to_contiguous();
            array_contig.as_slice().unwrap()
        };
        for (i, point) in array_slice.chunks(dim).enumerate() {
            let transformed = metric.pre_transform(point);
            tree.insert(transformed.into_owned(), i);
        }
        tree
    }

    pub fn collect_data(&self) -> Vec<T> {
        let mut data = vec![T::zero(); self.n_points * self.dim];
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

    fn insert_at(&mut self, node_idx: usize, point: Vec<T>, point_idx: usize, dist_to_parent: T) -> Option<(RoutingEntry<T>, RoutingEntry<T>)> {
        match &self.nodes[node_idx] {
            MNode::Leaf { .. } => {
                self.insert_into_leaf(node_idx, point, point_idx, dist_to_parent)
            }
            MNode::Internal { .. } => {
                 self.insert_into_internal(node_idx, point, point_idx)
            }
        }
    }

    pub fn insert(&mut self, point: Vec<T>, point_idx: usize) {
        let point = self.metric.pre_transform(&point).into_owned();
        if let Some((mut left, mut right)) = self.insert_at(self.root, point, point_idx, T::zero()) {
            left.dist_to_parent = T::zero();
            right.dist_to_parent = T::zero();

            let count = self.nodes[left.child_idx].count() + self.nodes[right.child_idx].count();

            let new_root = MNode::Internal {
                parent_dist: T::zero(),
                count,
                entries: vec![left, right],
            };

            let new_root_idx = self.nodes.len();
            self.nodes.push(new_root);
            self.root = new_root_idx;
        }
        self.n_points += 1;
    }

    fn insert_into_leaf(&mut self, node_idx: usize, point: Vec<T>, point_idx: usize, dist_to_parent: T) -> Option<(RoutingEntry<T>, RoutingEntry<T>)> {
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

    fn pick_best_child(&self, node_idx: usize, point: &[T]) -> usize {
        let MNode::Internal { entries, .. } = &self.nodes[node_idx] else { unreachable!() };

        let mut best_idx = 0;
        let mut best_cost = T::max_value();

        for (i, entry) in entries.iter().enumerate() {
            let d = self.metric.distance(point, &entry.object);
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

    fn insert_into_internal(&mut self, node_idx: usize, point: Vec<T>, point_idx: usize) -> Option<(RoutingEntry<T>, RoutingEntry<T>)> {
        let best_entry_idx = self.pick_best_child(node_idx, &point);

        let MNode::Internal { entries, .. } = &self.nodes[node_idx] else { unreachable!() };
        let child_idx = entries[best_entry_idx].child_idx;
        let parent_obj = entries[best_entry_idx].object.clone();
        let dist_to_child = self.metric.distance(&point, &parent_obj);

        let split = self.insert_at(child_idx, point, point_idx, dist_to_child);

        let MNode::Internal { entries, count, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        *count += 1;
        entries[best_entry_idx].covering_radius = entries[best_entry_idx].covering_radius.max(dist_to_child);

        if let Some((mut left, mut right)) = split {
            left.dist_to_parent = self.metric.distance(&left.object, &parent_obj);
            right.dist_to_parent = self.metric.distance(&right.object, &parent_obj);

            entries[best_entry_idx] = left;
            entries.push(right);

            if entries.len() > self.capacity {
                return Some(self.split_internal(node_idx));
            }
        }

        None
    }

    fn promote(&self, objects: &[Vec<T>]) -> (usize, usize) {
        let mut best = (0, 1);
        let mut best_dist = T::zero();

        for i in 0..objects.len() {
            for j in i+1..objects.len() {
                let d = self.metric.distance(&objects[i], &objects[j]);
                if d > best_dist {
                    best_dist = d;
                    best = (i, j);
                }
            }
        }
        best
    }

    fn split_leaf(&mut self, node_idx: usize) -> (RoutingEntry<T>, RoutingEntry<T>) {
        let MNode::Leaf { entries, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        let entries = std::mem::take(entries);

        let (p1_idx, p2_idx) = self.promote(&entries.iter().map(|e| e.object.clone()).collect::<Vec<_>>());

        let p1 = entries[p1_idx].object.clone();
        let p2 = entries[p2_idx].object.clone();

        let mut left_entries: Vec<LeafEntry<T>> = Vec::new();
        let mut right_entries: Vec<LeafEntry<T>> = Vec::new();

        for entry in entries {
            let d1 = self.metric.distance(&entry.object, &p1);
            let d2 = self.metric.distance(&entry.object, &p2);
            if d1 <= d2 {
                left_entries.push(LeafEntry { dist_to_parent: d1, ..entry });
            } else {
                right_entries.push(LeafEntry { dist_to_parent: d2, ..entry });
            }
        }

        let left_radius = left_entries.iter().map(|e| e.dist_to_parent).fold(T::zero(), |a, b| a.max(b));
        let right_radius = right_entries.iter().map(|e| e.dist_to_parent).fold(T::zero(), |a, b| a.max(b));

        let left_count = left_entries.len();
        let right_count = right_entries.len();

        self.nodes[node_idx] = MNode::Leaf {
            parent_dist: T::zero(),
            count: left_count,
            entries: left_entries,
        };


        self.nodes.push(MNode::Leaf {
            parent_dist: T::zero(),
            count: right_count,
            entries: right_entries
        });

        let right_idx = self.nodes.len() - 1;

        let left_entry = RoutingEntry { object: p1, covering_radius: left_radius, dist_to_parent: T::zero(), child_idx: node_idx };
        let right_entry = RoutingEntry { object: p2, covering_radius: right_radius, dist_to_parent: T::zero(), child_idx: right_idx };

        (left_entry, right_entry)
    }

    fn split_internal(&mut self, node_idx: usize) -> (RoutingEntry<T>, RoutingEntry<T>) {
        let MNode::Internal { entries, .. } = &mut self.nodes[node_idx] else { unreachable!() };
        let entries = std::mem::take(entries);

        let objects: Vec<Vec<T>> = entries.iter().map(|e| e.object.clone()).collect();
        let (p1_idx, p2_idx) = self.promote(&objects);

        let p1 = entries[p1_idx].object.clone();
        let p2 = entries[p2_idx].object.clone();

        let mut left_entries: Vec<RoutingEntry<T>> = Vec::new();
        let mut right_entries: Vec<RoutingEntry<T>> = Vec::new();

        for mut entry in entries {
            let d1 = self.metric.distance(&entry.object, &p1);
            let d2 = self.metric.distance(&entry.object, &p2);
            if d1 <= d2 {
                entry.dist_to_parent = d1;
                left_entries.push(entry);
            } else {
                entry.dist_to_parent = d2;
                right_entries.push(entry);
            }
        }

        let left_radius = left_entries.iter().map(|e| e.dist_to_parent + e.covering_radius).fold(T::zero(), |a, b| a.max(b));
        let right_radius = right_entries.iter().map(|e| e.dist_to_parent + e.covering_radius).fold(T::zero(), |a, b| a.max(b));

        let left_count: usize = left_entries.iter().map(|e| self.nodes[e.child_idx].count()).sum();
        let right_count: usize = right_entries.iter().map(|e| self.nodes[e.child_idx].count()).sum();

        self.nodes[node_idx] = MNode::Internal {
            parent_dist: T::zero(),
            count: left_count,
            entries: left_entries
        };

        self.nodes.push(MNode::Internal {
            parent_dist: T::zero(),
            count: right_count,
            entries: right_entries
        });

        let right_idx = self.nodes.len() - 1;
        let left_entry = RoutingEntry { object: p1, covering_radius: left_radius, dist_to_parent: T::zero(), child_idx: node_idx };
        let right_entry = RoutingEntry { object: p2, covering_radius: right_radius, dist_to_parent: T::zero(), child_idx: right_idx };

        (left_entry, right_entry)
    }

    //Have to recreate query methods manually as M-tree differs substantially from the others
    fn knn_recursive_inner(
        &self,
        node_idx: usize,
        query: &[T],
        d_query_parent: T,
        heap: &mut BinaryHeap<HeapItem<T>>,
        k: usize,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        T::zero()
                    };
                    let best = heap.peek().map(|h| h.distance).unwrap_or(T::infinity());
                    if heap.len() == k && lb > best {
                        continue;
                    }

                    let dist = self.metric.distance(query, &entry.object);
                    if heap.len() < k {
                        heap.push(HeapItem { distance: dist, index: entry.point_idx });
                    } else if dist < heap.peek().unwrap().distance {
                        heap.pop();
                        heap.push(HeapItem { distance: dist, index: entry.point_idx });
                    }
                }
            }
            MNode::Internal { entries, .. } => {
                let mut children: Vec<(T, T, usize)> = Vec::with_capacity(entries.len());

                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        T::zero()
                    };
                    let lb_child = (lb - entry.covering_radius).max(T::zero());
                    let best = heap.peek().map(|h| h.distance).unwrap_or(T::infinity());
                    if heap.len() == k && lb_child > best {
                        continue;
                    }

                    let d_real = self.metric.distance(query, &entry.object);
                    let min_dist_real = (d_real - entry.covering_radius).max(T::zero());

                    let best = heap.peek().map(|h| h.distance).unwrap_or(T::infinity());
                    if heap.len() == k && min_dist_real > best {
                        continue;
                    }

                    children.push((min_dist_real, d_real, entry.child_idx));
                }

                children.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                for (min_dist_reduced, d_to_routing_real, child_idx) in children {
                    let best = heap.peek().map(|h| h.distance).unwrap_or(T::infinity());
                    if heap.len() == k && min_dist_reduced > best {
                        break;
                    }
                    self.knn_recursive_inner(child_idx, query, d_to_routing_real, heap, k);
                }
            }
        }
    }

    fn radius_recursive_inner(
        &self,
        node_idx: usize,
        query: &[T],
        d_query_parent: T,
        radius: T,
        results: &mut Vec<(usize, T)>,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs()
                    } else {
                        T::zero()
                    };
                    if lb > radius {
                        continue;
                    }

                    let dist = self.metric.distance(query, &entry.object);
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
                        T::zero()
                    };

                    let lb_child = (lb - entry.covering_radius).max(T::zero());
                    if lb_child > radius {
                        continue;
                    }

                    let d_real = self.metric.distance(query, &entry.object);
                    let min_dist_real = (d_real - entry.covering_radius).max(T::zero());
                    if min_dist_real > radius {
                        continue;
                    }

                    self.radius_recursive_inner(entry.child_idx, query, d_real, radius, results);
                }
            }
        }
    }

    fn kde_recursive_inner(
        &self,
        node_idx: usize,
        query: &[T],
        d_query_parent: T,
        h: f64,
        density: &mut f64,
        kernel: KernelType,
    ) {
        match &self.nodes[node_idx] {
            MNode::Leaf { entries, .. } => {
                for entry in entries {
                    let lb: f64 = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs().to_f64().unwrap()
                    } else {
                        0.0
                    };

                    //single point so don't need to multiply by n
                    if kernel.evaluate(lb, h) < 1e-10 {
                        continue;
                    }

                    let dist: f64 = self.metric.distance(query, &entry.object).to_f64().unwrap();
                    *density += kernel.evaluate(dist, h);
                }
            }
            MNode::Internal { entries, .. } => {
                for entry in entries {
                    let lb: f64 = if d_query_parent.is_finite() {
                        (d_query_parent - entry.dist_to_parent).abs().to_f64().unwrap()
                    } else {
                        0.0
                    };
                    let covering_r_f64 = entry.covering_radius.to_f64().unwrap();
                    let min_dist_lb = (lb - covering_r_f64).max(0.0);
                    let n = self.nodes[entry.child_idx].count() as f64;

                    if kernel.evaluate(min_dist_lb, h) * n < 1e-10 {
                        continue;
                    }

                    let d_real = self.metric.distance(query, &entry.object);
                    let min_dist = (d_real.to_f64().unwrap() - covering_r_f64).max(0.0);

                    if kernel.evaluate(min_dist, h) * n < 1e-10 {
                        continue;
                    }

                    self.kde_recursive_inner(entry.child_idx, query, d_real, h, density, kernel);
                }
            }
        }
    }
}

impl<T: IronFloat> SpatialTree for MTree<T> {
    type Node = MNode<T>;
    type Float = T;
    const REDUCED: bool = false;
    //most stuff is overridden so most helpers are unneeded
    fn nodes(&self) -> &[MNode<T>] { &self.nodes }
    fn indices(&self) -> &[usize] { &[] }
    fn data(&self) -> &[T] { &[] }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize { self.n_points }
    // MTree stores points inside node entries rather than a flat data buffer,
    // so `get_point` is never called — this value is irrelevant.
    fn data_is_reordered(&self) -> bool { true }

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

    fn child_lower_bound(&self, child_idx: usize, query: &[Self::Float]) -> Self::Float {
        match &self.nodes[child_idx] {
            MNode::Internal { entries, .. } => {
                entries.iter().map(|e| {
                    let d = self.metric.distance(query, &e.object);
                    (d - e.covering_radius).max(T::zero())
                }).fold(T::infinity(), |a, b| a.min(b))
            }
            MNode::Leaf { entries, .. } => {
                entries.iter().map(|e| self.metric.distance(query, &e.object))
                    .fold(T::infinity(), |a, b| a.min(b))
            }
        }
    }
}


impl<T: IronFloat> KnnQuery for MTree<T> {
    fn query_knn_recursive(&self, node_idx: usize, query: &[T], heap: &mut BinaryHeap<HeapItem<T>>, k: usize) {
        self.knn_recursive_inner(node_idx, query, T::infinity(), heap, k);
    }
}

impl<T: IronFloat> RadiusQuery for MTree<T> {
    fn query_radius_recursive(&self, node_idx: usize, query: &[T], radius: T, results: &mut Vec<(usize, T)>) {
        self.radius_recursive_inner(node_idx, query, T::infinity(), radius, results);
    }
}

impl<T: IronFloat> KdeQuery for MTree<T> {
    fn kde_recursive(&self, node_idx: usize, query: &[T], h: f64, density: &mut f64, kernel: KernelType) {
        self.kde_recursive_inner(node_idx, query, T::infinity(), h, density, kernel);
    }
}
