use crate::{KernelType, Shape, array::NdArray, spatial::common::DistanceMetric};


#[derive(Clone, Debug)]
pub struct AggNode {
    pub center: Vec<f64>,
    pub radius: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}


#[derive(Debug, Clone)]
pub struct AggTree {
    pub nodes: Vec<AggNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub min_samples: usize,
    pub max_span: f64,
}

impl AggTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize, metric: DistanceMetric, min_samples: usize, max_span: f64) -> Self{
        let mut tree = AggTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
            metric,
            min_samples,
            max_span
        };

        tree.build_recursive(0, n_points);
        
        tree.reorder_data();
        
        let mut live = Vec::new();
        tree.collect_live_ranges(0, &mut live);
        let remap = tree.compact_data(&live);
        tree.remap_nodes(&remap);
        
        tree
    }

    pub fn from_ndarray(array: &NdArray<f64>, leaf_size: usize, metric: DistanceMetric, min_samples: usize, max_span: f64) -> Self {
        let shape = array.shape().dims();
        
        assert!(shape.len() == 2, "Expected 2D array (n_points, dim)");
        
        let n_points = shape[0];
        let dim = shape[1];
        
        Self::new(array.as_slice(), n_points, dim, leaf_size, metric, min_samples, max_span)
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

    fn collect_live_ranges(&self, node_idx: usize, live: &mut Vec<(usize, usize)>) {
        let node = &self.nodes[node_idx];

        if self.should_approximate(node) {
            return;
        }

        match (node.left, node.right) {
            (None, _) => live.push((node.start, node.end)),
            (Some(left), Some(right)) => {
                self.collect_live_ranges(left, live);
                self.collect_live_ranges(right, live);
            }
            _ => unreachable!()
        }
    }

    fn compact_data(&mut self, live: &[(usize, usize)]) -> Vec<usize> {
        let mut new_data = Vec::new();
        let mut remap = vec![usize::MAX; self.n_points];
        let mut new_idx = 0;

        for &(start, end) in live {
            for i in start..end {
                let offset = i * self.dim;
                new_data.extend_from_slice(&self.data[offset..offset + self.dim]);
                remap[i] = new_idx;
                new_idx += 1;
            }
        }

        self.data = new_data;
        remap
    }

    fn remap_nodes(&mut self, remap: &[usize]) {
        let max_span = self.max_span;
        let min_samples = self.min_samples;

        for node in &mut self.nodes {
            let is_live_leaf = node.left.is_none()
                && !((node.end - node.start) >= min_samples && 2.0 * node.radius <= max_span);

            if is_live_leaf {
                node.start = remap[node.start];
                node.end = remap[node.end - 1] + 1;
            }
        }
    }

    fn get_point_from_idx(&self, i: usize) -> &[f64] {
        let original_idx = self.indices[i];
        &self.data[original_idx * self.dim..(original_idx + 1) * self.dim]
    }

    fn get_point(&self, i: usize) -> &[f64] {
        let dim = self.dim;
        &self.data[i * dim..(i + 1) * dim]
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
            let p = self.get_point_from_idx(i);
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
        self.nodes.push(AggNode {
            center,
            radius,
            start,
            end,
            left: None,
            right: None,
        });

        let count = end - start;
        if count <= self.leaf_size || self.should_approximate(&self.nodes[node_idx]) {
            return node_idx;
        }

        let dim = self.select_split_dim(start, end);
        let mut mid = self.partition(start, end, dim);

        if mid == start { mid = start + 1; }
        else if mid == end { mid = end - 1; }

        let left_idx = self.build_recursive(start, mid);
        let right_idx = self.build_recursive(mid, end);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn min_distance_to_node(&self, node_idx: usize, query: &[f64]) -> f64 {
        let node = &self.nodes[node_idx];
        let d = self.metric.distance(query, &node.center);
        (d - node.radius).max(0.0)
    }

    fn approx_kde_for_node(&self, query: &[f64], node: &AggNode, h: f64, kernel: KernelType) -> f64 {
        let n_points = (node.end - node.start) as f64;

        let dist_to_center = self.metric.distance(query, &node.center);

        let min_possible = (dist_to_center - node.radius).max(0.0);
        let max_possible = dist_to_center + node.radius;

        let k_min = kernel.evaluate(min_possible, h);
        let k_max = kernel.evaluate(max_possible, h);
        let k_avg = (k_min + k_max) / 2.0;

        n_points * k_avg
    }

    fn should_approximate(&self, node: &AggNode) -> bool {
        let node_count = node.end - node.start;
        let span = 2.0 * node.radius;
        node_count >= self.min_samples && span <= self.max_span       
    }


    fn kde_recursive(&self, node_idx: usize, query: &[f64], h: f64, density: &mut f64, kernel: KernelType) {
        let node = &self.nodes[node_idx];

        if self.should_approximate(node) {
            *density += self.approx_kde_for_node(query, node, h, kernel);
            return;
        }
        
        if kernel.evaluate(self.min_distance_to_node(node_idx, query), h) < 1e-10 {
            return;
        }
        
        if self.nodes[node_idx].left.is_none() {
            for i in self.nodes[node_idx].start..self.nodes[node_idx].end {
                let dist = self.metric.distance(query, self.get_point(i));
                *density += kernel.evaluate(dist, h);
            }
            return;
        }

        if let Some(left) = self.nodes[node_idx].left {
            self.kde_recursive(left, query, h, density, kernel);
        }
        if let Some(right) = self.nodes[node_idx].right {
            self.kde_recursive(right, query, h, density, kernel);
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
            self.kde_recursive(0, query, bandwidth, &mut density, kernel);
            results[i] = density;
        }

        NdArray::from_vec(Shape::new(vec![n_queries]), results)
    }
}
