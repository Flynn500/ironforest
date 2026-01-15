#[derive(Clone, Debug)]
pub struct BallNode {
    pub center: Vec<f64>,
    pub radius: f64,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

pub struct BallTree {
    pub nodes: Vec<BallNode>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
}

impl BallTree {
    pub fn new(points: &[f64], n_points: usize, dim: usize, leaf_size: usize) -> Self{
        let mut tree = BallTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data: points.to_vec(),
            n_points,
            dim,
            leaf_size,
        };

        tree.build_recursive(0, n_points);
        tree
    }

    fn build_recursive(&mut self, start: usize, end: usize) -> usize {
        todo!()
    }

    
    fn get_point(&self, i: usize) -> &[f64] {
        let idx = self.indices[i];
        &self.data[idx * self.dim..(idx + 1) * self.dim]
    }

    //crude dist function for the time being
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());

        let mut sum = 0.0;

        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        sum
    }

    fn compute_bounding_ball(&self, start: usize, end: usize) -> (Vec<f64>, f64) {
        let n = (start - end) as f64;
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

        let mut max_dist: f64 = 0;
        for i in start..end {
            let p = self.get_point(i);
            let dist = Self::ssquared_dist(p, centroid);

            if  dist > max_dist {
                max_dist = dist;
            }
        }
        (centroid, max_dist.sqrt()) //REMOVE SQRT IF WE SWAP DISTANCE METHODS TO ALLOW MORE
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
        let mid = self.partition(start, end, dim);

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