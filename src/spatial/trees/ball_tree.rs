use crate::{Shape, array::NdArray, spatial::common::{DistanceMetric, IronFloat}};
use crate::spatial::queries::{KnnQuery, RadiusQuery, KdeQuery, AnnQuery};
use crate::spatial::SpatialTree;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct BallNode<T> {
    pub center: Vec<T>,
    pub radius: T,
    pub start: usize,
    pub end: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "T: IronFloat")]
pub struct BallTree<T: IronFloat> {
    pub nodes: Vec<BallNode<T>>,
    pub indices: Vec<usize>,
    pub data: NdArray<T>,
    pub n_points: usize,
    pub dim: usize,
    pub leaf_size: usize,
    pub metric: DistanceMetric,
    pub data_is_reordered: bool,
}

impl<T: IronFloat> BallTree<T> {
    pub fn new(mut data: NdArray<T>, leaf_size: usize, metric: DistanceMetric) -> Self {
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
        let mut tree = BallTree {
            nodes: Vec::new(),
            indices: (0..n_points).collect(),
            data,
            n_points,
            dim,
            leaf_size,
            metric,
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
        let mut new_data = vec![T::zero(); self.data.len()];

        for (new_idx, &old_idx) in self.indices.iter().enumerate() {
            let dst = new_idx * self.dim;
            new_data[dst..dst + self.dim].copy_from_slice(self.data.row(old_idx));
        }

        self.data = NdArray::from_vec(Shape::new(vec![self.n_points, self.dim]), new_data);
    }

    fn init_node(&self, start: usize, end: usize) -> (Vec<T>, T) {
        let n: T = T::from(end - start).unwrap();
        let mut centroid = vec![T::zero(); self.dim];

        for i in start..end {
            let p = self.data.row(self.indices[i]);
            for (j, &x) in p.iter().enumerate() {
                centroid[j] = centroid[j] + x;
            }
        }

        for c in &mut centroid {
            *c = *c / n;
        }

        let mut max_dist = T::zero();
        for i in start..end {
            let p = self.data.row(self.indices[i]);
            let dist = self.metric.post_transform(self.metric.reduced_distance(p, &centroid));

            if dist > max_dist {
                max_dist = dist;
            }
        }
        (centroid, max_dist)
    }

    fn furthest_from(&self, query: &[T], start: usize, end: usize) -> usize {
        (start..end)
            .max_by(|&a, &b| {
                let da = self.metric.reduced_distance(query, self.data.row(self.indices[a]));
                let db = self.metric.reduced_distance(query, self.data.row(self.indices[b]));
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
    }

    fn pivot_partition(&mut self, start: usize, end: usize) -> usize {
        let centroid: Vec<T> = {
            let n: T = T::from(end - start).unwrap();
            let mut c = vec![T::zero(); self.dim];
            for i in start..end {
                let p = self.data.row(self.indices[i]);
                for (j, &x) in p.iter().enumerate() { c[j] = c[j] + x / n; }
            }
            c
        };

        let p1_slot = self.furthest_from(&centroid, start, end);
        let p1 = self.data.row(self.indices[p1_slot]).to_vec();

        let p2_slot = self.furthest_from(&p1, start, end);
        let p2 = self.data.row(self.indices[p2_slot]).to_vec();

        let axis: Vec<T> = p2.iter().zip(&p1).map(|(a, b)| *a - *b).collect();

        let mut projections: Vec<(T, usize)> = (start..end)
            .map(|i| {
                let p = self.data.row(self.indices[i]);
                let proj: T = p.iter().zip(&axis).map(|(x, a)| *x * *a).sum();
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

impl<T: IronFloat> SpatialTree for BallTree<T> {
    type Node = BallNode<T>;
    type Float = T;
    const REDUCED: bool = true;

    fn nodes(&self) -> &[BallNode<T>] { &self.nodes }
    fn indices(&self) -> &[usize] { &self.indices }
    fn data(&self) -> &[T] { self.data.as_slice_unchecked() }
    fn dim(&self) -> usize { self.dim }
    fn metric(&self) -> &DistanceMetric { &self.metric }
    fn n_points(&self) -> usize { self.n_points }
    fn data_is_reordered(&self) -> bool { self.data_is_reordered }

    fn node_start(&self, idx: usize) -> usize { self.nodes[idx].start }
    fn node_end(&self, idx: usize) -> usize { self.nodes[idx].end }
    fn node_left(&self, idx: usize) -> Option<usize> { self.nodes[idx].left }
    fn node_right(&self, idx: usize) -> Option<usize> { self.nodes[idx].right }

    fn min_distance_to_node(&self, node_idx: usize, query: &[T]) -> T {
        let node = &self.nodes[node_idx];
        let d = self.metric.post_transform(self.metric.reduced_distance(query, &node.center));
        let min_real = (d - node.radius).max(T::zero());
        self.metric.to_reduced(min_real)
    }

    fn knn_child_order(&self, node_idx: usize, query: &[T]) -> (usize, usize) {
        let node = &self.nodes[node_idx];
        let (l, r) = (node.left.unwrap(), node.right.unwrap());
        let dl = self.metric.reduced_distance(query, &self.nodes[l].center);
        let dr = self.metric.reduced_distance(query, &self.nodes[r].center);
        if dl <= dr { (l, r) } else { (r, l) }
    }
}

impl<T: IronFloat> KnnQuery for BallTree<T> {}
impl<T: IronFloat> RadiusQuery for BallTree<T> {}
impl<T: IronFloat> KdeQuery for BallTree<T> {}
impl<T: IronFloat> AnnQuery for BallTree<T> {}
