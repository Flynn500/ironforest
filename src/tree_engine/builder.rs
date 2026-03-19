use crate::projection::{ProjectionType, RandomProjection};
use crate::tree_engine::config::{SplitCriterion, SplitGeometry, SplitResult, TaskType, TreeConfig};
use crate::tree_engine::impurity;
use crate::tree_engine::node::{Node, SplitType};
use crate::tree_engine::tree::Tree;
use crate::random::generator::Generator;

pub struct TreeBuilder<'a> {
    config: &'a TreeConfig,
    data: &'a [f64],
    labels: &'a [f64],
    n_samples: usize,
    n_features: usize,
    rng: Generator,
}

impl<'a> TreeBuilder<'a> {
    pub fn new(
        config: &'a TreeConfig,
        data: &'a [f64],
        labels: &'a [f64],
        n_samples: usize,
        n_features: usize,
    ) -> Self {
        debug_assert_eq!(data.len(), n_samples * n_features);
        debug_assert!(
            config.task_type == TaskType::AnomalyDetection || labels.len() == n_samples
        );
        Self {
            config,
            data,
            labels,
            n_samples,
            n_features,
            rng: Generator::from_seed(config.seed),
        }
    }

    pub fn build(&mut self) -> Tree {
        let indices = (0..self.n_samples).collect();
        self.build_with_indices(indices)
    }

    pub fn build_with_indices(&mut self, initial_indices: Vec<usize>) -> Tree {
        let mut tree = Tree::new(self.n_features, self.config.task_type);
        tree.n_training_samples = initial_indices.len();
        self.build_node(&mut tree, &initial_indices, 0);
        tree
    }

    fn build_node(
        &mut self,
        tree: &mut Tree,
        indices: &[usize],
        depth: usize,
    ) -> crate::tree_engine::node::NodeId {
        if depth > tree.max_depth_reached {
            tree.max_depth_reached = depth;
        }

        let n = indices.len();

        if n < self.config.min_samples_split {
            return tree.add_node(self.make_leaf(indices));
        }

        if let Some(max_d) = self.config.max_depth {
            if depth >= max_d {
                return tree.add_node(self.make_leaf(indices));
            }
        }

        if self.config.task_type == TaskType::Classification && self.is_pure(indices) {
            return tree.add_node(self.make_leaf(indices));
        }

        let split = match self.config.split_geometry {
            SplitGeometry::RandomProjection => self.find_spatial_split(indices),
            SplitGeometry::Axis => match self.config.criterion {
                SplitCriterion::Random => self.find_random_split(indices),
                _ => self.find_best_split(indices),
            },
        };

        let split = match split {
            Some(s) => s,
            None => return tree.add_node(self.make_leaf(indices)),
        };

        if split.n_left() < self.config.min_samples_leaf
            || split.n_right() < self.config.min_samples_leaf
        {
            return tree.add_node(self.make_leaf(indices));
        }

        let node_id = tree.add_node(Node::Leaf {
            value: 0.0,
            n_samples: 0,
        });

        let left_id = self.build_node(tree, &split.left_indices, depth + 1);
        let right_id = self.build_node(tree, &split.right_indices, depth + 1);

        tree.nodes[node_id.0] = Node::Internal {
            split: split.split_type,
            threshold: split.threshold,
            left: left_id,
            right: right_id,
        };

        node_id
    }

    fn make_leaf(&self, indices: &[usize]) -> Node {
        let n = indices.len();
        let value = match self.config.task_type {
            TaskType::Classification => self.majority_class(indices),
            TaskType::Regression => {
                let sum: f64 = indices.iter().map(|&i| self.labels[i]).sum();
                if n > 0 { sum / n as f64 } else { 0.0 }
            }
            TaskType::AnomalyDetection => 0.0,
        };
        Node::Leaf {
            value,
            n_samples: n,
        }
    }

    fn majority_class(&self, indices: &[usize]) -> f64 {
        let mut counts = vec![0usize; self.config.n_classes];
        for &i in indices {
            let c = self.labels[i] as usize;
            counts[c] += 1;
        }
        counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(class, _)| class as f64)
            .unwrap_or(0.0)
    }

    fn is_pure(&self, indices: &[usize]) -> bool {
        if indices.is_empty() {
            return true;
        }
        let first = self.labels[indices[0]] as usize;
        indices.iter().all(|&i| self.labels[i] as usize == first)
    }

    #[inline]
    fn feature_val(&self, sample: usize, feature: usize) -> f64 {
        self.data[sample * self.n_features + feature]
    }

    fn find_random_split(&mut self, indices: &[usize]) -> Option<SplitResult> {
        if indices.len() < 2 {
            return None;
        }

        let max_feat = self.config.effective_max_features(self.n_features);
        let mut feat_pool: Vec<usize> = (0..self.n_features).collect();
        self.rng.partial_shuffle(&mut feat_pool, max_feat);

        for &f in &feat_pool[..max_feat] {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &i in indices {
                let v = self.feature_val(i, f);
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }

            if (max_val - min_val).abs() < 1e-12 {
                continue;
            }

            let threshold = min_val + self.rng.next_f64() * (max_val - min_val);

            let (left, right): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| self.feature_val(i, f) <= threshold);

            if left.is_empty() || right.is_empty() {
                continue;
            }

            return Some(SplitResult {
                split_type: SplitType::Feature(f),
                threshold,
                gain: f64::NAN,
                left_indices: left,
                right_indices: right,
            });
        }

        None
    }

    fn find_best_split(&mut self, indices: &[usize]) -> Option<SplitResult> {
        let max_feat = self.config.effective_max_features(self.n_features);
        let mut feat_pool: Vec<usize> = (0..self.n_features).collect();
        self.rng.partial_shuffle(&mut feat_pool, max_feat);

        let parent_impurity = self.node_impurity(indices);

        let mut best: Option<SplitCandidate> = None;

        for &f in &feat_pool[..max_feat] {
            if let Some(cand) = self.best_split_for_feature(indices, f, parent_impurity) {
                let dominated = best.as_ref().map_or(true, |b| cand.gain > b.gain);
                if dominated {
                    best = Some(cand);
                }
            }
        }

        best.map(|b| {
            let (left, right): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| self.feature_val(i, b.feature) <= b.threshold);
            SplitResult {
                split_type: SplitType::Feature(b.feature),
                threshold: b.threshold,
                gain: b.gain,
                left_indices: left,
                right_indices: right,
            }
        })
    }

    fn best_split_for_feature(
        &self,
        indices: &[usize],
        feature: usize,
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        let n = indices.len();
        if n < 2 {
            return None;
        }

        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            self.feature_val(a, feature)
                .partial_cmp(&self.feature_val(b, feature))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match self.config.task_type {
            TaskType::Classification => {
                self.sweep_classification(&sorted, feature, parent_impurity)
            }
            TaskType::Regression => {
                self.sweep_regression(&sorted, feature, parent_impurity)
            }
            _ => None,
        }
    }

    fn find_spatial_split(&mut self, indices: &[usize]) -> Option<SplitResult> {
        if indices.len() < 2 {
            return None;
        }

        let projection_type = self.config.projection_type.unwrap_or(ProjectionType::Gaussian);
        let direction = RandomProjection::generate_direction(
            self.n_features,
            projection_type,
            &mut self.rng,
        );

        let mut projected: Vec<(f64, usize)> = indices
            .iter()
            .map(|&i| {
                let row = &self.data[i * self.n_features..(i + 1) * self.n_features];
                (direction.project(row), i)
            })
            .collect();

        projected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_indices: Vec<usize> = projected.iter().map(|&(_, i)| i).collect();
        let sorted_projections: Vec<f64> = projected.iter().map(|&(p, _)| p).collect();

        let parent_impurity = self.node_impurity(indices);
        let best = self.sweep_projected(
            &sorted_indices,
            &sorted_projections,
            parent_impurity,
        )?;

        let threshold = best.threshold;
        let (left, right): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| {
                let row = &self.data[i * self.n_features..(i + 1) * self.n_features];
                direction.project(row) <= threshold
            });

        if left.is_empty() || right.is_empty() {
            return None;
        }

        Some(SplitResult {
            split_type: SplitType::Projection(direction),
            threshold,
            gain: best.gain,
            left_indices: left,
            right_indices: right,
        })
    }

    fn sweep_classification(
        &self,
        sorted: &[usize],
        feature: usize,
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        let nc = self.config.n_classes;
        let n = sorted.len();

        let mut left_counts = vec![0usize; nc];
        let mut right_counts = vec![0usize; nc];
        for &i in sorted {
            let c = self.labels[i] as usize;
            right_counts[c] += 1;
        }

        let mut best: Option<SplitCandidate> = None;

        for pos in 0..n - 1 {
            let c = self.labels[sorted[pos]] as usize;
            left_counts[c] += 1;
            right_counts[c] -= 1;

            let n_left = pos + 1;
            let n_right = n - n_left;

            let val_curr = self.feature_val(sorted[pos], feature);
            let val_next = self.feature_val(sorted[pos + 1], feature);
            if (val_next - val_curr).abs() < 1e-12 {
                continue;
            }

            if n_left < self.config.min_samples_leaf
                || n_right < self.config.min_samples_leaf
            {
                continue;
            }

            let imp_left = self.impurity_from_counts(&left_counts);
            let imp_right = self.impurity_from_counts(&right_counts);
            let weighted = impurity::weighted_impurity(n_left, imp_left, n_right, imp_right);
            let gain = parent_impurity - weighted;

            let dominated = best.as_ref().map_or(true, |b| gain > b.gain);
            if dominated && gain > 0.0 {
                best = Some(SplitCandidate {
                    feature,
                    threshold: (val_curr + val_next) / 2.0,
                    gain,
                });
            }
        }

        best
    }

    fn sweep_regression(
        &self,
        sorted: &[usize],
        feature: usize,
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        let n = sorted.len();

        let mut total_sum = 0.0_f64;
        let mut total_sum_sq = 0.0_f64;
        for &i in sorted {
            let y = self.labels[i];
            total_sum += y;
            total_sum_sq += y * y;
        }

        let mut left_sum = 0.0_f64;
        let mut left_sum_sq = 0.0_f64;
        let mut best: Option<SplitCandidate> = None;

        for pos in 0..n - 1 {
            let y = self.labels[sorted[pos]];
            left_sum += y;
            left_sum_sq += y * y;

            let n_left = (pos + 1) as f64;
            let n_right = (n - pos - 1) as f64;

            let val_curr = self.feature_val(sorted[pos], feature);
            let val_next = self.feature_val(sorted[pos + 1], feature);
            if (val_next - val_curr).abs() < 1e-12 {
                continue;
            }

            if (pos + 1) < self.config.min_samples_leaf
                || (n - pos - 1) < self.config.min_samples_leaf
            {
                continue;
            }

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            let mse_left = (left_sum_sq / n_left) - (left_sum / n_left).powi(2);
            let mse_right = (right_sum_sq / n_right) - (right_sum / n_right).powi(2);

            let weighted = impurity::weighted_impurity(
                pos + 1,
                mse_left.max(0.0),
                n - pos - 1,
                mse_right.max(0.0),
            );
            let gain = parent_impurity - weighted;

            let dominated = best.as_ref().map_or(true, |b| gain > b.gain);
            if dominated && gain > 0.0 {
                best = Some(SplitCandidate {
                    feature,
                    threshold: (val_curr + val_next) / 2.0,
                    gain,
                });
            }
        }

        best
    }

    fn sweep_projected(
        &self,
        sorted_indices: &[usize],
        projections: &[f64],
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        match self.config.task_type {
            TaskType::Classification => {
                self.sweep_projected_classification(sorted_indices, projections, parent_impurity)
            }
            TaskType::Regression => {
                self.sweep_projected_regression(sorted_indices, projections, parent_impurity)
            }
            _ => None,
        }
    }

    fn sweep_projected_classification(
        &self,
        sorted_indices: &[usize],
        projections: &[f64],
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        let nc = self.config.n_classes;
        let n = sorted_indices.len();

        let mut left_counts = vec![0usize; nc];
        let mut right_counts = vec![0usize; nc];
        for &i in sorted_indices {
            let c = self.labels[i] as usize;
            right_counts[c] += 1;
        }

        let mut best: Option<SplitCandidate> = None;

        for pos in 0..n - 1 {
            let c = self.labels[sorted_indices[pos]] as usize;
            left_counts[c] += 1;
            right_counts[c] -= 1;

            let n_left = pos + 1;
            let n_right = n - n_left;

            if (projections[pos + 1] - projections[pos]).abs() < 1e-12 {
                continue;
            }

            if n_left < self.config.min_samples_leaf
                || n_right < self.config.min_samples_leaf
            {
                continue;
            }

            let imp_left = self.impurity_from_counts(&left_counts);
            let imp_right = self.impurity_from_counts(&right_counts);
            let weighted = impurity::weighted_impurity(n_left, imp_left, n_right, imp_right);
            let gain = parent_impurity - weighted;

            let dominated = best.as_ref().map_or(true, |b| gain > b.gain);
            if dominated && gain > 0.0 {
                best = Some(SplitCandidate {
                    feature: 0, // unused for spatial splits
                    threshold: (projections[pos] + projections[pos + 1]) / 2.0,
                    gain,
                });
            }
        }

        best
    }

    fn sweep_projected_regression(
        &self,
        sorted_indices: &[usize],
        projections: &[f64],
        parent_impurity: f64,
    ) -> Option<SplitCandidate> {
        let n = sorted_indices.len();

        let mut total_sum = 0.0_f64;
        let mut total_sum_sq = 0.0_f64;
        for &i in sorted_indices {
            let y = self.labels[i];
            total_sum += y;
            total_sum_sq += y * y;
        }

        let mut left_sum = 0.0_f64;
        let mut left_sum_sq = 0.0_f64;
        let mut best: Option<SplitCandidate> = None;

        for pos in 0..n - 1 {
            let y = self.labels[sorted_indices[pos]];
            left_sum += y;
            left_sum_sq += y * y;

            let n_left = (pos + 1) as f64;
            let n_right = (n - pos - 1) as f64;

            if (projections[pos + 1] - projections[pos]).abs() < 1e-12 {
                continue;
            }

            if (pos + 1) < self.config.min_samples_leaf
                || (n - pos - 1) < self.config.min_samples_leaf
            {
                continue;
            }

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;

            let mse_left = (left_sum_sq / n_left) - (left_sum / n_left).powi(2);
            let mse_right = (right_sum_sq / n_right) - (right_sum / n_right).powi(2);

            let weighted = impurity::weighted_impurity(
                pos + 1,
                mse_left.max(0.0),
                n - pos - 1,
                mse_right.max(0.0),
            );
            let gain = parent_impurity - weighted;

            let dominated = best.as_ref().map_or(true, |b| gain > b.gain);
            if dominated && gain > 0.0 {
                best = Some(SplitCandidate {
                    feature: 0,
                    threshold: (projections[pos] + projections[pos + 1]) / 2.0,
                    gain,
                });
            }
        }

        best
    }

    fn node_impurity(&self, indices: &[usize]) -> f64 {
        match self.config.task_type {
            TaskType::Classification => {
                let counts = self.class_counts(indices);
                self.impurity_from_counts(&counts)
            }
            TaskType::Regression => {
                let r = impurity::mse_stats(indices.iter().map(|&i| self.labels[i]));
                r.mse
            }
            TaskType::AnomalyDetection => 0.0,
        }
    }

    fn impurity_from_counts(&self, counts: &[usize]) -> f64 {
        match self.config.criterion {
            SplitCriterion::Entropy => impurity::entropy(counts),
            _ => impurity::gini(counts),
        }
    }

    fn class_counts(&self, indices: &[usize]) -> Vec<usize> {
        let mut counts = vec![0usize; self.config.n_classes];
        for &i in indices {
            let c = self.labels[i] as usize;
            counts[c] += 1;
        }
        counts
    }
}

struct SplitCandidate {
    feature: usize,
    threshold: f64,
    gain: f64,
}