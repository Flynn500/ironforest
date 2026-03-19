use crate::{projection::ProjectionType, tree_engine::node::SplitType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
    AnomalyDetection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCriterion {
    Gini,
    Entropy,
    Mse,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitGeometry {
    Axis,
    RandomProjection,
}

#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub criterion: SplitCriterion,
    pub split_geometry: SplitGeometry,
    pub task_type: TaskType,
    pub n_classes: usize,
    pub seed: u64,
    pub projection_type: Option<ProjectionType>,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: SplitCriterion::Gini,
            split_geometry: SplitGeometry::Axis,
            task_type: TaskType::Classification,
            n_classes: 2,
            seed: 0,
            projection_type: None,
        }
    }
}

impl TreeConfig {
    pub fn classification(n_classes: usize) -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            task_type: TaskType::Classification,
            n_classes,
            ..Default::default()
        }
    }

    pub fn regression() -> Self {
        Self {
            criterion: SplitCriterion::Mse,
            task_type: TaskType::Regression,
            n_classes: 0,
            ..Default::default()
        }
    }

    pub fn isolation(max_samples: usize) -> Self {
        let max_depth = (max_samples as f64).log2().ceil() as usize;
        Self {
            max_depth: Some(max_depth),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: Some(1),
            criterion: SplitCriterion::Random,
            task_type: TaskType::AnomalyDetection,
            n_classes: 0,
            ..Default::default()
        }
    }

    pub fn effective_max_features(&self, n_features: usize) -> usize {
        match self.max_features {
            Some(mf) => mf.min(n_features).max(1),
            None => n_features,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitResult {
    pub split_type: SplitType,
    pub threshold: f64,
    pub gain: f64,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
}

impl SplitResult {
    pub fn n_left(&self) -> usize {
        self.left_indices.len()
    }

    pub fn n_right(&self) -> usize {
        self.right_indices.len()
    }
}