use crate::tree_engine::config::TaskType;
use crate::tree_engine::node::{Node, NodeId, SplitType};
use crate::tree_engine::scoring;

#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub n_features: usize,
    pub task_type: TaskType,
    pub max_depth_reached: usize,
    pub n_training_samples: usize,
}

impl Tree {
    pub fn new(n_features: usize, task_type: TaskType) -> Self {
        Self {
            nodes: Vec::new(),
            n_features,
            task_type,
            max_depth_reached: 0,
            n_training_samples: 0,
        }
    }

    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn predict_single(&self, sample: &[f64]) -> PredictionResult {
        debug_assert_eq!(sample.len(), self.n_features);

        let mut current = NodeId(0);
        let mut depth: usize = 0;

        loop {
            match &self.nodes[current.0] {
                Node::Internal {
                    split,
                    threshold,
                    left,
                    right,
                } => {
                    let val = match split {
                        SplitType::Feature(f) => sample[*f],
                        SplitType::Projection(dir) => dir.project(sample),
                    };
                    current = if val <= *threshold { *left } else { *right };
                    depth += 1;
                }
                Node::Leaf { value, n_samples } => {
                    return PredictionResult {
                        value: *value,
                        n_samples: *n_samples,
                        depth,
                    };
                }
            }
        }
    }

    pub fn predict_batch(&self, data: &[f64], n_samples: usize) -> Vec<PredictionResult> {
        debug_assert_eq!(data.len(), n_samples * self.n_features);

        (0..n_samples)
            .map(|i| {
                let row = &data[i * self.n_features..(i + 1) * self.n_features];
                self.predict_single(row)
            })
            .collect()
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn predict_values(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        self.predict_batch(data, n_samples)
            .iter()
            .map(|r| r.value)
            .collect()
    }

    pub fn predict_anomaly_scores(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        self.predict_batch(data, n_samples)
            .iter()
            .map(|r| {
                scoring::anomaly_score(r.depth, r.n_samples, self.n_training_samples)
            })
            .collect()
    }

    pub fn predict_path_lengths(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        self.predict_batch(data, n_samples)
            .iter()
            .map(|r| {
                r.depth as f64 + scoring::average_path_length(r.n_samples)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PredictionResult {
    pub value: f64,
    pub n_samples: usize,
    pub depth: usize,
}