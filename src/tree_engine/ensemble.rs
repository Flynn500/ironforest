use crate::tree_engine::builder::TreeBuilder;
use crate::tree_engine::config::{TaskType, TreeConfig};
use crate::tree_engine::scoring;
use crate::tree_engine::tree::Tree;
use crate::random::generator::Generator;

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub n_trees: usize,
    pub tree_config: TreeConfig,
    pub bootstrap: bool,
    pub max_samples: Option<usize>,
    pub seed: u64,
}

impl EnsembleConfig {
    pub fn random_forest_classifier(n_trees: usize, n_classes: usize) -> Self {
        Self {
            n_trees,
            tree_config: TreeConfig::classification(n_classes),
            bootstrap: true,
            max_samples: None,
            seed: 42,
        }
    }

    pub fn random_forest_regressor(n_trees: usize) -> Self {
        Self {
            n_trees,
            tree_config: TreeConfig::regression(),
            bootstrap: true,
            max_samples: None,
            seed: 42,
        }
    }

    pub fn isolation_forest(n_trees: usize, max_samples: usize) -> Self {
        Self {
            n_trees,
            tree_config: TreeConfig::isolation(max_samples),
            bootstrap: false,
            max_samples: Some(max_samples),
            seed: 42,
        }
    }
}

pub struct Ensemble {
    pub trees: Vec<Tree>,
    pub config: EnsembleConfig,
    pub n_training_samples: usize,
}

impl Ensemble {
    pub fn fit(
        config: EnsembleConfig,
        data: &[f64],
        labels: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> Self {
        debug_assert_eq!(data.len(), n_samples * n_features);

        let mut cfg = config.clone();
        if cfg.tree_config.max_features.is_none()
            && cfg.tree_config.task_type != TaskType::AnomalyDetection
        {
            cfg.tree_config.max_features = Some(match cfg.tree_config.task_type {
                TaskType::Classification => (n_features as f64).sqrt().ceil() as usize,
                TaskType::Regression => (n_features as f64 / 3.0).ceil().max(1.0) as usize,
                _ => n_features,
            });
        }

        let effective_max_samples = cfg
            .max_samples
            .unwrap_or(n_samples)
            .min(n_samples);

        let tree_configs: Vec<TreeConfig> = (0..cfg.n_trees)
            .map(|i| {
                let mut tc = cfg.tree_config.clone();
                tc.seed = cfg.seed.wrapping_add(i as u64);
                tc
            })
            .collect();

        let trees: Vec<Tree> = tree_configs
            .into_par_iter()
            .map(|tc| {
                let mut rng = Generator::from_seed(tc.seed);

                let indices = if cfg.bootstrap {
                    bootstrap_sample(&mut rng, n_samples, effective_max_samples)
                } else if effective_max_samples < n_samples {
                    subsample_without_replacement(&mut rng, n_samples, effective_max_samples)
                } else {
                    (0..n_samples).collect()
                };

                let sub_n = indices.len();
                let mut sub_data = Vec::with_capacity(sub_n * n_features);
                let mut sub_labels = Vec::with_capacity(sub_n);

                for &i in &indices {
                    let row_start = i * n_features;
                    sub_data.extend_from_slice(&data[row_start..row_start + n_features]);
                    if !labels.is_empty() {
                        sub_labels.push(labels[i]);
                    }
                }

                let mut builder = TreeBuilder::new(&tc, &sub_data, &sub_labels, sub_n, n_features);
                let mut tree = builder.build();
                tree.n_training_samples = sub_n;
                tree
            })
            .collect();

        Ensemble {
            trees,
            config: cfg,
            n_training_samples: n_samples,
        }
    }

    pub fn predict(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        match self.config.tree_config.task_type {
            TaskType::Classification => self.predict_classification(data, n_samples),
            TaskType::Regression => self.predict_regression(data, n_samples),
            TaskType::AnomalyDetection => self.predict_anomaly(data, n_samples),
        }
    }

    fn predict_classification(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        let n_classes = self.config.tree_config.n_classes;

        let all_preds: Vec<Vec<f64>> = self
            .trees
            .iter()
            .map(|t| t.predict_values(data, n_samples))
            .collect();

        (0..n_samples)
            .map(|i| {
                let mut votes = vec![0usize; n_classes];
                for preds in &all_preds {
                    let class = preds[i] as usize;
                    if class < n_classes {
                        votes[class] += 1;
                    }
                }
                votes
                    .iter()
                    .enumerate()
                   .max_by_key(|(_, v)| *v)
                    .map(|(c, _)| c as f64)
                    .unwrap_or(0.0)
            })
            .collect()
    }

    fn predict_regression(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        let all_preds: Vec<Vec<f64>> = self
            .trees
            .iter()
            .map(|t| t.predict_values(data, n_samples))
            .collect();

        let n_trees = self.trees.len() as f64;
        (0..n_samples)
            .map(|i| {
                let sum: f64 = all_preds.iter().map(|p| p[i]).sum();
                sum / n_trees
            })
            .collect()
    }

    fn predict_anomaly(&self, data: &[f64], n_samples: usize) -> Vec<f64> {
        let all_paths: Vec<Vec<f64>> = self
            .trees
            .iter()
            .map(|t| t.predict_path_lengths(data, n_samples))
            .collect();

        let n_trees = self.trees.len() as f64;
        let eff_samples = self
            .config
            .max_samples
            .unwrap_or(self.n_training_samples);

        let c_n = scoring::average_path_length(eff_samples);

        (0..n_samples)
            .map(|i| {
                let mean_path: f64 = all_paths.iter().map(|p| p[i]).sum::<f64>() / n_trees;
                if c_n == 0.0 {
                    0.5
                } else {
                    2.0_f64.powf(-mean_path / c_n)
                }
            })
            .collect()
    }

    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

fn bootstrap_sample(rng: &mut Generator, n: usize, size: usize) -> Vec<usize> {
    (0..size).map(|_| rng.usize_below(n)).collect()
}

fn subsample_without_replacement(rng: &mut Generator, n: usize, size: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    let k = size.min(n);
    for i in 0..k {
        let j = i + rng.usize_below(n - i);
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}