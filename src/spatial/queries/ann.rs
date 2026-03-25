use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::{Generator, array::NdArray, spatial::HeapItem};
use rayon::prelude::*;
use crate::spatial::SpatialTree;
use num_traits::{ToPrimitive, identities::Zero as _, Float};


const ANN_PAR_THRESHOLD: usize = 512;

//enum for stochastic probing, controls whether probes follow an existing
//path, diverging at the specified depth or if they just follow the best
//route
pub enum ProbeInit<'a, F> {
    FromRoot {
        path: &'a mut Vec<(usize, usize, usize, F)>,
    },

    FromPath {
        path: &'a [(usize, usize, usize, F)],
        diverge_depth: usize,
    },
}

pub trait AnnQuery: SpatialTree {
    //deterministic aNN via n_candidates
    fn query_ann(&self, query: &[Self::Float], k: usize, n_candidates: usize) -> Vec<(usize, Self::Float)> {
        if k == 0 || self.n_points() == 0 {
            return Vec::new();
        }

        self.ann_candidates_inner(query, k, n_candidates.max(k))
    }

    fn ann_candidates_inner(&self, query: &[Self::Float], k: usize, n_candidates: usize) -> Vec<(usize, Self::Float)> {
        let mut queue: BinaryHeap<Reverse<HeapItem<Self::Float>>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<HeapItem<Self::Float>> = BinaryHeap::new();

        queue.push(Reverse(HeapItem { distance: Self::Float::zero(), index: self.root() }));

        while let Some(Reverse(HeapItem { distance: node_dist, index: node_idx })) = queue.pop() {
            if candidates.len() >= k {
                if node_dist > candidates.peek().unwrap().distance {
                    break;
                }
            }

            if self.node_left(node_idx).is_none() {
                for i in self.node_start(node_idx)..self.node_end(node_idx) {
                    let dist = match Self::REDUCED {
                        true => self.metric().reduced_distance(query, self.get_point(i)),
                        false => self.metric().distance(query, self.get_point(i)),
                    };
                    if candidates.len() < n_candidates {
                        candidates.push(HeapItem { distance: dist, index: self.indices()[i] });
                    } else if dist < candidates.peek().unwrap().distance {
                        candidates.pop();
                        candidates.push(HeapItem { distance: dist, index: self.indices()[i] });
                    }
                }
                if candidates.len() >= k {
                    if node_dist > candidates.peek().unwrap().distance {
                        break;
                    }
                }
            } else {
                let plan = self.plan_traversal(node_idx, query);

                queue.push(Reverse(HeapItem {
                    distance: plan.first.lower_bound,
                    index: plan.first.child_idx,
                }));
                queue.push(Reverse(HeapItem {
                    distance: plan.second.lower_bound,
                    index: plan.second.child_idx,
                }));
            }
        }
        let mut results: Vec<(usize, Self::Float)> = candidates.into_iter()
            .map(|item| {
                let dist = if Self::REDUCED {
                    self.metric().post_transform(item.distance)
                } else {
                    item.distance
                };
                (item.index, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn query_ann_batch(&self, queries: &NdArray<Self::Float>, k: usize, n_candidates: usize) -> Vec<Vec<(usize, Self::Float)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let n_queries = shape[0];
        let dim = shape[1];
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[Self::Float] = &queries_cow;
        if n_queries >= ANN_PAR_THRESHOLD {
            self.par_ann_batch(queries_slice, n_queries, dim, k, n_candidates)
        } else {
            self.seq_ann_batch(queries_slice, n_queries, dim, k, n_candidates)
        }
    }

    fn seq_ann_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }

    fn par_ann_batch(&self, queries: &[Self::Float], n_queries: usize, dim: usize, k: usize, n_candidates: usize) -> Vec<Vec<(usize, Self::Float)>> {
        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = &queries[i * dim..(i + 1) * dim];
                self.query_ann(query, k, n_candidates)
            })
            .collect()
    }

    // Stochastic aNN via  n_propes (the number of types to traverse the tree)
    fn stochastic_probe(
        &self,
        query: &[Self::Float],
        k: usize,
        n_candidates: usize,
        rng: &mut Generator,
        tau: Self::Float,
        candidates: &mut BinaryHeap<HeapItem<Self::Float>>,
        seen: &mut [u64],
        bail_threshold: f64,
        init: ProbeInit<'_, Self::Float>,
    ) {
        let mut queue: BinaryHeap<Reverse<HeapItem<Self::Float>>> = BinaryHeap::new();

        let mut bound = if candidates.len() >= n_candidates {
            candidates.peek().unwrap().distance
        } else {
            Self::Float::infinity()
        };

        let mut record_path: Option<&mut Vec<(usize, usize, usize, Self::Float)>> = None;

        match init {
            ProbeInit::FromRoot { path } => {
                queue.push(Reverse(HeapItem {
                    distance: Self::Float::zero(),
                    index: self.root(),
                }));
                record_path = Some(path);
            }
            ProbeInit::FromPath { path, diverge_depth } => {
                let replay_len = diverge_depth.min(path.len());
                for i in 0..replay_len {
                    let (_, _, other, margin) = path[i];
                    if margin <= bound {
                        queue.push(Reverse(HeapItem { distance: margin, index: other }));
                    }
                }
                if replay_len < path.len() {
                    let (node_idx, _, _, _) = path[replay_len];
                    queue.push(Reverse(HeapItem {
                        distance: Self::Float::zero(),
                        index: node_idx,
                    }));
                }
            }
        }

        let mut improvement_rate: f64 = 1.0;
        let alpha: f64 = 0.3;
        let min_leaves: usize = 3;
        let mut leaves_visited: usize = 0;

        while let Some(Reverse(HeapItem { distance: node_dist, index: node_idx })) = queue.pop() {
            if node_dist > bound {
                break;
            }

            if self.node_left(node_idx).is_none() {
                let mut improved = false;
                for i in self.node_start(node_idx)..self.node_end(node_idx) {
                    let idx = self.indices()[i];

                    let word = idx >> 6;
                    let bit = 1u64 << (idx & 63);
                    if seen[word] & bit != 0 {
                        continue;
                    }
                    seen[word] |= bit;

                    let dist = match Self::REDUCED {
                        true => self.metric().reduced_distance(query, self.get_point(i)),
                        false => self.metric().distance(query, self.get_point(i)),
                    };
                    if candidates.len() < n_candidates {
                        candidates.push(HeapItem { distance: dist, index: idx });
                        if candidates.len() == n_candidates {
                            bound = candidates.peek().unwrap().distance;
                        }
                        improved = true;
                    } else if dist < bound {
                        candidates.pop();
                        candidates.push(HeapItem { distance: dist, index: idx });
                        bound = candidates.peek().unwrap().distance;
                        improved = true;
                    }
                }

                leaves_visited += 1;
                improvement_rate = alpha * (if improved { 1.0 } else { 0.0 })
                    + (1.0 - alpha) * improvement_rate;
                if candidates.len() >= k
                    && leaves_visited >= min_leaves
                    && improvement_rate < bail_threshold
                {
                    break;
                }
            } else {
                let plan = self.plan_traversal(node_idx, query);

                if let Some(ref mut path) = record_path {
                    path.push((node_idx, plan.first.child_idx, plan.second.child_idx, plan.second.lower_bound));
                }

                queue.push(Reverse(HeapItem {
                    distance: Self::Float::zero(),
                    index: plan.first.child_idx,
                }));

                if plan.second.lower_bound <= bound {
                    let tau_f64 = tau.to_f64().unwrap();
                    if tau_f64 == 0.0 {
                        queue.push(Reverse(HeapItem { distance: plan.second.lower_bound, index: plan.second.child_idx }));
                    } else {
                        let norm_margin = plan.second.lower_bound.to_f64().unwrap() / tau_f64;
                        let p = 1.0 / (1.0 + (-norm_margin).exp());
                        let perturbed = if rng.next_f64() < p {
                            plan.second.lower_bound
                        } else {
                            Self::Float::zero()
                        };
                        queue.push(Reverse(HeapItem { distance: perturbed, index: plan.second.child_idx }));
                    }
                }
            }
        }
    }

    fn query_ann_stochastic_batch(
        &self,
        queries: &NdArray<Self::Float>,
        k: usize,
        n_candidates: usize,
        n_probes: usize,
    ) -> Vec<Vec<(usize, Self::Float)>> {
        let shape = queries.shape().dims();
        assert!(shape.len() == 2, "Expected 2D array (n_queries, dim)");
        let (n_queries, dim) = (shape[0], shape[1]);
        assert_eq!(dim, self.dim(), "Query dimension must match tree dimension");

        let queries_cow = queries.as_contiguous_slice();
        let queries_slice: &[Self::Float] = &queries_cow;

        if n_queries >= ANN_PAR_THRESHOLD {
            (0..n_queries)
                .into_par_iter()
                .map(|i| {
                    let query = &queries_slice[i * dim..(i + 1) * dim];
                    self.query_ann_stochastic(query, k, n_candidates, n_probes)
                })
                .collect()
        } else {
            (0..n_queries)
                .map(|i| {
                    let query = &queries_slice[i * dim..(i + 1) * dim];
                    self.query_ann_stochastic(query, k, n_candidates, n_probes)
                })
                .collect()
        }
    }

    fn query_ann_stochastic(
        &self,
        query: &[Self::Float],
        k: usize,
        n_candidates: usize,
        n_probes: usize,
    ) -> Vec<(usize, Self::Float)> {
        if k == 0 || self.n_points() == 0 {
            return Vec::new();
        }
        let mut candidates: BinaryHeap<HeapItem<Self::Float>> = BinaryHeap::new();
        
        let n_words = (self.n_points() + 63) / 64;
        let mut seen: Vec<u64> = vec![0u64; n_words];
        
        let bail_threshold = 0.075;

        let mut path: Vec<(usize, usize, usize, Self::Float)> = Vec::new();
        let mut rng = Generator::from_seed(0);
        self.stochastic_probe(
            query, k, n_candidates, &mut rng, Self::Float::zero(),
            &mut candidates, &mut seen, bail_threshold * 2.0,
            ProbeInit::FromRoot { path: &mut path },
        );

        let base_tau = if path.is_empty() {
            0.0
        } else {
            let mut margins: Vec<f64> = path.iter()
                .map(|(_, _, _, m)| m.to_f64().unwrap())
                .collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
            margins[margins.len() / 2]
        };

        if n_probes > 1 && !path.is_empty() {
            let max_depth = path.len();
            for i in 1..n_probes {
                let t = i as f64 / n_probes as f64;
                let tau_i = num_traits::cast::<f64, Self::Float>(base_tau * t).unwrap();
                let diverge_depth = max_depth * (n_probes - i) / n_probes;
                let mut rng = Generator::from_seed(1 + i as u64);
                self.stochastic_probe(
                    query, k, n_candidates, &mut rng, tau_i,
                    &mut candidates, &mut seen, bail_threshold,
                    ProbeInit::FromPath { path: &path, diverge_depth },
                );
            }
        }

        let mut results: Vec<(usize, Self::Float)> = candidates.into_iter()
            .map(|item| {
                let dist = if Self::REDUCED {
                    self.metric().post_transform(item.distance)
                } else {
                    item.distance
                };
                (item.index, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
}