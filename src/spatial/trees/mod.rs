pub(crate) mod kd_tree;
pub(crate) mod ball_tree;
pub(crate) mod vp_tree;
pub(crate) mod m_tree;
pub(crate) mod rp_tree;
pub(crate) mod agg_tree;
pub(crate) mod brute_force;

pub use vp_tree::VantagePointSelection;

// f64 type aliases — default public API, backward-compatible names
pub type KDTree = kd_tree::KDTree<f64>;
pub type BallTree = ball_tree::BallTree<f64>;
pub type VPTree = vp_tree::VPTree<f64>;
pub type MTree = m_tree::MTree<f64>;
pub type RPTree = rp_tree::RPTree<f64>;
pub type AggTree = agg_tree::AggTree<f64>;
pub type BruteForce = brute_force::BruteForce<f64>;

// Explicit f64 aliases (same as above, kept for clarity)
pub type KDTree64 = kd_tree::KDTree<f64>;
pub type BallTree64 = ball_tree::BallTree<f64>;
pub type VPTree64 = vp_tree::VPTree<f64>;
pub type MTree64 = m_tree::MTree<f64>;
pub type RPTree64 = rp_tree::RPTree<f64>;
pub type AggTree64 = agg_tree::AggTree<f64>;
pub type BruteForce64 = brute_force::BruteForce<f64>;

// f32 aliases
pub type KDTree32 = kd_tree::KDTree<f32>;
pub type BallTree32 = ball_tree::BallTree<f32>;
pub type VPTree32 = vp_tree::VPTree<f32>;
pub type MTree32 = m_tree::MTree<f32>;
pub type RPTree32 = rp_tree::RPTree<f32>;
pub type AggTree32 = agg_tree::AggTree<f32>;
pub type BruteForce32 = brute_force::BruteForce<f32>;
