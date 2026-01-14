pub(crate) mod broadcast;
pub(crate) mod ndarray;
pub(crate) mod shape;
pub(crate) mod storage;
pub(crate) mod stats;
pub(crate) mod linalg;
pub(crate) mod constructors;

pub use broadcast::BroadcastIter;
pub use ndarray::NdArray;
pub use shape::Shape;
pub use storage::Storage;
