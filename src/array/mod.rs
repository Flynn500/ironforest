pub(crate) mod broadcast;
pub(crate) mod ndarray;
pub(crate) mod shape;
pub(crate) mod storage;
pub(crate) mod constructors;
pub(crate) mod strided_iter;

pub use broadcast::BroadcastIter;
pub use ndarray::NdArray;
pub use shape::Shape;
pub use storage::Storage;
pub use strided_iter::StridedIter;
