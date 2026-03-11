use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::{projection::ProjectionType, tree_engine::{
    builder::TreeBuilder, config::{SplitCriterion as RustSplitCriterion, TaskType as RustTaskType, TreeConfig as RustTreeConfig}, ensemble::{Ensemble as RustEnsemble, EnsembleConfig as RustEnsembleConfig}, tree::Tree as RustTree
}};
use super::ArrayLike;
use numpy::{PyArray1, IntoPyArray};

#[pyclass(name = "TaskType")]
#[derive(Clone, Copy)]
pub struct PyTaskType {
    inner: RustTaskType,
}

#[pymethods]
impl PyTaskType {
    #[staticmethod]
    fn classification() -> Self {
        PyTaskType { inner: RustTaskType::Classification }
    }

    #[staticmethod]
    fn regression() -> Self {
        PyTaskType { inner: RustTaskType::Regression }
    }

    #[staticmethod]
    fn anomaly_detection() -> Self {
        PyTaskType { inner: RustTaskType::AnomalyDetection }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RustTaskType::Classification => "TaskType.CLASSIFICATION".to_string(),
            RustTaskType::Regression => "TaskType.REGRESSION".to_string(),
            RustTaskType::AnomalyDetection => "TaskType.ANOMALY_DETECTION".to_string(),
        }
    }
}

#[pyclass(name = "SplitCriterion")]
#[derive(Clone, Copy)]
pub struct PySplitCriterion {
    inner: RustSplitCriterion,
}

#[pymethods]
impl PySplitCriterion {
    #[staticmethod]
    fn gini() -> Self {
        PySplitCriterion { inner: RustSplitCriterion::Gini }
    }

    #[staticmethod]
    fn entropy() -> Self {
        PySplitCriterion { inner: RustSplitCriterion::Entropy }
    }

    #[staticmethod]
    fn mse() -> Self {
        PySplitCriterion { inner: RustSplitCriterion::Mse }
    }

    #[staticmethod]
    fn random() -> Self {
        PySplitCriterion { inner: RustSplitCriterion::Random }
    }

    #[staticmethod]
    fn random_projection() -> Self {
        PySplitCriterion { inner: RustSplitCriterion::RandomProjection }
    }


    fn __repr__(&self) -> String {
        match self.inner {
            RustSplitCriterion::Gini => "SplitCriterion.GINI".to_string(),
            RustSplitCriterion::Entropy => "SplitCriterion.ENTROPY".to_string(),
            RustSplitCriterion::Mse => "SplitCriterion.MSE".to_string(),
            RustSplitCriterion::Random => "SplitCriterion.RANDOM".to_string(),
            RustSplitCriterion::RandomProjection => "SplitCriterion.RANDOM_PROJECTION".to_string(),
        }
    }
}

#[pyclass(name = "TreeConfig")]
#[derive(Clone)]
pub struct PyTreeConfig {
    pub inner: RustTreeConfig,
}

#[pymethods]
impl PyTreeConfig {
    #[new]
    #[pyo3(signature = (
        task_type,
        n_classes=2,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        criterion=None,
        seed=0,
        projection_type=None,
        projection_density=None,
    ))]
    fn new(
        task_type: PyTaskType,
        n_classes: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        criterion: Option<PySplitCriterion>,
        seed: u64,
        projection_type: Option<String>,
        projection_density: Option<f64>,
    ) -> Self {
        let criterion = criterion.map(|c| c.inner).unwrap_or_else(|| {
            match task_type.inner {
                RustTaskType::Classification => RustSplitCriterion::Gini,
                RustTaskType::Regression => RustSplitCriterion::Mse,
                RustTaskType::AnomalyDetection => RustSplitCriterion::Random,
            }
        });

        let projection_type = if criterion == RustSplitCriterion::RandomProjection {
            Some(match projection_type.as_deref().unwrap_or("gaussian") {
                "sparse" => ProjectionType::Sparse(projection_density.unwrap_or(0.3)),
                _ => ProjectionType::Gaussian,
            })
        } else {
            None
        };

        PyTreeConfig {
            inner: RustTreeConfig {
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                criterion,
                task_type: task_type.inner,
                n_classes,
                seed,
                projection_type,
            }
        }
    }

    #[staticmethod]
    fn classification(n_classes: usize) -> Self {
        PyTreeConfig {
            inner: RustTreeConfig::classification(n_classes),
        }
    }

    #[staticmethod]
    fn regression() -> Self {
        PyTreeConfig {
            inner: RustTreeConfig::regression(),
        }
    }

    #[staticmethod]
    fn isolation(max_samples: usize) -> Self {
        PyTreeConfig {
            inner: RustTreeConfig::isolation(max_samples),
        }
    }

    #[getter]
    fn max_depth(&self) -> Option<usize> {
        self.inner.max_depth
    }

    #[setter]
    fn set_max_depth(&mut self, value: Option<usize>) {
        self.inner.max_depth = value;
    }

    #[getter]
    fn min_samples_split(&self) -> usize {
        self.inner.min_samples_split
    }

    #[setter]
    fn set_min_samples_split(&mut self, value: usize) {
        self.inner.min_samples_split = value;
    }

    #[getter]
    fn min_samples_leaf(&self) -> usize {
        self.inner.min_samples_leaf
    }

    #[setter]
    fn set_min_samples_leaf(&mut self, value: usize) {
        self.inner.min_samples_leaf = value;
    }

    #[getter]
    fn max_features(&self) -> Option<usize> {
        self.inner.max_features
    }

    #[setter]
    fn set_max_features(&mut self, value: Option<usize>) {
        self.inner.max_features = value;
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[setter]
    fn set_seed(&mut self, value: u64) {
        self.inner.seed = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeConfig(task_type={:?}, max_depth={:?}, min_samples_split={}, min_samples_leaf={}, max_features={:?}, criterion={:?}, seed={})",
            self.inner.task_type,
            self.inner.max_depth,
            self.inner.min_samples_split,
            self.inner.min_samples_leaf,
            self.inner.max_features,
            self.inner.criterion,
            self.inner.seed
        )
    }
}

#[pyclass(name = "Tree")]
pub struct PyTree {
    pub inner: RustTree,
}

#[pymethods]
impl PyTree {
    #[staticmethod]
    fn fit(
        config: PyTreeConfig,
        data: ArrayLike,
        labels: ArrayLike,
        n_samples: usize,
        n_features: usize,
    ) -> PyResult<Self> {
        let data_arr = data.into_ndarray()?;
        let labels_arr = labels.into_ndarray()?;

        if data_arr.shape().size() != n_samples * n_features {
            return Err(PyValueError::new_err(format!(
                "Data size {} doesn't match n_samples * n_features ({})",
                data_arr.shape().size(),
                n_samples * n_features
            )));
        }

        let data_slice = data_arr.as_slice();
        let labels_slice = labels_arr.as_slice();

        let mut builder = TreeBuilder::new(
            &config.inner,
            data_slice,
            labels_slice,
            n_samples,
            n_features,
        );

        Ok(PyTree {
            inner: builder.build(),
        })
    }

    fn predict<'py>(&self, py: Python<'py>, data: ArrayLike, n_samples: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_arr = data.into_ndarray()?;
        let data_slice = data_arr.as_slice();

        let predictions = self.inner.predict_values(data_slice, n_samples);
        Ok(predictions.into_pyarray(py))
    }

    fn predict_anomaly_scores<'py>(&self, py: Python<'py>, data: ArrayLike, n_samples: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_arr = data.into_ndarray()?;
        let data_slice = data_arr.as_slice();

        let scores = self.inner.predict_anomaly_scores(data_slice, n_samples);
        Ok(scores.into_pyarray(py))
    }

    fn predict_path_lengths<'py>(&self, py: Python<'py>, data: ArrayLike, n_samples: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_arr = data.into_ndarray()?;
        let data_slice = data_arr.as_slice();

        let path_lengths = self.inner.predict_path_lengths(data_slice, n_samples);
        Ok(path_lengths.into_pyarray(py))
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    #[getter]
    fn max_depth_reached(&self) -> usize {
        self.inner.max_depth_reached
    }

    #[getter]
    fn n_training_samples(&self) -> usize {
        self.inner.n_training_samples
    }

    fn __repr__(&self) -> String {
        format!(
            "Tree(n_nodes={}, n_features={}, max_depth={}, task_type={:?})",
            self.inner.n_nodes(),
            self.inner.n_features,
            self.inner.max_depth_reached,
            self.inner.task_type
        )
    }
}

#[pyclass(name = "EnsembleConfig")]
#[derive(Clone)]
pub struct PyEnsembleConfig {
    pub inner: RustEnsembleConfig,
}

#[pymethods]
impl PyEnsembleConfig {
    #[new]
    #[pyo3(signature = (
        n_trees,
        tree_config,
        bootstrap=true,
        max_samples=None,
        seed=42
    ))]
    fn new(
        n_trees: usize,
        tree_config: PyTreeConfig,
        bootstrap: bool,
        max_samples: Option<usize>,
        seed: u64,
    ) -> Self {
        PyEnsembleConfig {
            inner: RustEnsembleConfig {
                n_trees,
                tree_config: tree_config.inner,
                bootstrap,
                max_samples,
                seed,
            }
        }
    }

    #[staticmethod]
    fn random_forest_classifier(n_trees: usize, n_classes: usize) -> Self {
        PyEnsembleConfig {
            inner: RustEnsembleConfig::random_forest_classifier(n_trees, n_classes),
        }
    }

    #[staticmethod]
    fn random_forest_regressor(n_trees: usize) -> Self {
        PyEnsembleConfig {
            inner: RustEnsembleConfig::random_forest_regressor(n_trees),
        }
    }

    #[staticmethod]
    fn isolation_forest(n_trees: usize, max_samples: usize) -> Self {
        PyEnsembleConfig {
            inner: RustEnsembleConfig::isolation_forest(n_trees, max_samples),
        }
    }

    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees
    }

    #[setter]
    fn set_n_trees(&mut self, value: usize) {
        self.inner.n_trees = value;
    }

    #[getter]
    fn bootstrap(&self) -> bool {
        self.inner.bootstrap
    }

    #[setter]
    fn set_bootstrap(&mut self, value: bool) {
        self.inner.bootstrap = value;
    }

    #[getter]
    fn max_samples(&self) -> Option<usize> {
        self.inner.max_samples
    }

    #[setter]
    fn set_max_samples(&mut self, value: Option<usize>) {
        self.inner.max_samples = value;
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[setter]
    fn set_seed(&mut self, value: u64) {
        self.inner.seed = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "EnsembleConfig(n_trees={}, bootstrap={}, max_samples={:?}, seed={})",
            self.inner.n_trees,
            self.inner.bootstrap,
            self.inner.max_samples,
            self.inner.seed
        )
    }
}


#[pyclass(name = "Ensemble")]
pub struct PyEnsemble {
    pub inner: RustEnsemble,
}

#[pymethods]
impl PyEnsemble {
    #[staticmethod]
    fn fit(
        config: PyEnsembleConfig,
        data: ArrayLike,
        labels: ArrayLike,
        n_samples: usize,
        n_features: usize,
    ) -> PyResult<Self> {
        let data_arr = data.into_ndarray()?;
        let labels_arr = labels.into_ndarray()?;

        if data_arr.shape().size() != n_samples * n_features {
            return Err(PyValueError::new_err(format!(
                "Data size {} doesn't match n_samples * n_features ({})",
                data_arr.shape().size(),
                n_samples * n_features
            )));
        }

        let data_slice = data_arr.as_slice();
        let labels_slice = labels_arr.as_slice();

        Ok(PyEnsemble {
            inner: RustEnsemble::fit(
                config.inner,
                data_slice,
                labels_slice,
                n_samples,
                n_features,
            ),
        })
    }

    fn predict<'py>(&self, py: Python<'py>, data: ArrayLike, n_samples: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_arr = data.into_ndarray()?;
        let data_slice = data_arr.as_slice();

        let predictions = self.inner.predict(data_slice, n_samples);
        Ok(predictions.into_pyarray(py))
    }

    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }

    #[getter]
    fn n_training_samples(&self) -> usize {
        self.inner.n_training_samples
    }

    fn __repr__(&self) -> String {
        format!(
            "Ensemble(n_trees={}, n_training_samples={})",
            self.inner.n_trees(),
            self.inner.n_training_samples
        )
    }
}


pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTaskType>()?;
    m.add_class::<PySplitCriterion>()?;
    m.add_class::<PyTreeConfig>()?;
    m.add_class::<PyTree>()?;
    m.add_class::<PyEnsembleConfig>()?;
    m.add_class::<PyEnsemble>()?;
    Ok(())
}
