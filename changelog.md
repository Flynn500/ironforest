## 0.6

### Added
- Zero copy interop with python buffer protocol for contigious arrays.
- 
### Changed
- aNN method has improved via the use of stochastic probes, controlled by n_probes. This improves both recall and query speed for most usecases.

### Fixed
- Some models would panic in the rust core, returning a vauge and useless error message. Have added checks so errors are raised explicitly and up front.
- Pickle serialization was broken across most spatial index trees.

## 0.5.2

### Added
- Random projection splits in random forest and decision tree

### Fixed
- Bug in exact kNN for random projection tree
- Modules showing up as classes in type stubs

## 0.5.1

### Added
- Python wrapper kNN Regression & Classification for spatial index trees.
- Additional indexing options for arrays to closer match numpy

### Fixed
- VPTree bug causing it to miss results in kNN & Radius queries.
- Local Regression bug after tree API changes.

## 0.5

### Added
- RP-Tree
- Local Regression
- Projection Reducer object for reducing dimensions
- Serialization for spatial trees
- aNN methods for KD, Ball & RP trees
- Reshape & linspace methods for arrays
- Standard \__init\_\_ methods for spatial trees that take an ArrayLike input
- \_\_array\_\_ protocol for numpy conversion 

### Changed
- kNN & radius spatial queries now return a spatial result object
- ML models now use rust core for ensemble aggregation
- Added SIMD instructions to distance kernel & dot product calculations

### Fixed
- There were several unsafe unwrap scenarios where instead of a python error being reuturned, the rust side would panic

## 0.4

### Added
- Brute Force
- AggTree - KDE approximation tree with faster query speeds and reduced memory usage

### Changed
- Array utility methods moved to ndutils module


### Fixed
- Ball Tree had creation error using pre-reorder get method leading to incorrect construction
- Isolation Forest type stubs were missing

## 0.3

### Added
- Decision Tree
- Random Forest
- Isolation Forest

## 0.2

### Added
- KDTree
- BallTree
- VPTree

## 0.1

### Added
- NdArray
- Random
- Linalg