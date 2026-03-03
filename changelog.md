## 0.5

### Added
- RP-Tree
- M-Tree
- Local Regression
- Projection Reducer object for reducing dimensions
- Serialization for spatial trees
- Reshape & linspace methods for arrays
- Standard __init__ methods for spatial trees that take an ArrayLike input

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