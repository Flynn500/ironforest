# IronForest
IronForest is a rust-powered python library supporting spatial queries, array-based computation and tree-based machine learning. 

I started this project to support my previous python library dubious, a project which had the personal constraint of no external dependecies other than numpy. IronForest started as a way for me to eliminate numpy as dubious's lone dependency but has since grown into a standalone project as I have pivoted towards spatial indexing trees and tree-based models to support some of my other projects.

## Status
This is largely a learning project and the API is subject to change. We achieve similar performance to numpy (beating them in a rare few) across most operations but basic broadcasting arithmetic is around 4x slower in most cases. I intend to add a few unsafe methods to speed things up where applicable but I don't intend to optimize much further at this stage. 

## Installation
`pip install ironforest`

You can also build with `maturin build --release` assuming maturin is installed.

## Quickstart

```python
import ironforest as irn

gen = irn.random.Generator.from_seed(0)
points = gen.uniform(0.0, 100.0, [100, 2])

tree = irn.spatial.VPTree.from_array(points, leaf_size=10)

query_point = [50.0, 50.0]
neighbors = tree.query_knn(query_point, k=5)

print(neighbors)
```
Output: 
`[(28, 7.435055148638418), (34, 7.549053283615578), (3, 8.55705353151589), (43, 8.961446938534534), (45, 9.772273124615534)]`

## Features
- Array, an N-dimensional array object with broadcasting
- matrix operations and constructors
- KDTree, BallTree & VPTree with knn radius KDE and KDE approx queries
- Linear and Local Regression
- Decision trees, Random Forest and Isolation Forest
- Random sampling from uniform, normal lognormal, gamma and beta distributions.
- Statistical methods (mean, median, var, std, quantile, Pearson and Spearman correlation)

## Rationale
The core rational behind this project was to get a better grasp of how libraries I use on a regular basis work, and learn how to write python bindings to offload computationally expensive tasks to tools better suited. Like dubious, I focused on building things from the ground up. I didn't want to glue dependecies together to get something functional, I wanted to understand from input to output how these algorithms worked under the hood. I chose rust because I had read the book around a year prior to starting this project, and pyo3 bindings are relitively easy to get working. This library is only exposed through python as that's where I've actually use its features. I don't intend to package this as a rust create at this stage.

### Top-level
- `ironforest.Array`

### Modules
- [ironforest.linalg](#linalg)
- [ironforest.stats](#stats)
- [ironforest.random](#random)
- [ironforest.spatial](#spatial)
- [ironforest.model](#model)

## Examples
```python
  import ironforest as irn

  gen = irn.random.Generator.from_seed(123)
  X = gen.uniform(0.0, 10.0, [50, 1])
  noise = gen.normal(0.0, 1.0, [50, 1])
  y = X * 2.0 + noise + 5.0

  train_size = 40
  X_train = X[0:train_size]
  y_train = y[0:train_size]
  X_test = X[train_size:50]
  y_test = y[train_size:50]

  model = irn.models.LinearRegression(fit_intercept=True)
  model.fit(X_train, y_train)

  print(f"Fitted coefficients: {model.coef_.}")
  print(f"Fitted intercept: {model.intercept_:.2f}")

  test_score = model.score(X_test, y_test)
  print(f"\nTest R² score: {test_score:.3f}")

  predictions = model.predict(X_test)
  print(f"\nFirst 5 predictions vs actual:")
  for i in range(5):
      pred = predictions[i].item()
      actual = y_test[i].item()
      print(f"  Predicted: {pred:.2f}, Actual: {actual:.2f}")
```
`Fitted coefficients: [[1.9324300205909533]]`  
`Fitted intercept: 5.35`

`Test R² score: 0.985`

`First 5 predictions vs actual:`  
`Predicted: 19.52, Actual: 19.53`  

`Predicted: 5.45, Actual: 4.94`  
`Predicted: 6.20, Actual: 5.26`  
`Predicted: 7.42, Actual: 7.64`  
`Predicted: 15.35, Actual: 14.53`  

## Modules

### Spatial
- `KDTree` kNN, Kernel Density Estimation and radius queries. 
- `BallTree` with kNN, Kernel Density Estimation and radius queries. 
- `VPTree` with kNN, Kernel Density Estimation and radius queries.

### Model
- Linear Regression
- Local Regression (soon)
- Decision Trees
- Random Forest
- Isolation Forest (soon)

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions. Support for additional distributions is planned.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.
- Least Squares & Weighted Least Squares solver.

### Stats
- Basic statistical methods for `Array` objects, mean, var and quantile.
- Pearson and Spearman correlation.
