# IronForest
Fast spatial indexing and approximate density estimation for Python, powered by Rust. Zero external dependencies.

### Deprecation Note
Decision Tree, Isolation Forest & Random Forest are deprecated for future releases. They do not sit in line with the core idea of this library and position us as a "scikit-learn alternative" which is not the space we'd like to be. The core focus will be on spatial data structures and models going forwards.

## Quickstart

Quickly find the k nearest neighbours in high-dimensional space using random projections and our KDTree.

```python
import ironforest as irn

dims = 256
n = 100_000
k = 10

#Randomly generate data
gen = irn.random.Generator.from_seed(0)
data = gen.uniform(0.0, 100.0, [n, dims])
query_point = ([50.0] * dims)

#Use random projections to reduce the dimensionality
reducer, points = irn.spatial.ProjectionReducer.fit_transform(data, 50)
tree = irn.spatial.KDTree.from_array(points, leaf_size=50)

#Use our reducer to convert the query
query_point = reducer.transform(query_point)
result = tree.query_knn(query_point, k=k)

#k nearest neighbours
for result_idx, original_idx in enumerate(result.indices):
    print(f"index: {original_idx}, dist: {result.distances[result_idx]}")

#print mean, meadian and max distances
print(f"{result.mean():.2f}, {result.median():.2f}, {result.radius():.2f}")
```

## Installation

From PyPi: `pip install ironforest`

You can also build with `maturin build --release` assuming maturin is and rust are installed.

## Status

The main things I need to finish before 1.0 are improving compatibility with the wider python ecosystem and improving the robustness of my algorithms. Main goals are:
- Adding buffer protocols (DONE)
- Integration with pandas and polars (DONE)
- f32 intergration (DONE)
- NaN handling
- Spatial & RPForest objects

I'd also like to highlight the fact that before 1.0, serialization will not be gauranteed across versions. This is because the underlying trees are still going through a fair amount of iteration as we are still in the early stages of this library. 

## Spatial
Spatial trees support kNN, radius, and KDE queries. All spatial trees support serialization via `save()` & `load()`, alternatively you can use pickle. Trees can be constructed and queried with our own array, numpy, pandas or polars. Anything that implements the python buffer protocol should be a valid input, although f64 values are only accepted at this point and NaN and non numeric types are not handled.

- KDTree - axis-aligned splits, best for low-to-moderate dimensions
- BallTree - pivot-based splits, handles higher dimensions well
- VPTree - vantage-point splits, strong in general metric spaces
- RPTree - random-projection splits, strong in high dimensions with low intrinsic dimensionality. 
- MTree (SOON) - pivot-based splits, supports dynamic insertion at the cost of query speed.
- AggTree - approximate KDE via aggregated nodes, tunable accuracy via atol
- ProjectionReducer - use random projections to reduce dimensionality for more effecient spatial queries.

___

Speed comparison of our KDTree vs SciPy & Scikit-Learn on a randomly generated uniform dataset. KDE is not exposed directly on either of their trees, but `algorithm="kd_tree"` was specified for scikit-learn's `KernelDensity` object for the below comparison. More comprehensive bencmarks can be found at docs/spatial.md

<div align="center">

| Dataset | Structure | Build (s) | kNN (s) | Radius (s) | KDE (s) | k | radius | bandwidth |
|---|---|---|---|---|---|---|---|---|
| 50000x8 | sklearn KDTree | 0.048036 | 0.367485 | 1.977652 | 15.304560 | 10 | 0.5 | 0.5 |
| 50000x8 | scipy KDTree | 0.011791 | 0.152537 | 1.146680 | N/A | 10 | 0.5 | 0.5 |
| 50000x8 | irn KDTree | 0.010846 | 0.021540 | 0.093523 | 0.719237 | 10 | 0.5 | 0.5 |

</div>

KDTree is generally seen as the baseline spatial indexing tree. Our other trees scale better with dimensionality or provide better results depending on the nature of the dataset used, see `docs/spatial.md` & `docs/agg_tree.md` for more detailed information.

## Models
IronForest also includes a variety of models. Many of these leverage our spatial trees or other parts of our exsiting infrastructure but they aren't our core focus.

- Decision Trees
- Random Forest
- Isolation Forest
- Linear Regression
- Local Regression
- KNN Regression & Classification

## Supporting Modules

These modules are not the primary focus of the library but we still expose them through our python bindings. Numpy and SciPy should be preffered if a wider array of linear algebra and statistical methods are needed, all our functions support `ArrayLike` inputs, which can be numpy NdArrays, python lists or our own internal Array object. Our arrays can also be converted to alternative  formats via to_numpy() & tolist() for display and use alongside other libraries. We also support the numpy array protocol.

### Array
- An N-dimensional array object with broadcasting
- Matrix operations & constructors
- Zero copy interop with contigious numpy arrays via `to_numpy()` & `from_numpy()`

### NdUtils
- Array constructors
- numpy conversions
- Column stack (additional stack methods planned)
- Linspace

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.
- Least Squares & Weighted Least Squares solver.

### Stats
- Basic statistical methods for `Array` objects, mean, median, var std, and quantile.
- Pearson and Spearman correlation.

## Rationale

This project very much started out as a learning project, which is primarily why we dip into many niches that are already covered within the python ecosystem. Going forward however the focus is on our spatial & models modules. IronForest will never be a fully fledged alternative to the likes of scikit-learn, nor will it ever be competitve with some of the state of the art aNN libraries for high speed aNN queries on large datasets. 

What we will provide is an easy to use API alongside high performance while maintaing zero dependencies. Currently our spatial module is in a fairly good spot, but our models can and will be improved in this regard. Any bugs reports would be much appreciated and I'm open to feature requests that align with the above stated goals.


