# Substratum
Substratum is a rust package designed for the dubious python library. Dubious was a project with the personal constraint that no libraries can be used other than numpy. Substratum is my attempt to rewrite some of numpy's features to replace numpy as dubious's lone dependency. I also intend to expand Substratum to include more features, although it is not intended to be a full fledged numpy replacement.

The core rational behind this library was to get a better grasp of how libraries I use on a regular basis work, and learn how to write python bindings to offload computationally expensive tasks to tools better suited. I chose rust because I have read the book around a year prior to this project, and pyo3 bindings are relitively easy to get working.

## Status
This is largely a learning project and the API is subject to change. We achieve similar performance (beating them in a rare few) to numpy across most operations but basic broadcasting arithmetic is around 4x slower in most cases. I intend to add a few unsafe methods to speed things up where applicable but I don't intend to optimize much further at this stage. 

## Installation
A PyPi release is planned soon, for now you can build with `maturin build --release` assuming maturin is installed.

## Quickstart

```python
import substratum as ss

a = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
b = ss.Array([2, 2], [5.0, 6.0, 7.0, 8.0])

print(f"a @ b = {(a @ b).tolist()}")
```
Output: 
`a @ b = [19.0, 22.0, 43.0, 50.0]`

## Features
- Array, an N-dimensional array object. 
- Broadcast operations
- trig methods
- Statistical methods (mean, var, quantile)
- matrix methods and constructors
- cholesky and eigen decomposition
  - Ball Tree, knn and radius queries

### Top-level
- `substratum.Array`

### Modules
- `substratum.linalg`
- `substratum.stats`
- `substratum.random`
- `substratum.spatial`

## Examples
```python
import substratum as ss

a = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
b = ss.Array([2, 2], [5.0, 6.0, 7.0, 8.0])

print(f"a + b = {(a + b).tolist()}")
print(f"a * b = {(a * b).tolist()}")
```
Output: 
`
a + b = [6.0, 8.0, 10.0, 12.0]
a * b = [5.0, 12.0, 21.0, 32.0]
`
```python
import substratum as ss

gen = ss.Generator.from_seed(123)

uniform = gen.uniform(0.0, 1.0, [2, 3])
print(f"Uniform [0, 1): {uniform.tolist()}")

normal = gen.standard_normal([2, 3])
print(f"Standard normal: {normal.tolist()}")
```
Output: 
`Uniform [0, 1): [0.19669435215621578, 0.9695722925002218, 0.46744032361670884, 0.12698379756585432]
Standard normal: [-0.0008585765206425146, 1.4733334715623352, -1.16180050645278, -0.772101732825336]`

## Modules

### Random
`Generator` object that can sample from uniform, normal, lognormal, gamma and beta distributions. Support for additional distributions is planned.

### Linalg
- Standard matrix methods and constructors.
- cholesky and eigen and qr decomposition.

### Stats
- Basic statistical methods for `Array` objects, mean, var and quantile.
- Pearson and Spearman correlation.

### Spatial
`BallTree` Ball tree object with a kNN and radius queries.