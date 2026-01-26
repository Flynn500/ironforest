import time
import substratum as ss
import numpy as np
from sklearn.neighbors import KDTree, BallTree


def benchmark_tree(
    n_points=100_000,
    n_queries=500,
    dim=2,
    leaf_size=32,
    bandwidth=0.5,
    runs=5,
    seed=42,
):
    gen = ss.random.Generator.from_seed(seed)

    points = gen.uniform(0.0, 1.0, [n_points, dim])
    queries = gen.uniform(0.0, 1.0, [n_queries, dim])

    build_times = []
    query_times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        tree = ss.spatial.KDTree.from_array(
            points,
            leaf_size=leaf_size,
            metric="euclidean",
        )
        t1 = time.perf_counter()
        build_times.append(t1 - t0)

        t0 = time.perf_counter()
        _ = tree.kernel_density_approx(
            queries,
            bandwidth=bandwidth,
            kernel="gaussian",
            criterion="min_samples",
            min_samples=100
        )
        # _ = tree.kernel_density(
        #     queries,
        #     bandwidth=bandwidth,
        #     kernel="gaussian",
        # )
        t1 = time.perf_counter()
        query_times.append(t1 - t0)

    return {
        "build_min": min(build_times),
        "build_mean": sum(build_times) / runs,
        "query_min": min(query_times),
        "query_mean": sum(query_times) / runs,
    }

if __name__ == "__main__":
    result = benchmark_tree(
        n_points=100_000,
        n_queries=500,
        dim=2,
        leaf_size=32,
        runs=10,
    )

    print(" benchmark results:")
    for k, v in result.items():
        print(f"{k:>12}: {v:.6f} sec")

"""

SKlearn vs substratum
KDE Query - Gaussian, Euclidian
n_points=100_000,
n_queries=500

substratum KDTree benchmark results:
   build_min: 0.012436 sec
  build_mean: 0.014963 sec
   query_min: 0.607634 sec
  query_mean: 0.635967 sec

substratum benchmark results:
   build_min: 0.019365 sec
  build_mean: 0.022635 sec
   query_min: 0.607534 sec
  query_mean: 0.639607 sec

substratum KDTree approx benchmark results: min_samples = 100 
   build_min: 0.013954 sec
  build_mean: 0.014673 sec
   query_min: 0.000024 sec
  query_mean: 0.000034 sec

substratum BallTree approx benchmark results: min_samples = 100
   build_min: 0.019431 sec
  build_mean: 0.020502 sec
   query_min: 0.000026 sec
  query_mean: 0.000040 sec


sklearn KDTree benchmark results:
   build_min: 0.038591 sec
  build_mean: 0.040955 sec
   query_min: 0.929484 sec
  query_mean: 0.986913 sec

sklearn BallTree benchmark results:
   build_min: 0.037026 sec
  build_mean: 0.038487 sec
   query_min: 0.948515 sec
  query_mean: 0.996769 sec


KDE Query - Gaussian, Euclidian
n_points=200_000,
n_queries=1000,

old cache hating implementation:

KDTree benchmark results:
    build_min: 0.035546 sec
    build_mean: 0.037652 sec
    query_min: 3.458297 sec
    query_mean: 3.813541 sec

BallTree benchmark results:
    build_min: 0.050904 sec
    build_mean: 0.053747 sec
    query_min: 3.484985 sec
    query_mean: 3.655839 sec

VPTree benchmark results:
    build_min: 0.035535 sec
    build_mean: 0.037596 sec
    query_min: 3.428210 sec
    query_mean: 3.515841 sec


new cache friendly implementation:

KDTree benchmark results:
    build_min: 0.033573 sec
    build_mean: 0.035239 sec
    query_min: 2.342033 sec
    query_mean: 2.391603 sec

BallTree benchmark results:
    build_min: 0.043306 sec
    build_mean: 0.046842 sec
    query_min: 2.473859 sec
    query_mean: 2.506054 sec

VPTree benchmark results:
    build_min: 0.034412 sec
    build_mean: 0.035934 sec
    query_min: 2.354319 sec
    query_mean: 2.403241 sec
"""