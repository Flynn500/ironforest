import time
import ironforest as irn
import numpy as np
from sklearn.neighbors import KDTree, BallTree

def benchmark_tree_aggtree(
    n_points=100_000,
    n_queries=500,
    dim=2,
    leaf_size=32,
    bandwidth=0.5,
    runs=5,
    seed=42,
):
    gen = irn.random.Generator.from_seed(seed)

    points = gen.uniform(0.0, 1.0, [n_points, dim])
    queries = gen.uniform(0.0, 1.0, [n_queries, dim])

    build_times = []
    query_times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        tree = irn.spatial.AggTree.from_array(
            points,
            leaf_size=leaf_size,
            metric="euclidean",
            max_span=1.75
        )
        t1 = time.perf_counter()
        build_times.append(t1 - t0)

        t0 = time.perf_counter()

        _ = tree.kernel_density(
            queries,
            bandwidth=bandwidth,
            kernel="gaussian",
        )
        t1 = time.perf_counter()
        query_times.append(t1 - t0)

    return {
        "build_min": min(build_times),
        "build_mean": sum(build_times) / runs,
        "query_min": min(query_times),
        "query_mean": sum(query_times) / runs,
    }

def benchmark_tree_irn_kdtree(
    n_points=100_000,
    n_queries=500,
    dim=2,
    leaf_size=32,
    bandwidth=0.5,
    runs=5,
    seed=42,
):
    gen = irn.random.Generator.from_seed(seed)
    
    points = gen.uniform(0.0, 1.0, [n_points, dim])
    queries = gen.uniform(0.0, 1.0, [n_queries, dim])

    build_times = []
    query_times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        tree = irn.spatial.KDTree.from_array(
            points,
            leaf_size=leaf_size,
            metric="euclidean",
        )
        t1 = time.perf_counter()
        build_times.append(t1 - t0)

        t0 = time.perf_counter()
        _ = tree.kernel_density(
            queries,
            bandwidth=bandwidth,
            kernel="gaussian",
        )
        t1 = time.perf_counter()
        query_times.append(t1 - t0)

    return {
        "build_min": min(build_times),
        "build_mean": sum(build_times) / runs,
        "query_min": min(query_times),
        "query_mean": sum(query_times) / runs,
    }

def benchmark_tree_sklearn(
    n_points=100_000,
    n_queries=500,
    dim=2,
    leaf_size=32,
    bandwidth=0.5,
    runs=5,
    seed=42,
):
    rng = np.random.default_rng(seed)

    points = rng.uniform(0.0, 1.0, size=(n_points, dim))
    queries = rng.uniform(0.0, 1.0, size=(n_queries, dim))

    build_times = []
    query_times = []

    for _ in range(runs):
        # Build
        t0 = time.perf_counter()
        tree = KDTree(
            points,
            leaf_size=leaf_size,
            metric="euclidean",
        )
        t1 = time.perf_counter()
        build_times.append(t1 - t0)

        # Query (exact KDE)
        t0 = time.perf_counter()
        _ = tree.kernel_density(
            queries,
            h=bandwidth,
            kernel="gaussian",
        )
        t1 = time.perf_counter()
        query_times.append(t1 - t0)

    return {
        "build_min": min(build_times),
        "build_mean": sum(build_times) / runs,
        "query_min": min(query_times),
        "query_mean": sum(query_times) / runs,
    }


if __name__ == "__main__":

    print("\n results comparison (single run):")

    rng = np.random.default_rng(42)
    points_np = rng.uniform(0.0, 1.0, size=(100000, 4))
    queries_np = rng.uniform(0.0, 1.0, size=(20, 4))

    points_irn = irn.Array.from_numpy(points_np)
    queries_irn = irn.Array.from_numpy(queries_np)

    t0 = time.perf_counter()
    agg_result = irn.spatial.AggTree.from_array(points_irn, leaf_size=32, metric="euclidean", max_span= 1.5) \
        .kernel_density(queries_irn, bandwidth=0.5, kernel="gaussian", normalize=True)
    t1 = time.perf_counter()
    agg_time = t1-t0

    t0 = time.perf_counter()
    kd_result = irn.spatial.KDTree.from_array(points_irn, leaf_size=32, metric="euclidean") \
        .kernel_density(queries_irn, bandwidth=0.5, kernel="gaussian",normalize = True)
    t1 = time.perf_counter()
    kd_time = t1-t0

    t0 = time.perf_counter()
    sk_result = KDTree(points_np, leaf_size=32, metric="euclidean") \
        .kernel_density(queries_np, h=0.5, kernel="gaussian")
    t1 = time.perf_counter()
    sk_time = t1-t0

    agg = np.array(irn.Array.to_numpy(agg_result)) # type: ignore
    kd  = np.array(irn.Array.to_numpy(kd_result)) # type: ignore
    sk  = np.array(sk_result)
    
    print(f"\n  {'pair':<25} {'max_abs_diff':>14} {'mean_abs_diff':>14}")
    print(f"  {'AggTree vs irn.KDTree':<25} {np.max(np.abs(agg - kd)):>14.6e} {np.mean(np.abs(agg - kd)):>14.6e}")
    print(f"  {'AggTree vs sklearn':<25} {np.max(np.abs(agg - sk)):>14.6e} {np.mean(np.abs(agg - sk)):>14.6e}")
    print(f"  {'irn.KDTree vs sklearn':<25} {np.max(np.abs(kd  - sk)):>14.6e} {np.mean(np.abs(kd  - sk)):>14.6e}")
    print(f"time: SK KD {sk_time} - IRN KD {kd_time} - IRN AGG {agg_time}")
    print(f"\n  per-query values (first 5 queries):")
    print(f"  {'query':<8} {'AggTree':>12} {'irn.KDTree':>12} {'sklearn':>12}")
    for i in range(5):
        print(f"  {i:<8} {agg[i]:>12.6f} {kd[i]:>12.6f} {sk[i]:>12.6f}")




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