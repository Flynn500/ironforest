import time
import ironforest as irn
import numpy as np
from sklearn.neighbors import KDTree, BallTree
from sklearn.datasets import make_blobs

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
            kernel="gaussian",
            bandwidth=0.5,
            atol=0.1
        )
        t1 = time.perf_counter()
        build_times.append(t1 - t0)

        t0 = time.perf_counter()

        _ = tree.kernel_density(
            queries,
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
        tree = irn.spatial.BallTree.from_array(
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
        tree = BallTree(
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
    rng = np.random.default_rng(101)

    all_data, _ = make_blobs(n_samples=101000, centers=20, cluster_std=0.05, n_features=3, random_state=101) # type: ignore
    points_np = all_data[:100000]
    queries_np = all_data[100000:]

    points_irn = irn.ndutils.from_numpy(points_np)
    queries_irn = irn.ndutils.from_numpy(queries_np)

    t0 = time.perf_counter()
    agg_result = irn.spatial.AggTree.from_array(points_irn,leaf_size=32,metric="euclidean",kernel="gaussian",bandwidth=0.5,atol=1e-4) \
        .kernel_density(queries_irn, normalize=True)
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
