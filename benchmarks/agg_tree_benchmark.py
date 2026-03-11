import time
import numpy as np
from sklearn.neighbors import KDTree
import ironforest as irn
from sklearn.datasets import make_blobs

def silvermans_bandwidth(n: int, d: int) -> float:
    return (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))


def benchmark_aggtree(points, queries, bandwidth, leaf_size=32):
    p = irn.ndutils.from_numpy(points)
    q = irn.ndutils.from_numpy(queries)

    t0 = time.perf_counter()
    result = (
        irn.spatial.AggTree.from_array(p, leaf_size=leaf_size, metric="euclidean",
                                        kernel="gaussian", bandwidth=bandwidth, atol=0.01)
        .kernel_density(q, normalize=True)
    )

    t1 = time.perf_counter()

    return np.array(irn.Array.to_numpy(result)), t1 - t0 


def benchmark_sklearn(points, queries, bandwidth, leaf_size=32):
    t0 = time.perf_counter()
    result = (
        KDTree(points, leaf_size=leaf_size, metric="euclidean")
        .kernel_density(queries, h=bandwidth, kernel="gaussian")
    )
    t1 = time.perf_counter()

    return np.array(result), t1 - t0

def gen_spherical_clusters():
    n_blobs = int(n_points * 0.8)
    n_noise = n_points - n_blobs

    blobs, _ = make_blobs(n_samples=n_blobs, centers=10, cluster_std=0.1, n_features=d, random_state=seed) # type: ignore

    blobs = (blobs - blobs.min(axis=0)) / (blobs.max(axis=0) - blobs.min(axis=0))
    noise = rng.uniform(0.0, 1.0, size=(n_noise, d))

    points = np.vstack([blobs, noise])
    rng.shuffle(points)

    queries, _ = make_blobs(n_samples=n_queries, centers=10, cluster_std=0.1, n_features=d, random_state=seed + 1) # type: ignore
    queries = (queries - queries.min(axis=0)) / (queries.max(axis=0) - queries.min(axis=0))
    return points, queries

def gen_elliptical_clusters():
    n_clusters = 12
    n_blobs = int(n_points * 0.8)
    n_noise = n_points - n_blobs

    centers = rng.uniform(0.1, 0.9, size=(n_clusters, d))
    points_per_center = n_blobs // n_clusters
    blob_list = []

    for c in centers:
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        scales = np.exp(rng.uniform(-2.5, -1.0, size=d)) * 0.05
        cov = Q @ np.diag(scales**2) @ Q.T

        blob = rng.multivariate_normal(mean=c, cov=cov, size=points_per_center)
        blob_list.append(blob)

    blobs = np.vstack(blob_list)
    blobs = np.clip(blobs, 0.0, 1.0)

    noise = rng.uniform(0.0, 1.0, size=(n_noise, d))
    points = np.vstack([blobs, noise])
    rng.shuffle(points)

    query_centers = rng.uniform(0.1, 0.9, size=(n_clusters, d))
    points_per_center_q = n_queries // n_clusters
    queries_list = []

    for c in query_centers:
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        scales = np.exp(rng.uniform(-2.5, -1.0, size=d)) * 0.05
        cov = Q @ np.diag(scales**2) @ Q.T

        qblob = rng.multivariate_normal(mean=c, cov=cov, size=points_per_center_q)
        queries_list.append(qblob)

    queries = np.vstack(queries_list)
    queries = np.clip(queries, 0.0, 1.0)

    return points, queries

if __name__ == "__main__":
    n_points = 100_000
    n_queries = 1_000
    dims = [2, 4, 8, 16, 32, 64]
    seed = 42
    rng = np.random.default_rng(2)

    print(f"\n{'dim':>5} {'bw':>8} {'agg_time':>10} {'sk_time':>10} {'speedup':>9} {'max_err%':>10} {'mean_err%':>10}")
    print("-" * 68)

    for d in dims:
        points, queries = gen_spherical_clusters()
        bw = silvermans_bandwidth(n_points, d)

        agg_vals, agg_time = benchmark_aggtree(points, queries, bw)
        sk_vals,  sk_time  = benchmark_sklearn(points, queries, bw)

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.abs(agg_vals - sk_vals) / np.abs(sk_vals)
            rel_err = np.where(np.isfinite(rel_err), rel_err, 0.0)

        max_err  = np.max(rel_err) * 100
        mean_err = np.mean(rel_err) * 100
        speedup  = sk_time / agg_time

        print(f"{d:>5} {bw:>8.4f} {agg_time:>10.3f} {sk_time:>10.3f} {speedup:>8.2f}x {max_err:>9.3f}% {mean_err:>9.3f}%")