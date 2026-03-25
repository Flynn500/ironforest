import time
import numpy as np
import ironforest as irn
from ironforest import spatial
import matplotlib.pyplot as plt
import math

TREES = {
    "KDTree_zero_copy":   lambda d, ls: spatial.KDTree(d, leaf_size=ls, copy=False),
    "KDTree_copy": lambda d, ls: spatial.KDTree(d, leaf_size=ls, copy=True),
}

DIMS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
INTRINISC_DIMS = [2, 2, 4, 4, 8, 8, 16, 16, 32, 64]


def time_call(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def format_md_table(results, dims, tree_names):
    header = "| Dim | " + " | ".join(tree_names) + " |"
    sep    = "|" + "|".join(["---"] * (len(tree_names) + 1)) + "|"

    lines = [header, sep]
    for dim in dims:
        row = f"| {dim} |"
        for name in tree_names:
            t = results[dim].get(name, float("nan"))
            row += f" {t:.6f} |"
        lines.append(row)

    return "\n".join(lines)


def run_benchmark(n=50_000, k=10, leaf_size=100, n_queries=1_000):
    rng = np.random.default_rng(0)
    results = {}
    n_clusters = 20
    cluster_std = 0.05
    for i in range(0, len(DIMS)):
        dim = DIMS[i]
        idim = INTRINISC_DIMS[i]
        print(f"  dim={dim}", flush=True)
        results[dim] = {}

        centers = rng.random((n_clusters, dim))
        labels = rng.integers(0, n_clusters, size=n)
        data = centers[labels] + cluster_std * rng.standard_normal((n, dim))

        query_labels = rng.integers(0, n_clusters, size=n_queries)
        queries = centers[query_labels] + cluster_std * rng.standard_normal((n_queries, dim))

        irn_data = irn.ndutils.from_numpy(data)

        for name, builder in TREES.items():
            try:
                tree  = builder(irn_data, leaf_size)
                if name == "RPTree":
                    knn_t = time_call(lambda: tree.query_ann(queries, k, 1000))
                else: knn_t = time_call(lambda: tree.query_knn(queries, k))
                results[dim][name] = knn_t
            except Exception as e:
                print(f"    {name} failed: {e}")
                results[dim][name] = float("nan")

    return results


if __name__ == "__main__":
    results = run_benchmark()
    md = format_md_table(results, DIMS, list(TREES.keys()))
    print("\n" + md)
