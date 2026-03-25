import numpy as np
import time
import ironforest as irn
from ironforest import spatial
import matplotlib.pyplot as plt

# --- Config ---
LEAF_SIZE = 200
N_POINTS = 50_000
N_QUERIES = 513
K = 20
DIMS = [8, 16, 32, 64, 128, 256]

# ANN params
N_CANDIDATES_LIST= [K, K*2, K*4, K*8, K*64, K*256, K*512]


# Stochastic params
N_PROBES_LIST = [2,3,5]


TREES = {
    #"KDTree": lambda d, ls: spatial.KDTree(d, leaf_size=ls),
    # "BallTree": lambda d, ls: spatial.BallTree(d, leaf_size=ls),
    "RPTree": lambda d, ls: spatial.RPTree(d, leaf_size=ls),
}


def brute_force_knn(data, queries, k):
    bf = irn.spatial.BruteForce(data, copy=False)
    return bf.query_knn(queries, k).indices


def compute_recall(ref_indices, test_indices, k):
    """Average recall: fraction of true neighbors found per query."""
    n_queries = ref_indices.shape[0]
    total = 0.0
    for q in range(n_queries):
        ref_set = set(ref_indices[q])
        test_set = set(test_indices[q])
        total += len(ref_set & test_set) / k
    return total / n_queries


def time_query(fn, n_runs=3):
    """Run fn multiple times, return median elapsed time."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.median(times), result


def benchmark_dim(dim, data, queries, ref_indices):
    irn_queries = irn.ndutils.from_numpy(queries)
    rows = []

    for tree_name, builder in TREES.items():
        tree = builder(data, LEAF_SIZE)

        # Deterministic ANN
        for nc in N_CANDIDATES_LIST:
            elapsed, result = time_query(
                lambda nc=nc: tree.query_ann(irn_queries, K, n_candidates=nc)
            )
            test_idx = np.array(irn.ndutils.to_numpy(result.indices)).reshape(-1, K)
            recall = compute_recall(ref_indices, test_idx, K)
            rows.append({
                "tree": tree_name,
                "method": "deterministic",
                "n_candidates": nc,
                "n_probes": "-",
                "score": recall / np.log(elapsed * 1000),
                "recall": recall,
                "time_ms": elapsed * 1000,
            })

        # Stochastic ANN
        for nc in N_CANDIDATES_LIST:
            for np_ in N_PROBES_LIST:
                tau=0
                elapsed, result = time_query(
                    lambda nc=nc, np_=np_, tau=tau: tree.query_ann(
                        irn_queries, K,
                        n_candidates=nc,
                        n_probes=np_,
                    )
                )
                test_idx = np.array(irn.ndutils.to_numpy(result.indices)).reshape(-1, K)
                recall = compute_recall(ref_indices, test_idx, K)
                rows.append({
                    "tree": tree_name,
                    "method": "stochastic",
                    "n_candidates": nc,
                    "n_probes": np_,
                    "score": recall / np.log(elapsed * 1000),
                    "recall": recall,
                    "time_ms": elapsed * 1000,
                })

    return rows


def print_results(dim, rows):
    print(f"\n{'=' * 90}")
    print(f"  dim={dim}  |  N={N_POINTS}  |  queries={N_QUERIES}  |  k={K}")
    print(f"{'=' * 90}")
    header = f"{'Tree':<10} {'Method':<14} {'n_cand':>6} {'probes':>6} {'score':>5} {'Recall':>7} {'Time(ms)':>9}"
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda x: (-x["score"], x["time_ms"], -x["recall"])):
        print(
            f"{r['tree']:<10} {r['method']:<14} {str(r['n_candidates']):>6} "
            f"{str(r['n_probes']):>6} {r['score']:>5.2f} "
            f"{r['recall']:>7.3f} {r['time_ms']:>9.2f}"
        )

def make_blobs(n_points, n_queries, dim, n_clusters=10, cluster_std=0.05, seed=42):
    rng = np.random.default_rng(seed)

    centers = rng.uniform(0, 1, size=(n_clusters, dim))

    point_clusters = rng.integers(0, n_clusters, size=n_points)
    query_clusters = rng.integers(0, n_clusters, size=n_queries)

    data = centers[point_clusters] + rng.normal(0, cluster_std, size=(n_points, dim))
    queries = centers[query_clusters] + rng.normal(0, cluster_std, size=(n_queries, dim))

    return data.astype(np.float64), queries.astype(np.float64)

def plot_pareto(dim, rows):
    """Plot recall vs time for deterministic vs stochastic methods."""
    fig, ax = plt.subplots(figsize=(8,5))
    
    for method in ["deterministic", "stochastic"]:
        method_rows = [r for r in rows if r["method"] == method]
        times = [r["time_ms"] for r in method_rows]
        recalls = [r["recall"] for r in method_rows]
        ax.scatter(times, recalls, label=method, alpha=0.7)
    
    ax.set_title(f"ANN Recall vs Query Time (dim={dim})")
    ax.set_xlabel("Query Time (ms)")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{dim}pareto.png")

def run_benchmarks_blobs():
    print("=== ANN Benchmark (Clustered Blobs) ===")

    for dim in DIMS:
        n_clusters = max(5, dim // 2)

        data, queries = make_blobs(
            N_POINTS,
            N_QUERIES,
            dim,
            n_clusters=n_clusters,
            cluster_std=0.05,
            seed=42
        )

        ref_indices = brute_force_knn(data, queries, K)

        rows = benchmark_dim(dim, data, queries, ref_indices)
        print_results(dim, rows)

    print("\n=== Done (Blobs) ===")
    



def run_benchmarks():
    rng = np.random.default_rng(42)
    print("=== ANN Benchmark: Deterministic vs Stochastic ===")

    for dim in DIMS:
        data = rng.uniform(0,1,(N_POINTS, dim)).astype(np.float64)
        queries = rng.uniform(0,1,(N_QUERIES, dim)).astype(np.float64)
        ref_indices = brute_force_knn(data, queries, K)

        rows = benchmark_dim(dim, data, queries, ref_indices)
        print_results(dim, rows)

    print("\n=== Done ===")


if __name__ == "__main__":
    run_benchmarks_blobs()
    