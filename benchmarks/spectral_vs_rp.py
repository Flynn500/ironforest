import time
import numpy as np
import ironforest as irn
from ironforest import spatial
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
TREES = {
    "RPTree": lambda d, ls: spatial.RPTree.from_array(
        d, leaf_size=ls, projection="sparse"
    ),
    "SpectralTree": lambda d, ls: spatial.SpectralTree.from_array(
        d, leaf_size=ls
    ),
}

N_CANDIDATES = [10, 20, 50, 100, 200, 500, 1000, 2000]

# Fixed dimensionality
DIM = 128
INTRINSIC_DIM = 128


# =========================
# UTILITIES
# =========================
def time_call(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def compute_recall(true_neighbors, approx_neighbors, k):
    true_neighbors = np.asarray(true_neighbors)
    approx_neighbors = np.asarray(approx_neighbors)

    # --- FIX: reshape if flattened ---
    if true_neighbors.ndim == 1:
        true_neighbors = true_neighbors.reshape(-1, k)

    if approx_neighbors.ndim == 1:
        approx_neighbors = approx_neighbors.reshape(-1, k)

    total = 0
    correct = 0

    for t, a in zip(true_neighbors, approx_neighbors):
        t_set = set(t)
        a_set = set(a)

        correct += len(t_set.intersection(a_set))
        total += len(t_set)

    return correct / total


def format_tradeoff_table(results):
    lines = [
        "| Tree | n_candidates | Time (s) | Recall |",
        "|------|--------------|----------|--------|",
    ]

    for name, res in results.items():
        for nc, t, r in zip(
            res["n_candidates"], res["time"], res["recall"]
        ):
            lines.append(f"| {name} | {nc} | {t:.6f} | {r:.4f} |")

    return "\n".join(lines)


def plot_recall_vs_time(results, filename="recall_vs_time.png"):
    plt.figure(figsize=(8, 5))

    for name, res in results.items():
        plt.plot(
            res["time"],
            res["recall"],
            marker="o",
            label=name,
        )

    plt.xlabel("Query time (seconds)")
    plt.ylabel("Recall@k")
    plt.title("Recall vs Query Time")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


# =========================
# BENCHMARK
# =========================
def run_benchmark(
    n=50_000,
    k=10,
    leaf_size=200,
    n_queries=1_000,
):
    rng = np.random.default_rng(0)
    results = {}

    print(f"Generating data (dim={DIM}, intrinsic_dim={INTRINSIC_DIM})")

    # --- Generate low intrinsic dimensional data ---
    low_d_data = rng.uniform(-1, 1, (n, INTRINSIC_DIM))

    # Random orthogonal projection
    q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    projection_matrix = q[:INTRINSIC_DIM, :]

    data = low_d_data @ projection_matrix
    data += 0.01 * rng.standard_normal((n, DIM))

    queries = rng.uniform(-1, 1, (n_queries, INTRINSIC_DIM)) @ projection_matrix

    irn_data = irn.ndutils.from_numpy(data)

    # =========================
    # Ground truth (exact kNN)
    # =========================
    print("Computing ground truth (BruteForce)...", flush=True)
    exact_tree = spatial.BruteForce.from_array(irn_data)
    true_neighbors = exact_tree.query_knn(queries, k).indices


    # =========================
    # Benchmark ANN trees
    # =========================
    for name, builder in TREES.items():
        print(f"\n{name}", flush=True)

        results[name] = {
            "recall": [],
            "time": [],
            "n_candidates": [],
        }

        try:
            tree = builder(irn_data, leaf_size)

            for nc in N_CANDIDATES:
                print(f"  n_candidates={nc}", flush=True)

                def run():
                    return tree.query_ann(queries, k, n_candidates=nc, n_probes=2)

                t = time_call(run)
                approx_neighbors = run().indices

                recall = compute_recall(true_neighbors, approx_neighbors, k)

                results[name]["recall"].append(recall)
                results[name]["time"].append(t)
                results[name]["n_candidates"].append(nc)

        except Exception as e:
            print(f"  {name} failed: {e}")

    return results


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    results = run_benchmark()

    plot_recall_vs_time(results)

    md = format_tradeoff_table(results)
    print("\n" + md)