import time
import numpy as np
from sklearn.neighbors import KDTree as SkKDTree
from sklearn.neighbors import BallTree as SkBallTree

import ironforest as irn
from ironforest import spatial

def time_call(fn, repeat=3):
    """Return best-of-N timing."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def format_md_table(rows):
    headers = [
        "Dataset",
        "Structure",
        "Build (s)",
        "kNN (s)",
        "Radius (s)",
        "k",
        "radius",
    ]

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        out.append(
            "| {dataset} | {structure} | {build:.6f} | {knn:.6f} | {radius_t:.6f} | {k} | {radius} |".format(
                **r
            )
        )

    return "\n".join(out)


def run_benchmark(
    sizes=(10_000, 50_000),
    dim=8,
    k=10,
    radius=0.5,
    leaf_size=40,
    n_queries=1000,
):
    rng = np.random.default_rng(42)
    rows = []

    for n in sizes:
        print(f"Running size={n}")

        data = rng.random((n, dim))
        queries = rng.random((n_queries, dim))

        irn_data = irn.ndutils.from_numpy(data)

        build = time_call(lambda: SkKDTree(data, leaf_size=leaf_size))
        sk_kd = SkKDTree(data, leaf_size=leaf_size)

        knn_t = time_call(lambda: sk_kd.query(queries, k=k))
        rad_t = time_call(lambda: sk_kd.query_radius(queries, r=radius))

        rows.append(
            dict(
                dataset=f"{n}x{dim}",
                structure="sklearn.KDTree",
                build=build,
                knn=knn_t,
                radius_t=rad_t,
                k=k,
                radius=radius,
            )
        )

        build = time_call(lambda: SkBallTree(data, leaf_size=leaf_size))
        sk_ball = SkBallTree(data, leaf_size=leaf_size)

        knn_t = time_call(lambda: sk_ball.query(queries, k=k))
        rad_t = time_call(lambda: sk_ball.query_radius(queries, r=radius))

        rows.append(
            dict(
                dataset=f"{n}x{dim}",
                structure="sklearn.BallTree",
                build=build,
                knn=knn_t,
                radius_t=rad_t,
                k=k,
                radius=radius,
            )
        )

        build = time_call(
            lambda: spatial.KDTree.from_array(irn_data, leaf_size=leaf_size)
        )
        my_kd = spatial.KDTree.from_array(irn_data, leaf_size=leaf_size)

        knn_t = time_call(lambda: my_kd.query_knn(queries, k))
        rad_t = time_call(lambda: my_kd.query_radius(queries, radius))

        rows.append(
            dict(
                dataset=f"{n}x{dim}",
                structure="irn.KDTree",
                build=build,
                knn=knn_t,
                radius_t=rad_t,
                k=k,
                radius=radius,
            )
        )

        build = time_call(
            lambda: spatial.BallTree.from_array(irn_data, leaf_size=leaf_size)
        )
        my_ball = spatial.BallTree.from_array(irn_data, leaf_size=leaf_size)

        knn_t = time_call(lambda: my_ball.query_knn(queries, k))
        rad_t = time_call(lambda: my_ball.query_radius(queries, radius))

        rows.append(
            dict(
                dataset=f"{n}x{dim}",
                structure="irn.BallTree",
                build=build,
                knn=knn_t,
                radius_t=rad_t,
                k=k,
                radius=radius,
            )
        )

    return rows

if __name__ == "__main__":
    rows = run_benchmark(
        sizes=(10_000, 50_000),
        dim=8,
        k=10,
        radius=0.5,
        leaf_size=40,
        n_queries=5000,
    )

    md = format_md_table(rows)
    print("\n" + md)