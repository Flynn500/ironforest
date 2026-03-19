import numpy as np
import ironforest as irn
from ironforest import spatial


TREES = {
    "BruteForce":   lambda d, ls: spatial.BruteForce.from_array(d),
    "KDTree":   lambda d, ls: spatial.KDTree.from_array(d, leaf_size=ls),
    "BallTree":   lambda d, ls: spatial.BallTree.from_array(d, leaf_size=ls),
    #"MTree":   lambda d, ls: spatial.MTree.from_array(d, capacity=ls),
    "VPTree":   lambda d, ls: spatial.VPTree.from_array(d, leaf_size=ls, selection="variance"),
    "RPTree":   lambda d, ls: spatial.RPTree.from_array(d, leaf_size=ls),
}

LEAF_SIZE  = 20
N_POINTS   = 2000
N_QUERIES  = 100
K          = 10
RADIUS     = 0.4
DIMS       = [2,4,8,16,32]


def bf_knn(data: np.ndarray, queries: np.ndarray, k: int):
    """Returns (indices, distances) arrays, shape (n_queries, k), sorted by distance."""
    diffs = data[None, :, :] - queries[:, None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    idx   = np.argsort(dists, axis=1)[:, :k]
    return idx, np.take_along_axis(dists, idx, axis=1)


def bf_radius(data: np.ndarray, queries: np.ndarray, r: float):
    """Returns list of (indices, distances) per query."""
    diffs = data[None, :, :] - queries[:, None, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    results = []
    for row in dists:
        mask = row <= r
        idx  = np.where(mask)[0]
        results.append((idx, row[idx]))
    return results


PassFail = dict[str, bool]

def check_knn(data: np.ndarray, queries: np.ndarray, k: int) -> PassFail:
    """Compare each tree's kNN results against brute-force reference."""
    irn_data    = irn.ndutils.from_numpy(data)
    irn_queries = irn.ndutils.from_numpy(queries)

    ref_idx, ref_dists = bf_knn(data, queries, k)
    results: PassFail  = {}

    for name, builder in TREES.items():
        tree   = builder(irn_data, LEAF_SIZE)
        result = tree.query_knn(irn_queries, k)

        tree_idx   = np.array(irn.ndutils.to_numpy(result.indices))
        tree_dists = np.array(irn.ndutils.to_numpy(result.distances))

        passed = True
        for q in range(len(queries)):
            ref_set  = set(ref_idx[q])
            tree_set = set(tree_idx[q])
            if ref_set != tree_set:
                print(f"\n  [DEBUG {name} q={q}]")
                print(f"    ref  idx:   {sorted(ref_set)}")
                print(f"    tree idx:   {sorted(tree_set)}")
                print(f"    ref  dists: {np.sort(ref_dists[q])}")
                print(f"    tree dists: {np.sort(tree_dists[q])}")
                print(f"    only in ref:  {ref_set - tree_set}")
                print(f"    only in tree: {tree_set - ref_set}")
                passed = False

                if not np.isclose(tree_dists[q].max(), ref_dists[q].max(), rtol=1e-5):
                    passed = False
                    break

        results[name] = passed

    return results


def check_radius(data: np.ndarray, queries: np.ndarray, r: float) -> PassFail:
    """Compare each tree's radius results against brute-force reference."""
    irn_data    = irn.ndutils.from_numpy(data)
    irn_queries = irn.ndutils.from_numpy(queries)

    ref_results = bf_radius(data, queries, r)
    results: PassFail = {}

    for name, builder in TREES.items():
        tree   = builder(irn_data, LEAF_SIZE)
        batch  = tree.query_radius(irn_queries, r)

        passed = True
        for q, single in enumerate(batch.split()):
            ref_set  = set(ref_results[q][0])
            tree_set = set(single.indices.tolist())

            if ref_set != tree_set:
                passed = False
                break

        results[name] = passed

    return results


def format_results(label: str, dim: int, pass_fail: PassFail):
    status = {name: "✓" if ok else "✗" for name, ok in pass_fail.items()}
    row    = " | ".join(f"{name}: {s}" for name, s in status.items())
    print(f"  [{label} dim={dim}] {row}")
    return pass_fail


def run_correctness_tests():
    rng     = np.random.default_rng(1)
    failed  = []

    print("=== Correctness Tests ===\n")

    for dim in DIMS:
        data    = rng.standard_normal((N_POINTS, dim))
        queries = rng.standard_normal((N_QUERIES, dim))

        for label, check, extra in [
            ("kNN",    check_knn,    {"k": K}),
            ("Radius", check_radius, {"r": RADIUS}),
        ]:
            pass_fail = check(data, queries, **extra)
            format_results(label, dim, pass_fail)

            for name, ok in pass_fail.items():
                if not ok:
                    failed.append((label, dim, name))

    print(f"\n=== Summary ===")
    if not failed:
        print("All tests passed ✓")
    else:
        print(f"{len(failed)} failure(s):")
        for label, dim, name in failed:
            print(f"  ✗ {name} {label} dim={dim}")

    assert not failed, f"Correctness failures: {failed}"


if __name__ == "__main__":
    run_correctness_tests()