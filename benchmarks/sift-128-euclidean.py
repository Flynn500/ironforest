import h5py
import numpy as np
import time
import ironforest as irn


DATASET_PATH = "tests\sift-128-euclidean.hdf5"
K = 10

def load_dataset(path):
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float64)
        test  = np.array(f["test"],  dtype=np.float64)
        neighbors = np.array(f["neighbors"])
    return train, test, neighbors


def recall_at_k(results, ground_truth, k):
    recalls = []
    for i, found in enumerate(results):
        true_neighbors = set(ground_truth[i, :k])
        found_set = set(found[:k])
        recalls.append(len(true_neighbors & found_set) / k)
    return np.mean(recalls)

train, test, ground_truth = load_dataset(DATASET_PATH)
print(f"Train: {train.shape}, Test: {test.shape}")

train = irn.ndutils.from_numpy(train)
test = irn.ndutils.from_numpy(test)
print("Building index...")
t0 = time.perf_counter()
tree = irn.spatial.RPTree.from_array(train, leaf_size=500)
build_time = time.perf_counter() - t0
print(f"Build time: {build_time:.2f}s")

t0 = time.perf_counter()
n_queries = test.shape[0]

print(f"Build time: {build_time:.2f}s")
print("Querying...")
for n_candidates in [500, 1000, 5000, 10000, 20000]:
    results = []
    t0 = time.perf_counter()
    for i in range(n_queries):
        result = tree.query_ann(test[i], k=K, n_candidates=n_candidates)
        results.append(result.indices.tolist())
    query_time = time.perf_counter() - t0
    recall = recall_at_k(results, ground_truth, K)
    print(f"n_candidates={n_candidates:5d}  Recall@{K}: {recall:.4f} Query time: {query_time:.4f}")
