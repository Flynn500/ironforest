import ironforest as irn
from ironforest import spatial
import numpy as np

def brute_force_ground_truth(data, query, k=5, radius=1):
    """
    Pure NumPy implementation of KNN and Radius search.
    """
    # Calculate Euclidean distance: sqrt(sum((a - b)^2))
    diff = data - query
    dist_sq = np.sum(diff**2, axis=1)
    distances = np.sqrt(dist_sq)
    
    if k is not None:
        # Get indices of k smallest distances
        idx = np.argsort(distances)[:k]
        return distances[idx], idx
    
    if radius is not None:
        # Get indices where distance <= radius
        idx = np.where(distances <= radius)[0]
        # Sort them by distance for easier comparison
        sorted_sub_idx = np.argsort(distances[idx])
        return distances[idx][sorted_sub_idx], idx[sorted_sub_idx]
    
def run_correctness_suite():
    # Setup
    n, dims = 1000, 10
    k, rad = 5, 15.0
    rng = np.random.default_rng(42)
    
    # Generate data
    raw_data = rng.uniform(0.0, 100.0, (n, dims))
    query = rng.uniform(0.0, 100.0, (dims,))
    irn_data = irn.ndutils.from_numpy(raw_data)

    # Define all structures to test
    structures = [
        ("BruteForce", spatial.BruteForce),
        ("KDTree", spatial.KDTree),
        ("BallTree", spatial.BallTree),
        ("VPTree", spatial.VPTree),
        ("RPTree", spatial.RPTree),
        ("MTree", spatial.MTree)
    ]

    # Get Ground Truth
    gt_knn_dist, gt_knn_idx = brute_force_ground_truth(raw_data, query, k=k) # type: ignore
    gt_rad_dist, gt_rad_idx = brute_force_ground_truth(raw_data, query, radius=rad) # type: ignore

    for name, Cls in structures:
        print(f"Validating {name}...")
        tree = Cls.from_array(irn_data, leaf_size=10)

        # --- Test kNN ---
        res_knn = tree.query_knn(query, k=k)
        
        # Check distances (allows for float precision jitter)
        np.testing.assert_allclose(res_knn.distances, gt_knn_dist, rtol=1e-5, 
                                   err_msg=f"{name} KNN distances mismatch")
        
        # Check metadata
        assert np.isclose(res_knn.mean(), np.mean(gt_knn_dist))
        assert np.isclose(res_knn.radius(), gt_knn_dist[-1])

        # --- Test Radius ---
        res_rad = tree.query_radius(query, rad)
        
        # Sort results to ensure order doesn't break the test
        sort_mask = np.argsort(res_rad.indices)
        test_indices = np.array(res_rad.indices)[sort_mask]
        
        assert len(test_indices) == len(gt_rad_idx), f"{name} Radius count mismatch"
        np.testing.assert_array_equal(test_indices, sorted(gt_rad_idx), 
                                      err_msg=f"{name} Radius indices mismatch")

    print("\n✨ All structures validated successfully against NumPy Brute Force!")

run_correctness_suite()