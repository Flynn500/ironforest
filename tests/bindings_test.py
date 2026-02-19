import ironforest as irn

def test_array_creation():
    """Test array creation methods"""
    print("Testing array creation...")
    
    # Test zeros
    arr = irn.zeros([2, 3])
    print(f"Zeros array: {arr}")
    print(f"Shape: {arr.shape}")
    print(f"Data: {arr.tolist()}")

    # Test from data
    arr2 = irn.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    print(f"\nArray from data: {arr2}")
    print(f"Element at [0, 1]: {arr2.get([0, 1])}")
    arr3 = irn.eye(3,3)
    print(f"\nArray from data: {arr3}")
    
def test_operations():
    """Test array operations"""
    print("\n\nTesting operations...")
    
    a = irn.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    b = irn.Array([2, 2], [5.0, 6.0, 7.0, 8.0])
    
    print(f"a = {a.tolist()}")
    print(f"b = {b.tolist()}")
    print(f"a + b = {(a + b).tolist()}")
    print(f"a * b = {(a * b).tolist()}")
    print(f"-a = {(-a).tolist()}")

def test_math_functions():
    """Test mathematical functions"""
    print("\n\nTesting math functions...")
    
    arr = irn.Array([1, 4], [0.0, 1.0, 4.0, 9.0])
    print(f"Original: {arr.tolist()}")
    print(f"sqrt: {arr.sqrt().tolist()}")
    print(f"exp: {arr.exp().tolist()}")
    print(f"sin: {arr.sin().tolist()}")

def test_generator():
    """Test random number generation"""
    print("\n\nTesting random generation...")
    
    gen = irn.random.Generator.from_seed(42)
    
    # Uniform random
    uniform = gen.uniform(0.0, 1.0, [2, 3])
    print(f"Uniform [0, 1): {uniform.tolist()}")
    
    # Normal distribution
    normal = gen.standard_normal([2, 3])
    print(f"Standard normal: {normal.tolist()}")
    
    # Random integers
    ints = gen.randint(0, 10, [2, 3])
    print(f"Random ints [0, 10): {ints}")

def test_matmul():
    print("\n\nTesting mat mul:")
    arr1 = irn.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    arr2 = irn.Array([2, 2], [2.0, 3.0, 5.0, 6.0])
    arr3 = arr1 @ arr2
    print(f"output: {arr3.tolist()}")

def test_ball_tree():
    points = irn.Array(
        shape=(6, 2),
        data=[
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
            5.0, 5.0,
            5.0, 6.0,
            9.0, 9.0,
        ]
    )

    print("\nTesting BallTree with Euclidean metric (default)")
    tree = irn.spatial.BallTree.from_array(points, leaf_size=2, metric="chebyshev")
    query_point = [1.0, 1.0]
    radius = 1.5

    indices = tree.query_radius(query_point, radius)
    print(f"Query point: {query_point}, radius: {radius}")
    print(f"Neighbor indices: {indices}")

    print(f"Neighbor points:")
    for i in range(len(indices)):
        point_idx = int(indices[i])
        point = points[point_idx]
        print(f"  Index {point_idx}: {point}")

    print("\nTesting BallTree with Manhattan metric")
    tree_manhattan = irn.spatial.BallTree.from_array(points, leaf_size=2, metric="manhattan")
    indices_manhattan = tree_manhattan.query_radius(query_point, radius)
    print(f"Query point: {query_point}, radius: {radius}")
    print(f"Neighbor indices (Manhattan): {indices_manhattan}")

    print("\nTesting BallTree KNN with Chebyshev metric")
    tree_chebyshev = irn.spatial.BallTree.from_array(points, leaf_size=2, metric="chebyshev")
    k = 3
    knn_indices = tree_chebyshev.query_knn(query_point, k)
    print(f"Query point: {query_point}, k: {k}")
    print(f"KNN indices (Chebyshev): {knn_indices}")


def test_column_stack():
    print("\n\nTesting column_stack...")

    # Test with 1D arrays
    a = irn.Array([3], [1.0, 2.0, 3.0])
    b = irn.Array([3], [4.0, 5.0, 6.0])
    result = irn.column_stack([a, b])
    print(f"column_stack of 1D arrays:")
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  result shape: {result.shape}")
    print(f"  result: {result.tolist()}")
    # Expected: [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]] flattened

    # Test with 2D arrays
    c = irn.Array([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = irn.Array([3, 1], [7.0, 8.0, 9.0])
    result2 = irn.column_stack([c, d])
    print(f"\ncolumn_stack of 2D arrays:")
    print(f"  c shape: {c.shape}, data: {c.tolist()}")
    print(f"  d shape: {d.shape}, data: {d.tolist()}")
    print(f"  result shape: {result2.shape}")
    print(f"  result: {result2.tolist()}")
    # Expected: [[1.0, 2.0, 7.0], [3.0, 4.0, 8.0], [5.0, 6.0, 9.0]] flattened

    # Test with mixed 1D and 2D
    e = irn.Array([3], [10.0, 11.0, 12.0])
    f = irn.Array([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result3 = irn.column_stack([e, f])
    print(f"\ncolumn_stack of 1D and 2D arrays:")
    print(f"  e = {e.tolist()}")
    print(f"  f shape: {f.shape}, data: {f.tolist()}")
    print(f"  result shape: {result3.shape}")
    print(f"  result: {result3.tolist()}")

def test_kernel_density():
    print("\n\nTesting kernel density estimation...")

    # Create a simple 2D dataset
    points = irn.Array(
        shape=[10, 2],
        data=[
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            1.0, 1.0,
            1.1, 1.0,
            1.0, 1.1,
            5.0, 5.0,
            5.1, 5.1,
            5.0, 5.2,
            5.2, 5.0,
        ]
    )

    print("Building BallTree with 10 points in 2D...")
    tree = irn.spatial.BallTree.from_array(points, leaf_size=3, metric="euclidean")

    # Test 1: Single point query (should return float)
    print("\nTest 1: Single point query [0.5, 0.5]")
    density_single = tree.kernel_density([0.5, 0.5], bandwidth=0.5, kernel="gaussian")
    print(f"  Type: {type(density_single)}")
    print(f"  Density: {density_single}")

    # Test 2: Multiple points query (should return Array)
    print("\nTest 2: Multiple points query")
    query_points = irn.Array(
        shape=[3, 2],
        data=[
            0.0, 0.0,   # Near first cluster
            1.0, 1.0,   # Near second cluster
            5.0, 5.0,   # Near third cluster
        ]
    )
    densities = tree.kernel_density(query_points, bandwidth=0.5, kernel="gaussian")
    print(f"  Type: {type(densities)}")
    print(f"  Shape: {densities}")
    print(f"  Densities: {densities}")

    # Test 3: Different kernels
    point = [1.0, 1.0]
    print("\nTest 3: Testing different kernel types at point [1.0, 1.0]")
    for kernel in ["gaussian", "epanechnikov", "uniform", "triangular"]:
        density = tree.kernel_density(point, bandwidth=0.5, kernel=kernel) # type: ignore
        print(f"  {kernel:15s}: {density:.6f}")

    # Test 4: Different bandwidths
    print("\nTest 4: Testing different bandwidths at point [0.0, 0.0]")
    for bandwidth in [0.1, 0.5, 1.0, 2.0]:
        density = tree.kernel_density([0.0, 0.0], bandwidth=bandwidth, kernel="gaussian")
        print(f"  bandwidth={bandwidth:.1f}: {density:.6f}")

    # Test 5: 1D array as single point
    print("\nTest 5: Using 1D array as single point")
    point_1d = irn.Array([2], [0.0, 0.0])
    density_1d = tree.kernel_density(point_1d, bandwidth=0.5, kernel="gaussian")
    print(f"  Type: {type(density_1d)}")
    print(f"  Density: {density_1d}")

if __name__ == "__main__":

    print(irn.Array([2,2], [0.0, 0.0,1.0,1.0]).tolist())

    # Simple linear relationship: y = 2x + 1 with some noise
    # Points: (1, 3), (2, 5), (3, 7), (4, 9)
    A = irn.Array([4, 2], [1.0, 1.0,  # [1, 1]
                        2.0, 1.0,  # [2, 1]
                        3.0, 1.0,  # [3, 1]
                        4.0, 1.0]) # [4, 1]
    b = irn.Array([4], [3.0, 5.0, 7.0, 9.0])

    # Equal weights - should give same result as ordinary least squares
    weights = irn.Array([4], [1.0, 1.0, 1.0, 1.0])
    x = irn.linalg.weighted_lstsq(A, b, weights)#A.weighted_least_squares(b, weights)
    print("Solution:", x[0].tolist())  # Should be close to [2.0, 1.0]
    
    # Now give more weight to first and last points
    weights2 = irn.Array([4], [10.0, 1.0, 1.0, 10.0])
    x2 = irn.linalg.weighted_lstsq(A, b, weights2)
    print("Weighted:", x2[0].tolist())  # Should still be close to [2.0, 1.0] since data is perfect

    test_array_creation()
    test_operations()
    test_math_functions()
    test_generator()
    test_matmul()
    test_ball_tree()
    test_column_stack()
    test_kernel_density()

