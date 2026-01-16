import substratum as ss

def test_array_creation():
    """Test array creation methods"""
    print("Testing array creation...")
    
    # Test zeros
    arr = ss.zeros([2, 3])
    print(f"Zeros array: {arr}")
    print(f"Shape: {arr.shape}")
    print(f"Data: {arr.tolist()}")

    # Test from data
    arr2 = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    print(f"\nArray from data: {arr2}")
    print(f"Element at [0, 1]: {arr2.get([0, 1])}")
    arr3 = ss.eye(3,3)
    print(f"\nArray from data: {arr3}")
    
def test_operations():
    """Test array operations"""
    print("\n\nTesting operations...")
    
    a = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    b = ss.Array([2, 2], [5.0, 6.0, 7.0, 8.0])
    
    print(f"a = {a.tolist()}")
    print(f"b = {b.tolist()}")
    print(f"a + b = {(a + b).tolist()}")
    print(f"a * b = {(a * b).tolist()}")
    print(f"-a = {(-a).tolist()}")

def test_math_functions():
    """Test mathematical functions"""
    print("\n\nTesting math functions...")
    
    arr = ss.Array([1, 4], [0.0, 1.0, 4.0, 9.0])
    print(f"Original: {arr.tolist()}")
    print(f"sqrt: {arr.sqrt().tolist()}")
    print(f"exp: {arr.exp().tolist()}")
    print(f"sin: {arr.sin().tolist()}")

def test_generator():
    """Test random number generation"""
    print("\n\nTesting random generation...")
    
    gen = ss.random.Generator.from_seed(42)
    
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
    arr1 = ss.Array([2, 2], [1.0, 2.0, 3.0, 4.0])
    arr2 = ss.Array([2, 2], [2.0, 3.0, 5.0, 6.0])
    arr3 = arr1 @ arr2
    print(f"output: {arr3.tolist()}")

def test_ball_tree():
    points = ss.Array(
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
    tree = ss.spatial.BallTree.from_array(points, leaf_size=2, metric="chebyshev")
    query_point = [1.0, 1.0]
    radius = 1.5

    indices = tree.query_radius(query_point, radius)
    print(f"Query point: {query_point}, radius: {radius}")
    print(f"Neighbor indices: {indices.tolist()}")

    print(f"Neighbor points:")
    for i in range(len(indices.tolist())):
        point_idx = int(indices.tolist()[i])
        point = points[point_idx]
        print(f"  Index {point_idx}: {point}")

    print("\nTesting BallTree with Manhattan metric")
    tree_manhattan = ss.spatial.BallTree.from_array(points, leaf_size=2, metric="manhattan")
    indices_manhattan = tree_manhattan.query_radius(query_point, radius)
    print(f"Query point: {query_point}, radius: {radius}")
    print(f"Neighbor indices (Manhattan): {indices_manhattan.tolist()}")

    print("\nTesting BallTree KNN with Chebyshev metric")
    tree_chebyshev = ss.spatial.BallTree.from_array(points, leaf_size=2, metric="chebyshev")
    k = 3
    knn_indices = tree_chebyshev.query_knn(query_point, k)
    print(f"Query point: {query_point}, k: {k}")
    print(f"KNN indices (Chebyshev): {knn_indices.tolist()}")

def test_column_stack():
    print("\n\nTesting column_stack...")

    # Test with 1D arrays
    a = ss.Array([3], [1.0, 2.0, 3.0])
    b = ss.Array([3], [4.0, 5.0, 6.0])
    result = ss.column_stack([a, b])
    print(f"column_stack of 1D arrays:")
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  result shape: {result.shape}")
    print(f"  result: {result.tolist()}")
    # Expected: [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]] flattened

    # Test with 2D arrays
    c = ss.Array([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = ss.Array([3, 1], [7.0, 8.0, 9.0])
    result2 = ss.column_stack([c, d])
    print(f"\ncolumn_stack of 2D arrays:")
    print(f"  c shape: {c.shape}, data: {c.tolist()}")
    print(f"  d shape: {d.shape}, data: {d.tolist()}")
    print(f"  result shape: {result2.shape}")
    print(f"  result: {result2.tolist()}")
    # Expected: [[1.0, 2.0, 7.0], [3.0, 4.0, 8.0], [5.0, 6.0, 9.0]] flattened

    # Test with mixed 1D and 2D
    e = ss.Array([3], [10.0, 11.0, 12.0])
    f = ss.Array([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result3 = ss.column_stack([e, f])
    print(f"\ncolumn_stack of 1D and 2D arrays:")
    print(f"  e = {e.tolist()}")
    print(f"  f shape: {f.shape}, data: {f.tolist()}")
    print(f"  result shape: {result3.shape}")
    print(f"  result: {result3.tolist()}")

if __name__ == "__main__":
    test_array_creation()
    test_operations()
    test_math_functions()
    test_generator()
    test_matmul()
    test_ball_tree()
    test_column_stack()
    print("\n\nAll tests passed! ✓")

