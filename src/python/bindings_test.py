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

if __name__ == "__main__":
    test_array_creation()
    test_operations()
    test_math_functions()
    test_generator()
    test_matmul()
    print("\n\nAll tests passed! ✓")