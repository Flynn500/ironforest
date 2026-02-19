import ironforest as irn
import numpy as np
import time

def benchmark(func, name, iterations=100):
    """Time a function over multiple iterations"""
    # Warmup
    func()
    
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / iterations) * 1000
    print(f"{name}: {avg_ms:.4f} ms/iter")
    return avg_ms

def bench_array_creation(size=1000):
    """Benchmark array creation"""
    print(f"\n=== Array Creation ({size}x{size}) ===")
    
    shape_ss = [size, size]
    shape_np = (size, size)
    
    ss_time = benchmark(lambda: irn.zeros(shape_ss), "substratum zeros")
    np_time = benchmark(lambda: np.zeros(shape_np), "numpy zeros")
    
    print(f"Ratio (numpy/substratum): {np_time/ss_time:.2f}x")

def bench_operations(size=1000):
    """Benchmark element-wise operations"""
    print(f"\n=== Element-wise Operations ({size}x{size}) ===")
    
    # Setup
    data = list(range(size * size))
    data_f = [float(x) for x in data]
    
    a_ss = irn.Array([size, size], data_f)
    b_ss = irn.Array([size, size], data_f)
    
    a_np = np.array(data_f).reshape(size, size)
    b_np = np.array(data_f).reshape(size, size)
    
    # Addition
    ss_add = benchmark(lambda: a_ss + b_ss, "substratum add")
    np_add = benchmark(lambda: a_np + b_np, "numpy add")
    print(f"Ratio: {np_add/ss_add:.2f}x\n")
    
    # Multiplication
    ss_mul = benchmark(lambda: a_ss * b_ss, "substratum mul")
    np_mul = benchmark(lambda: a_np * b_np, "numpy mul")
    print(f"Ratio: {np_mul/ss_mul:.2f}x\n")
    
    # Negation
    ss_neg = benchmark(lambda: -a_ss, "substratum neg")
    np_neg = benchmark(lambda: -a_np, "numpy neg")
    print(f"Ratio: {np_neg/ss_neg:.2f}x")

def bench_math_functions(size=1000):
    """Benchmark math functions"""
    print(f"\n=== Math Functions ({size}x{size}) ===")
    
    # Setup with positive values for sqrt
    data = [float(i + 1) for i in range(size * size)]
    
    arr_ss = irn.Array([size, size], data)
    arr_np = np.array(data).reshape(size, size)
    
    # sqrt
    ss_sqrt = benchmark(lambda: arr_ss.sqrt(), "substratum sqrt")
    np_sqrt = benchmark(lambda: np.sqrt(arr_np), "numpy sqrt")
    print(f"Ratio: {np_sqrt/ss_sqrt:.2f}x\n")
    
    # exp
    small_data = [float(i % 10) for i in range(size * size)]
    arr_ss_small = irn.Array([size, size], small_data)
    arr_np_small = np.array(small_data).reshape(size, size)
    
    ss_exp = benchmark(lambda: arr_ss_small.exp(), "substratum exp")
    np_exp = benchmark(lambda: np.exp(arr_np_small), "numpy exp")
    print(f"Ratio: {np_exp/ss_exp:.2f}x\n")
    
    # sin
    ss_sin = benchmark(lambda: arr_ss.sin(), "substratum sin")
    np_sin = benchmark(lambda: np.sin(arr_np), "numpy sin")
    print(f"Ratio: {np_sin/ss_sin:.2f}x")

def bench_random(size=1000):
    """Benchmark random number generation"""
    print(f"\n=== Random Generation ({size}x{size}) ===")
    
    gen_ss = irn.random.Generator.from_seed(42)
    rng_np = np.random.default_rng(42)
    
    shape_ss = [size, size]
    shape_np = (size, size)
    
    # Uniform
    ss_uni = benchmark(lambda: gen_ss.uniform(0.0, 1.0, shape_ss), "substratum uniform")
    np_uni = benchmark(lambda: rng_np.uniform(0.0, 1.0, shape_np), "numpy uniform")
    print(f"Ratio: {np_uni/ss_uni:.2f}x\n")
    
    # Normal
    ss_norm = benchmark(lambda: gen_ss.standard_normal(shape_ss), "substratum normal")
    np_norm = benchmark(lambda: rng_np.standard_normal(shape_np), "numpy normal")
    print(f"Ratio: {np_norm/ss_norm:.2f}x\n")
    
    # Randint
    ss_int = benchmark(lambda: gen_ss.randint(0, 100, shape_ss), "substratum randint")
    np_int = benchmark(lambda: rng_np.integers(0, 100, shape_np), "numpy randint")
    print(f"Ratio: {np_int/ss_int:.2f}x")

if __name__ == "__main__":
    print("=" * 50)
    print("Substratum vs NumPy Benchmark")
    print("=" * 50)
    
    bench_array_creation()
    bench_operations()
    bench_math_functions()
    bench_random()
    
    print("\n" + "=" * 50)
    print("Benchmark complete!")