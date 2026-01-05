use substratum::{NdArray, Shape};
use substratum::random::Generator;

//cargo run --bin playground
fn main() {
    println!("=== Basic Array Creation ===");
    
    // From vec with explicit shape
    let a = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
    println!("2x3 array: {:?}", a.as_slice());
    
    // Zeros and filled
    let zeros: NdArray<f64> = NdArray::zeros(Shape::d2(2, 2));
    println!("2x2 zeros: {:?}", zeros.as_slice());
    
    let fives = NdArray::filled(Shape::d1(4), 5);
    println!("Filled with 5s: {:?}", fives.as_slice());

    println!("\n=== Indexing ===");
    
    // Multi-dimensional indexing (row-major)
    let matrix = NdArray::from_vec(Shape::d2(2, 3), vec![10, 20, 30, 40, 50, 60]);
    println!("Matrix [0,0]: {:?}", matrix.get(&[0, 0])); // 10
    println!("Matrix [0,2]: {:?}", matrix.get(&[0, 2])); // 30
    println!("Matrix [1,1]: {:?}", matrix.get(&[1, 1])); // 50
    println!("Matrix [5,5] (out of bounds): {:?}", matrix.get(&[5, 5])); // None

    println!("\n=== Arithmetic (same shape) ===");
    
    let x = NdArray::from_vec(Shape::d1(4), vec![1.0_f64, 2.0, 3.0, 4.0]);
    let y = NdArray::from_vec(Shape::d1(4), vec![10.0_f64, 20.0, 30.0, 40.0]);
    
    println!("x: {:?}", x.as_slice());
    println!("y: {:?}", y.as_slice());
    println!("x + y: {:?}", (&x + &y).as_slice());
    println!("x * y: {:?}", (&x * &y).as_slice());
    println!("y - x: {:?}", (&y - &x).as_slice());
    println!("y / x: {:?}", (&y / &x).as_slice());

    println!("\n=== Scalar Operations ===");
    
    let arr = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 3.0]);
    println!("arr: {:?}", arr.as_slice());
    println!("arr + 10: {:?}", (&arr + 10.0).as_slice());
    println!("arr * 2: {:?}", (&arr * 2.0).as_slice());
    println!("10 - arr: {:?}", (10.0 - &arr).as_slice());
    println!("12 / arr: {:?}", (12.0 / &arr).as_slice());

    println!("\n=== Negation ===");
    
    let pos = NdArray::from_vec(Shape::d1(3), vec![1, -2, 3]);
    println!("original: {:?}", pos.as_slice());
    println!("negated: {:?}", (-&pos).as_slice());

    println!("\n=== Broadcasting ===");
    
    // [2, 3] + [3] -> broadcasts [3] to each row
    let matrix = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
    let row = NdArray::from_vec(Shape::d1(3), vec![10, 20, 30]);
    
    println!("matrix (2x3): {:?}", matrix.as_slice());
    println!("row (3,): {:?}", row.as_slice());
    println!("matrix + row: {:?}", (&matrix + &row).as_slice());
    
    // Scalar broadcast
    let scalar = NdArray::from_vec(Shape::scalar(), vec![100]);
    println!("matrix + scalar(100): {:?}", (&matrix + &scalar).as_slice());
    
    // [3, 1] + [1, 4] -> [3, 4]
    let col: NdArray<f64> = NdArray::from_vec(Shape::new(vec![3, 1]), vec![1.0_f64, 2.0, 3.0]);
    let row: NdArray<f64> = NdArray::from_vec(Shape::new(vec![1, 4]), vec![10.0_f64, 20.0, 30.0, 40.0]);
    let outer: NdArray<f64> = &col * &row;
    
    println!("\nOuter product via broadcasting:");
    println!("col (3x1): {:?}", col.as_slice());
    println!("row (1x4): {:?}", row.as_slice());
    println!("col * row (3x4): {:?}", outer.as_slice());
    println!("result shape: {:?}", outer.shape().dims());

    println!("\n=== Chained Operations ===");
    
    // (a + b) * 2 - 1
    let a = NdArray::from_vec(Shape::d1(3), vec![1.0_f64, 2.0, 3.0]);
    let b = NdArray::from_vec(Shape::d1(3), vec![4.0_f64, 5.0, 6.0]);
    
    let result = (&(&a + &b) * 2.0) - 1.0;
    println!("a: {:?}", a.as_slice());
    println!("b: {:?}", b.as_slice());
    println!("(a + b) * 2 - 1: {:?}", result.as_slice());

    println!("\n=== Random Generation ===");
    
    // Deterministic seeding
    let mut rng = Generator::from_seed(42);
    
    let uniform01 = rng.standard_uniform(Shape::d2(2, 3));
    println!("standard_uniform (2x3): {:?}", uniform01.as_slice());
    
    let uniform_range = rng.uniform(5.0, 10.0, Shape::d1(5));
    println!("uniform [5, 10) (5,): {:?}", uniform_range.as_slice());
    
    let integers = rng.randint(-10, 10, Shape::d2(2, 4));
    println!("randint [-10, 10) (2x4): {:?}", integers.as_slice());
    
    // Verify determinism
    println!("\n=== Determinism Check ===");
    
    let mut rng1 = Generator::from_seed(12345);
    let mut rng2 = Generator::from_seed(12345);
    
    let arr1 = rng1.standard_uniform(Shape::d1(5));
    let arr2 = rng2.standard_uniform(Shape::d1(5));
    
    println!("rng1 output: {:?}", arr1.as_slice());
    println!("rng2 output: {:?}", arr2.as_slice());
    println!("identical: {}", arr1.as_slice() == arr2.as_slice());
}