//Xoshiro256** Generator
use std::time::{SystemTime, UNIX_EPOCH};

use crate::array::ndarray::NdArray;
use crate::array::shape::Shape;
pub struct Generator {
    state: [u64; 4],
}

impl Generator {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_nanos() as u64;
        
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = [0u64; 4];
        let mut s = seed;
        
        for i in 0..4 {
            s = s.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            state[i] = z ^ (z >> 31);
        }
        
        Self { state }
    }
    
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        
        let t = self.state[1] << 17;
        
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        
        result
    }
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn randint(&mut self, low: i64, high: i64, shape: Shape) -> NdArray<i64> {
        assert!(low < high, "low must be less than high");
        
        let range: u64 = (high - low) as u64;
        let data: Vec<i64> = (0..shape.size())
            .map(|_| low + (self.next_u64() % range) as i64)
            .collect();
        
        NdArray::from_vec(shape, data)
    }

    pub fn uniform(&mut self, low: f64, high: f64, shape: Shape) -> NdArray<f64>{
        assert!(low < high, "low must be less than high");

        let range: f64 = (high - low);
        let data: Vec<f64> = (0..shape.size())
            .map(|_| low + (self.next_f64() * range))
            .collect();

        NdArray::from_vec(shape, data)
    }
 
    pub fn standard_uniform(&mut self, shape: Shape) -> NdArray<f64> {
        let data: Vec<f64> = (0..shape.size())
            .map(|_| self.next_f64())
            .collect();
        
        NdArray::from_vec(shape, data)
    }
}

mod tests {
    use super::*;
    use crate::array::shape::Shape;

    #[test]
    fn deterministic_from_seed() {
        let mut rng1 = Generator::from_seed(12345);
        let mut rng2 = Generator::from_seed(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut rng1 = Generator::from_seed(11111);
        let mut rng2 = Generator::from_seed(22222);

        assert_ne!(rng1.next_u64(), rng2.next_u64());
    }

    #[test]
    fn next_f64_in_range() {
        let mut rng = Generator::from_seed(42);

        for _ in 0..1000 {
            let val = rng.next_f64();
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn standard_uniform_shape() {
        let mut rng = Generator::from_seed(42);
        let arr = rng.standard_uniform(Shape::d2(3, 4));

        assert_eq!(arr.shape().dims(), &[3, 4]);
        assert_eq!(arr.len(), 12);
    }

    #[test]
    fn uniform_in_range() {
        let mut rng = Generator::from_seed(42);
        let arr = rng.uniform(5.0, 10.0, Shape::d1(1000));

        for &val in arr.as_slice() {
            assert!(val >= 5.0 && val < 10.0);
        }
    }

    #[test]
    fn randint_in_range() {
        let mut rng = Generator::from_seed(42);
        let arr = rng.randint(-10, 10, Shape::d1(1000));

        for &val in arr.as_slice() {
            assert!(val >= -10 && val < 10);
        }
    }
}