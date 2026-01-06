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

        let range: f64 = high - low;
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
 
    pub fn standard_normal(&mut self, shape: Shape) -> NdArray<f64> {
        let pi= std::f64::consts::PI;
        let mut data= Vec::with_capacity(shape.size());

        for _ in 0..(shape.size() / 2){
            let rand1 = self.next_f64();
            let rand2 = self.next_f64();
            let r = f64::sqrt(-2.0 * f64::ln(rand1));
            let z0 = r * f64::sin(2.0 * pi * rand2);
            let z1 = r * f64::cos(2.0 * pi * rand2);
            data.push(z0);
            data.push(z1);
        }
        if shape.size() % 2 == 1 {
            let rand1 = self.next_f64();
            let rand2 = self.next_f64();
            let z0 = f64::sqrt(-2.0 * f64::ln(rand1)) * f64::sin(2.0 * pi * rand2);
            data.push(z0);
        }

        NdArray::from_vec(shape, data)
    }

    fn sample_standard_normal_single(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn normal(&mut self, mu: f64, sigma: f64, shape: Shape) -> NdArray<f64> {
        let z = self.standard_normal(shape);
        mu + sigma * z
    }

    pub fn lognormal(&mut self, mu: f64, sigma: f64, shape: Shape) -> NdArray<f64> {
        let z = self.normal(mu, sigma, shape);
        z.sqrt()
    }

    fn sample_gamma_single(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            return self.sample_gamma_single(alpha + 1.0) * self.next_f64().powf(1.0 / alpha);
        }
        
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        
        loop {
            let mut x;
            let mut v;

            loop {
                x = self.sample_standard_normal_single();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            
            v = v * v * v;
            let u = self.next_f64();

            if u < 1.0 - 0.0331 * x * x * x * x {
                return d * v;
            }
            
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn gamma(&mut self, shape_param: f64, scale: f64, shape: Shape) -> NdArray<f64> {
        assert!(shape_param > 0.0, "shape parameter must be positive");
        assert!(scale > 0.0, "scale must be positive");
        
        let data: Vec<f64> = (0..shape.size())
            .map(|_| self.sample_gamma_single(shape_param) * scale)
            .collect();
        
        NdArray::from_vec(shape, data)
    }
}

mod tests {
    #[allow(dead_code)]
    use super::*;

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

    #[allow(dead_code)]
    fn mean(data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    #[allow(dead_code)]
    fn variance(data: &[f64], mean: f64) -> f64 {
        data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64
    }
    
    #[test]
    fn test_standard_normal_moments() {
        let mut rng = Generator::new();
        let samples = rng.standard_normal(Shape::new(vec![100000]));
        let data = samples.as_slice();
        
        let sample_mean = mean(data);
        let sample_var = variance(data, sample_mean);
        
        assert!((sample_mean).abs() < 0.01, "Mean should be ~0, got {}", sample_mean);
        assert!((sample_var - 1.0).abs() < 0.02, "Variance should be ~1, got {}", sample_var);
    }
    
    #[test]
    fn test_normal_moments() {
        let mut rng = Generator::new();
        let mu = 5.0;
        let sigma = 2.0;
        let samples = rng.normal(mu, sigma, Shape::new(vec![100000]));
        let data = samples.as_slice();
        
        let sample_mean = mean(data);
        let sample_var = variance(data, sample_mean);
        
        assert!((sample_mean - mu).abs() < 0.02, "Mean should be ~{}, got {}", mu, sample_mean);
        assert!((sample_var - sigma * sigma).abs() < 0.05, 
                "Variance should be ~{}, got {}", sigma * sigma, sample_var);
    }

    #[test]
    fn test_gamma_moments() {
        let mut rng = Generator::new();
        let shape_param = 2.0;
        let scale = 3.0;
        let samples = rng.gamma(shape_param, scale, Shape::new(vec![100000]));
        let data = samples.as_slice();
        
        let sample_mean = mean(data);
        let sample_var = variance(data, sample_mean);
        
        let expected_mean = shape_param * scale;  // 6.0
        let expected_var = shape_param * scale * scale;  // 18.0
        
        assert!((sample_mean - expected_mean).abs() < 0.1, 
                "Mean should be ~{}, got {}", expected_mean, sample_mean);
        assert!((sample_var - expected_var).abs() < 0.3,
                "Variance should be ~{}, got {}", expected_var, sample_var);
    }

    #[test]
    fn test_gamma_shape_less_than_one() {
        let mut rng = Generator::new();
        let shape_param = 0.5;
        let scale = 1.0;
        let samples = rng.gamma(shape_param, scale, Shape::new(vec![50000]));
        let data = samples.as_slice();
        
        let sample_mean = mean(data);
        let expected_mean = shape_param * scale;
        
        assert!((sample_mean - expected_mean).abs() < 0.02,
                "Mean should be ~{}, got {}", expected_mean, sample_mean);
    }
}
