use std::ops::Neg;
use crate::array::ndarray::NdArray;
use crate::array::storage::Storage;


impl<T: Copy> NdArray<T> {
    pub fn map<F, U>(&self, f: F) -> NdArray<U>
    where
        F: Fn(T) -> U,
        U: Copy,
    {
        let result_data: Vec<U> = self.as_slice().iter().map(|&x| f(x)).collect();
        NdArray::new(self.shape().clone(), Storage::from_vec(result_data))
    }
}

impl<T> Neg for NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = NdArray<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

impl<T> Neg for &NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = NdArray<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

pub trait Float: Copy + Sized {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn abs(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
}

impl Float for f32 {
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn tan(self) -> Self { self.tan() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn abs(self) -> Self { self.abs() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn powf(self, n: Self) -> Self { self.powf(n) }
}

impl Float for f64 {
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn tan(self) -> Self { self.tan() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn abs(self) -> Self { self.abs() }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn powf(self, n: Self) -> Self { self.powf(n) }
}

impl<T: Float> NdArray<T> {
    pub fn sin(&self) -> Self { self.map(|x| x.sin()) }
    pub fn cos(&self) -> Self { self.map(|x| x.cos()) }
    pub fn tan(&self) -> Self { self.map(|x| x.tan()) }
    pub fn sqrt(&self) -> Self { self.map(|x| x.sqrt()) }
    pub fn exp(&self) -> Self { self.map(|x| x.exp()) }
    pub fn ln(&self) -> Self { self.map(|x| x.ln()) }
    pub fn abs(&self) -> Self { self.map(|x| x.abs()) }
    pub fn powi(&self, n: i32) -> Self { self.map(|x| x.powi(n)) }
    pub fn powf(&self, n: T) -> Self { self.map(|x| x.powf(n)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::shape::Shape;

    #[test]
    fn neg_integers() {
        let a = NdArray::from_vec(Shape::d1(3), vec![1, -2, 3]);
        let b = -&a;
        assert_eq!(b.as_slice(), &[-1, 2, -3]);
    }

    #[test]
    fn neg_floats() {
        let a = NdArray::from_vec(Shape::d1(2), vec![1.5, -2.5]);
        let b = -a;
        assert_eq!(b.as_slice(), &[-1.5, 2.5]);
    }

    #[test]
    fn neg_preserves_shape() {
        let a = NdArray::from_vec(Shape::d2(2, 3), vec![1, 2, 3, 4, 5, 6]);
        let b = -&a;
        assert_eq!(b.shape().dims(), &[2, 3]);
    }

    #[test]
    fn sin_f64() {
        let a = NdArray::from_vec(Shape::d1(3), vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
        let b = a.sin();
        
        assert!((b.as_slice()[0] - 0.0).abs() < 1e-10);
        assert!((b.as_slice()[1] - 1.0).abs() < 1e-10);
        assert!((b.as_slice()[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn cos_f32() {
        let a = NdArray::from_vec(Shape::d1(2), vec![0.0f32, std::f32::consts::PI]);
        let b = a.cos();
        
        assert!((b.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((b.as_slice()[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn sqrt_preserves_shape() {
        let a = NdArray::from_vec(Shape::d2(2, 2), vec![4.0, 9.0, 16.0, 25.0]);
        let b = a.sqrt();
        
        assert_eq!(b.shape().dims(), &[2, 2]);
        assert_eq!(b.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn exp_and_ln() {
        let a = NdArray::from_vec(Shape::d1(3), vec![0.0, 1.0, 2.0]);
        let b = a.exp();
        let c = b.ln();
        
        for i in 0..3 {
            assert!((c.as_slice()[i] - a.as_slice()[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn abs_mixed_signs() {
        let a = NdArray::from_vec(Shape::d1(4), vec![-2.5, 3.5, -1.0, 0.0]);
        let b = a.abs();
        
        assert_eq!(b.as_slice(), &[2.5, 3.5, 1.0, 0.0]);
    }
}