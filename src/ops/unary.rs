use std::ops::Neg;
use crate::array::ndarray::NdArray;


impl<T> Neg for NdArray<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = NdArray<T>;

    fn neg(mut self) -> Self::Output {
        for elem in self.as_mut_slice().expect("Neg: array must be owned") {
            *elem = -(*elem);
        }
        self
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
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log(self) -> Self;
    fn abs(self) -> Self;
    fn sign(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
}

impl Float for f32 {
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn tan(self) -> Self { self.tan() }
    fn asin(self) -> Self { self.asin() }
    fn acos(self) -> Self { self.acos() }
    fn atan(self) -> Self { self.atan() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn log(self) -> Self { self.ln() }
    fn abs(self) -> Self { self.abs() }
    fn sign(self) -> Self { if self > 0.0 { 1.0 } else if self < 0.0 { -1.0 } else { 0.0 } }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn powf(self, n: Self) -> Self { self.powf(n) }
}

impl Float for f64 {
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn tan(self) -> Self { self.tan() }
    fn asin(self) -> Self { self.asin() }
    fn acos(self) -> Self { self.acos() }
    fn atan(self) -> Self { self.atan() }
    fn sqrt(self) -> Self { self.sqrt() }
    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn log(self) -> Self { self.ln() }
    fn abs(self) -> Self { self.abs() }
    fn sign(self) -> Self { if self > 0.0 { 1.0 } else if self < 0.0 { -1.0 } else { 0.0 } }
    fn powi(self, n: i32) -> Self { self.powi(n) }
    fn powf(self, n: Self) -> Self { self.powf(n) }
}

fn map_inplace<T, F>(mut arr: NdArray<T>, f: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    for elem in arr.as_mut_slice().expect("map_inplace: array must be owned") {
        *elem = f(*elem);
    }
    arr
}

impl<T: Float> NdArray<T> {
    pub fn sin(&self) -> Self { self.map(|x| x.sin()) }
    pub fn cos(&self) -> Self { self.map(|x| x.cos()) }
    pub fn tan(&self) -> Self { self.map(|x| x.tan()) }
    pub fn asin(&self) -> Self { self.map(|x| x.asin()) }
    pub fn acos(&self) -> Self { self.map(|x| x.acos()) }
    pub fn atan(&self) -> Self { self.map(|x| x.atan()) }
    pub fn sqrt(&self) -> Self { self.map(|x| x.sqrt()) }
    pub fn exp(&self) -> Self { self.map(|x| x.exp()) }
    pub fn ln(&self) -> Self { self.map(|x| x.ln()) }
    pub fn log(&self) -> Self { self.map(|x| x.log()) }
    pub fn abs(&self) -> Self { self.map(|x| x.abs()) }
    pub fn sign(&self) -> Self { self.map(|x| x.sign()) }
    pub fn powi(&self, n: i32) -> Self { self.map(|x| x.powi(n)) }
    pub fn powf(&self, n: T) -> Self { self.map(|x| x.powf(n)) }

    pub fn into_sin(self) -> Self { map_inplace(self, |x| x.sin()) }
    pub fn into_cos(self) -> Self { map_inplace(self, |x| x.cos()) }
    pub fn into_tan(self) -> Self { map_inplace(self, |x| x.tan()) }
    pub fn into_asin(self) -> Self { map_inplace(self, |x| x.asin()) }
    pub fn into_acos(self) -> Self { map_inplace(self, |x| x.acos()) }
    pub fn into_atan(self) -> Self { map_inplace(self, |x| x.atan()) }
    pub fn into_sqrt(self) -> Self { map_inplace(self, |x| x.sqrt()) }
    pub fn into_exp(self) -> Self { map_inplace(self, |x| x.exp()) }
    pub fn into_ln(self) -> Self { map_inplace(self, |x| x.ln()) }
    pub fn into_log(self) -> Self { map_inplace(self, |x| x.log()) }
    pub fn into_abs(self) -> Self { map_inplace(self, |x| x.abs()) }
    pub fn into_sign(self) -> Self { map_inplace(self, |x| x.sign()) }
    pub fn into_powi(self, n: i32) -> Self { map_inplace(self, |x| x.powi(n)) }
    pub fn into_powf(self, n: T) -> Self { map_inplace(self, |x| x.powf(n)) }
}

impl<T: PartialOrd + Copy> NdArray<T> {
    pub fn clip(&self, min: T, max: T) -> Self {
        self.map(|x| {
            if x < min { min }
            else if x > max { max }
            else { x }
        })
    }

    pub fn into_clip(self, min: T, max: T) -> Self {
        map_inplace(self, |x| {
            if x < min { min }
            else if x > max { max }
            else { x }
        })
    }
}
