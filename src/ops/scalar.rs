use std::ops::{Add, Sub, Mul, Div};

use crate::array::ndarray::NdArray;
use crate::array::storage::Storage;

macro_rules! impl_scalar_ops {
    ($scalar:ty) => {
        // array + scalar
        impl Add<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: $scalar) -> Self::Output {
                scalar_op_inplace(self, rhs, |a, b| a + b)
            }
        }

        impl Add<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: $scalar) -> Self::Output {
                scalar_op_copy(self, rhs, |a, b| a + b)
            }
        }

        // scalar + array
        impl Add<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_inplace(rhs, self, |a, b| a + b)
            }
        }

        impl Add<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn add(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_copy(rhs, self, |a, b| a + b)
            }
        }

        // array - scalar
        impl Sub<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: $scalar) -> Self::Output {
                scalar_op_inplace(self, rhs, |a, b| a - b)
            }
        }

        impl Sub<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: $scalar) -> Self::Output {
                scalar_op_copy(self, rhs, |a, b| a - b)
            }
        }

        // scalar - array
        impl Sub<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_inplace_rev(rhs, self, |a, b| a - b)
            }
        }

        impl Sub<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn sub(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_copy_rev(rhs, self, |a, b| a - b)
            }
        }

        // array * scalar
        impl Mul<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: $scalar) -> Self::Output {
                scalar_op_inplace(self, rhs, |a, b| a * b)
            }
        }

        impl Mul<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: $scalar) -> Self::Output {
                scalar_op_copy(self, rhs, |a, b| a * b)
            }
        }

        // scalar * array
        impl Mul<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_inplace(rhs, self, |a, b| a * b)
            }
        }

        impl Mul<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn mul(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_copy(rhs, self, |a, b| a * b)
            }
        }

        // array / scalar
        impl Div<$scalar> for NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: $scalar) -> Self::Output {
                scalar_op_inplace(self, rhs, |a, b| a / b)
            }
        }

        impl Div<$scalar> for &NdArray<$scalar> {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: $scalar) -> Self::Output {
                scalar_op_copy(self, rhs, |a, b| a / b)
            }
        }

        // scalar / array
        impl Div<NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: NdArray<$scalar>) -> Self::Output {
                scalar_op_inplace_rev(rhs, self, |a, b| a / b)
            }
        }

        impl Div<&NdArray<$scalar>> for $scalar {
            type Output = NdArray<$scalar>;

            fn div(self, rhs: &NdArray<$scalar>) -> Self::Output {
                scalar_op_copy_rev(rhs, self, |a, b| a / b)
            }
        }
    };
}

fn scalar_op_copy<T, F>(arr: &NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let result: Vec<T> = arr.iter_logical().map(|a| op(a, scalar)).collect();
    NdArray::new(arr.shape().clone(), Storage::from_vec(result))
}

fn scalar_op_inplace<T, F>(mut arr: NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    // arr is always Owned here (consumed by value).
    for elem in arr.as_mut_slice().expect("scalar_op_inplace: arr must be owned") {
        *elem = op(*elem, scalar);
    }
    arr
}

fn scalar_op_copy_rev<T, F>(arr: &NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let result: Vec<T> = arr.iter_logical().map(|a| op(scalar, a)).collect();
    NdArray::new(arr.shape().clone(), Storage::from_vec(result))
}

fn scalar_op_inplace_rev<T, F>(mut arr: NdArray<T>, scalar: T, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    for elem in arr.as_mut_slice().expect("scalar_op_inplace_rev: arr must be owned") {
        *elem = op(scalar, *elem);
    }
    arr
}

impl_scalar_ops!(f32);
impl_scalar_ops!(f64);
impl_scalar_ops!(i32);
impl_scalar_ops!(i64);
impl_scalar_ops!(u32);
impl_scalar_ops!(u64);
