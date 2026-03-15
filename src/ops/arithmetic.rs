use std::ops::{Add, Sub, Mul, Div};

use crate::array::broadcast::BroadcastIter;
use crate::array::ndarray::NdArray;
use crate::array::storage::Storage;

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        //owned + owned
        impl<T> $trait for NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output {
                binary_op_inplace(self, &rhs, |a, b| a $op b)
            }
        }

        //owned + ref
        impl<T> $trait<&NdArray<T>> for NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output {
                binary_op_inplace(self, rhs, |a, b| a $op b)
            }
        }

        //ref + owned
        impl<T> $trait<NdArray<T>> for &NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: NdArray<T>) -> Self::Output {
                binary_op_inplace_rev(rhs, self, |a, b| a $op b)
            }
        }

        //ref + ref
        impl<T> $trait<&NdArray<T>> for &NdArray<T>
        where
            T: $trait<Output = T> + Copy,
        {
            type Output = NdArray<T>;

            fn $method(self, rhs: &NdArray<T>) -> Self::Output {
                binary_op_copy(self, rhs, |a, b| a $op b)
            }
        }
    };
}

fn binary_op_copy<T, F>(a: &NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    // Materialise any Strided inputs to contiguous; no-op for Owned/External.
    let a_c; let b_c;
    let a = if a.is_contiguous() { a } else { a_c = a.to_contiguous(); &a_c };
    let b = if b.is_contiguous() { b } else { b_c = b.to_contiguous(); &b_c };

    let iter = BroadcastIter::new(a.shape(), b.shape())
        .expect("Shapes are not broadcast-compatible");

    let output_shape = iter.output_shape().clone();
    let a_data = a.as_slice_unchecked();
    let b_data = b.as_slice_unchecked();

    let result_data: Vec<T> = iter
        .map(|(idx_a, idx_b)| op(a_data[idx_a], b_data[idx_b]))
        .collect();

    NdArray::new(output_shape, Storage::from_vec(result_data))
}

fn binary_op_inplace<T, F>(mut a: NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if a.shape() == b.shape() {
        // `a` is always Owned here (consumed by value); `b` may be External.
        // Materialise `b` if strided so flat indexing is valid.
        let b_c;
        let b = if b.is_contiguous() { b } else { b_c = b.to_contiguous(); &b_c };
        let b_data = b.as_slice_unchecked();
        let a_data = a.as_mut_slice().expect("binary_op_inplace: a must be owned");
        for i in 0..a_data.len() {
            a_data[i] = op(a_data[i], b_data[i]);
        }
        a
    } else {
        binary_op_copy(&a, b, op)
    }
}

fn binary_op_inplace_rev<T, F>(mut a: NdArray<T>, b: &NdArray<T>, op: F) -> NdArray<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if a.shape() == b.shape() {
        let b_c;
        let b = if b.is_contiguous() { b } else { b_c = b.to_contiguous(); &b_c };
        let b_data = b.as_slice_unchecked();
        let a_data = a.as_mut_slice().expect("binary_op_inplace_rev: a must be owned");
        for i in 0..a_data.len() {
            a_data[i] = op(b_data[i], a_data[i]);
        }
        a
    } else {
        binary_op_copy(b, &a, op)
    }
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Sub, sub, -);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);
