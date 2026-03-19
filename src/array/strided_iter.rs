use super::ndarray::NdArray;

pub struct StridedIter<'a, T> {
    array: &'a NdArray<T>,
    pos: usize,
    total: usize,
}

impl<'a, T> StridedIter<'a, T> {
    pub(crate) fn new(array: &'a NdArray<T>) -> Self {
        StridedIter {
            array,
            pos: 0,
            total: array.len(),
        }
    }
}

impl<'a, T: Copy> Iterator for StridedIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.pos >= self.total {
            return None;
        }
        let storage_offset = self.array.linear_to_storage_offset(self.pos);
        self.pos += 1;
        self.array.storage_get(storage_offset).copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.total - self.pos;
        (rem, Some(rem))
    }
}

impl<T: Copy> ExactSizeIterator for StridedIter<'_, T> {}
