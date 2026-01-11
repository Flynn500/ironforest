#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    pub fn d1(len: usize) -> Self {
        Shape { dims: vec![len] }
    }

    pub fn d2(rows: usize, cols: usize) -> Self {
        Shape { dims: vec![rows, cols] }
    }

    pub fn d3(d0: usize, d1: usize, d2: usize) -> Self {
        Shape { dims: vec![d0, d1, d2] }
    }

    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn dim(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    pub fn strides_row_major(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; self.dims.len()];

        for i in (0..self.dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        
        strides
    }

    pub fn strides_col_major(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; self.dims.len()];

        for i in 1..self.dims.len() {
            strides[i] = strides[i - 1] * self.dims[i - 1];
        }
        
        strides
    }

    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim_a = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };

            let dim_b = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if dim_a == dim_b {
                result_dims.push(dim_a);
            } else if dim_a == 1 {
                result_dims.push(dim_b);
            } else if dim_b == 1 {
                result_dims.push(dim_a);
            } else {
                return None;
            }
        }

        result_dims.reverse();
        Some(Shape::new(result_dims))
    }

    pub fn broadcasts_to(&self, target: &Shape) -> bool {
        match self.broadcast(target) {
            Some(result) => result == *target,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_constructors() {
        //assert_eq!(Shape::scalar().dims(), &[]);
        assert_eq!(Shape::d1(5).dims(), &[5]);
        assert_eq!(Shape::d2(3, 4).dims(), &[3, 4]);
        assert_eq!(Shape::d3(2, 3, 4).dims(), &[2, 3, 4]);
    }

    #[test]
    fn shape_size() {
        assert_eq!(Shape::scalar().size(), 1);
        assert_eq!(Shape::d1(5).size(), 5);
        assert_eq!(Shape::d2(3, 4).size(), 12);
        assert_eq!(Shape::d3(2, 3, 4).size(), 24);
    }

    #[test]
    fn strides_row_major() {
        let shape = Shape::d3(3, 4, 5);
        assert_eq!(shape.strides_row_major(), vec![20, 5, 1]);

        let shape2 = Shape::d2(2, 3);
        assert_eq!(shape2.strides_row_major(), vec![3, 1]);
    }

    #[test]
    fn strides_col_major() {
        let shape = Shape::d3(3, 4, 5);
        assert_eq!(shape.strides_col_major(), vec![1, 3, 12]);
    }

    #[test]
    fn broadcast_same_shape() {
        let a = Shape::d2(3, 4);
        let b = Shape::d2(3, 4);
        assert_eq!(a.broadcast(&b), Some(Shape::d2(3, 4)));
    }

    #[test]
    fn broadcast_scalar() {
        let a = Shape::d2(3, 4);
        let b = Shape::scalar();
        assert_eq!(a.broadcast(&b), Some(Shape::d2(3, 4)));
    }

    #[test]
    fn broadcast_trailing_dim() {
        let a = Shape::d2(3, 4);
        let b = Shape::d1(4);
        assert_eq!(a.broadcast(&b), Some(Shape::d2(3, 4)));
    }

    #[test]
    fn broadcast_expand_ones() {
        let a = Shape::new(vec![3, 1]);
        let b = Shape::new(vec![1, 4]);
        assert_eq!(a.broadcast(&b), Some(Shape::d2(3, 4)));
    }

    #[test]
    fn broadcast_prepend_dims() {
        let a = Shape::d1(4);
        let b = Shape::d3(2, 3, 4);
        assert_eq!(a.broadcast(&b), Some(Shape::d3(2, 3, 4)));
    }

    #[test]
    fn broadcast_incompatible() {
        let a = Shape::d2(3, 4);
        let b = Shape::d2(2, 4);
        assert_eq!(a.broadcast(&b), None);
    }

    #[test]
    fn broadcasts_to_check() {
        let a = Shape::d1(4);
        let target = Shape::d2(3, 4);
        assert!(a.broadcasts_to(&target));
        
        let incompatible = Shape::d2(3, 5);
        assert!(!a.broadcasts_to(&incompatible));
    }
}