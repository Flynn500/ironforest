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
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_constructors() {
        assert_eq!(Shape::scalar().dims(), &[]);
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
}
