#[derive(Debug, Clone)]
pub struct Storage<T> {
    data: Vec<T>,
}

impl<T> Storage<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        Storage { data }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Storage {
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }
}

impl<T: Clone> Storage<T> {
    pub fn filled(value: T, len: usize) -> Self {
        Storage {
            data: vec![value; len],
        }
    }
}

impl<T: Default + Clone> Storage<T> {
    pub fn zeros(len: usize) -> Self {
        Storage {
            data: vec![T::default(); len],
        }
    }
}