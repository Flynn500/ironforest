use crate::array::{NdArray, Shape};
use crate::linalg::basic::simd_dot;
use crate::random::Generator;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProjectionType {
    Gaussian,
    Sparse(f64),
    // Achlioptas,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProjectionDirection {
    Dense(Vec<f64>),
    Sparse(SparseVector),
    Empty,
}

impl ProjectionDirection {
    pub fn project(&self, point: &[f64]) -> f64 {
        match self {
            Self::Dense(v) => simd_dot(point, v),
            Self::Sparse(sv) => sv.dot(point),
            Self::Empty => 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
    pub dim: usize,
}

impl SparseVector {
    pub fn dot(&self, dense: &[f64]) -> f64 {
        debug_assert!(self.dim <= dense.len());
        self.indices.iter()
            .zip(&self.values)
            .map(|(&i, &v)| v * dense[i])
            .sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RandomProjection {
    pub input_dim: usize,
    pub output_dim: usize,
    pub projection_type: ProjectionType,
    pub matrix: NdArray<f64>,
}

impl RandomProjection {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        projection_type: ProjectionType,
        rng: &mut Generator,
    ) -> Self {
        let matrix = match projection_type {
            ProjectionType::Gaussian => Self::gaussian_matrix(input_dim, output_dim, rng),
            ProjectionType::Sparse(_) => unimplemented!("Sparse projection matrix not yet supported, use generate_direction for tree splits")
        };

        Self { input_dim, output_dim, projection_type, matrix }
    }

    fn gaussian_matrix(input_dim: usize, output_dim: usize, rng: &mut Generator) -> NdArray<f64> {
        let scale = 1.0 / (output_dim as f64).sqrt();
        let raw = rng.standard_normal(Shape::new(vec![input_dim, output_dim]));
        raw.map(|x| x * scale)
    }

    pub fn generate_direction(
        dim: usize,
        projection_type: ProjectionType,
        rng: &mut Generator,
    ) -> ProjectionDirection {
        match projection_type {
            ProjectionType::Gaussian => {
                let raw = rng.standard_normal(Shape::d1(dim));
                let slice = raw.as_slice_unchecked();
                let norm: f64 = slice.iter().map(|x| x * x).sum::<f64>().sqrt();
                ProjectionDirection::Dense(slice.iter().map(|x| x / norm).collect())
            }
            ProjectionType::Sparse(density) => {
                ProjectionDirection::Sparse(Self::generate_sparse_vector(dim, density, rng))
            }
        }
    }

    fn generate_sparse_vector(
        dim: usize,
        density: f64,
        rng: &mut Generator,
    ) -> SparseVector {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..dim {
            if rng.next_f64() < density {
                indices.push(i);
                values.push(rng.next_gaussian());
            }
        }

        if indices.is_empty() {
            let i = rng.next_u64() as usize % dim;
            indices.push(i);
            values.push(1.0);
        }

        let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        for v in &mut values {
            *v /= norm;
        }

        SparseVector { indices, values, dim }
    }


    pub fn project(&self, data: &NdArray<f64>) -> NdArray<f64> {
        assert_eq!(
            data.shape().dims()[1], self.input_dim,
            "Data dim {} doesn't match projection input_dim {}",
            data.shape().dims()[1], self.input_dim
        );
        data.dot(&self.matrix)
    }

    pub fn project_onto(data: &NdArray<f64>, vector: &[f64]) -> Vec<f64> {
        let dim = data.shape().dims()[1];
        assert_eq!(dim, vector.len(), "Data dim {} doesn't match vector len {}", dim, vector.len());

        let col = NdArray::from_vec(Shape::new(vec![dim, 1]), vector.to_vec());
        let result = data.dot(&col);
        result.as_slice_unchecked().to_vec()
    }

    pub fn rp_split(
        data: &NdArray<f64>,
        indices: &mut [usize],
        start: usize,
        end: usize,
        dim: usize,
        projection_type: ProjectionType,
        rng: &mut Generator,
    ) -> (ProjectionDirection, f64, usize) {
        let direction = Self::generate_direction(dim, projection_type, rng);

        let mut indexed: Vec<(f64, usize)> = (start..end)
            .map(|i| {
                let row = data.row(indices[i]);
                let proj = direction.project(row);
                (proj, indices[i])
            })
            .collect();

        let mid_offset = (end - start) / 2;
        indexed.select_nth_unstable_by(mid_offset, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });

        let split = indexed[mid_offset].0;

        for (i, &(_, idx)) in indexed.iter().enumerate() {
            indices[start + i] = idx;
        }

        let mut mid = start + mid_offset;
        if mid == start {
            mid = start + 1;
        } else if mid == end {
            mid = end - 1;
        }

        (direction, split, mid)
    }
}