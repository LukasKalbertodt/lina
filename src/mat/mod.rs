use std::ops::{Index, IndexMut};

use crate::{Scalar, Vector};


/// A `C`×`R` matrix with element type `T` (`C` many columns, `R` many rows).
/// Column-major memory layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Matrix<T: Scalar, const C: usize, const R: usize>([[T; R]; C]);

impl<T: Scalar, const C: usize, const R: usize> Matrix<T, C, R> {
    /// Returns a matrix with all elements being zero.
    pub fn zero() -> Self {
        Self([[T::zero(); R]; C])
    }

    // TODO:
    // - col/col_mut + row + set_row
}

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// Returns the identity matrix with all elements 0, except the diagonal
    /// which is all 1.
    pub fn identity() -> Self {
        let mut m = Self::zero();
        m.set_diagonal([T::one(); N].into());
        m
    }

    /// Returns the diagonal of this matrix.
    pub fn diagonal(&self) -> Vector<T, N> {
        // We use the fact that `array::map` visits each element in order. That
        // way, we can advance an index.
        let mut row = 0;
        self.0.map(|col| {
            let i = row;
            row += 1;
            col[i]
        }).into()
    }

    /// Sets the diagonal to the given vector.
    pub fn set_diagonal(&mut self, v: Vector<T, N>) {
        for i in 0..N {
            self[i][i] = v[i];
        }
    }
}


impl<T: Scalar, const C: usize, const R: usize> Index<usize> for Matrix<T, C, R> {
    type Output = [T; R];
    fn index(&self, idx: usize) -> &Self::Output {
        &self.0[idx]
    }
}

impl<T: Scalar, const C: usize, const R: usize> IndexMut<usize> for Matrix<T, C, R> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.0[idx]
    }
}
