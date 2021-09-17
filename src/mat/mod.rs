use std::ops::{Index, IndexMut};

use crate::{Scalar, Vector, util::array_from_index};


/// A `C`×`R` matrix with element type `T` (`C` many columns, `R` many rows).
/// Column-major memory layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Matrix<T: Scalar, const C: usize, const R: usize>([[T; R]; C]);

/// A 4×4 matrix.
pub type Mat4<T> = Matrix<T, 4, 4>;
/// A 3×3 matrix.
pub type Mat3<T> = Matrix<T, 3, 3>;
/// A 2×2 matrix.
pub type Mat2<T> = Matrix<T, 2, 2>;

/// A 4×4 matrix with `f32` elements.
pub type Mat4f = Mat4<f32>;
/// A 4×4 matrix with `f64` elements.
pub type Mat4d = Mat4<f64>;

/// A 3×3 matrix with `f32` elements.
pub type Mat3f = Mat3<f32>;
/// A 3×3 matrix with `f63` elements.
pub type Mat3d = Mat3<f64>;

/// A 2×2 matrix with `f32` elements.
pub type Mat2f = Mat2<f32>;
/// A 2×2 matrix with `f62` elements.
pub type Mat2d = Mat2<f64>;


impl<T: Scalar, const C: usize, const R: usize> Matrix<T, C, R> {
    /// Returns a matrix with all elements being zero.
    pub fn zero() -> Self {
        Self([[T::zero(); R]; C])
    }

    /// Returns a matrix with the specified rows. This is opposite of the memory
    /// layout (which is column-major), but using this constructor usually
    /// leads to easier-to-read code as the element order matches code order.
    ///
    /// ```
    /// use lina::{Matrix, vec3};
    ///
    /// let m = <Matrix<_, 3, 2>>::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    ///
    /// assert_eq!(m.row(0), vec3(1, 2, 3));
    /// assert_eq!(m.row(1), vec3(4, 5, 6));
    /// ```
    pub fn from_rows<V>(rows: [V; R]) -> Self
    where
        V: Into<Vector<T, C>>,
    {
        let mut out = Self::zero();
        for (i, row) in IntoIterator::into_iter(rows).enumerate() {
            out.set_row(i, row);
        }
        out
    }

    /// Returns the column with index `idx`.
    pub fn col(&self, idx: usize) -> Vector<T, R> {
        self[idx].into()
    }

    /// Sets the column with index `idx` to the given vector.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::identity();
    /// mat.set_col(1, vec3(2, 4, 6));
    ///
    /// assert_eq!(mat.row(0), vec3(1, 2, 0));
    /// assert_eq!(mat.row(1), vec3(0, 4, 0));
    /// assert_eq!(mat.row(2), vec3(0, 6, 1));
    /// ```
    pub fn set_col(&mut self, idx: usize, v: impl Into<Vector<T, R>>) {
        self[idx] = v.into().into();
    }

    /// Returns the row with index `idx`.
    pub fn row(&self, idx: usize) -> Vector<T, C> {
        array_from_index(|i| self[i][idx]).into()
    }

    /// Sets the row with index `idx` to the given vector.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::identity();
    /// mat.set_row(1, vec3(2, 4, 6));
    ///
    /// assert_eq!(mat.row(0), vec3(1, 0, 0));
    /// assert_eq!(mat.row(1), vec3(2, 4, 6));
    /// assert_eq!(mat.row(2), vec3(0, 0, 1));
    /// ```
    pub fn set_row(&mut self, idx: usize, v: impl Into<Vector<T, C>>) {
        let v = v.into();
        for i in 0..C {
            self[i][idx] = v[i];
        }
    }

    /// Applies the given function to each element and returns the resulting new
    /// matrix.
    ///
    /// ```
    /// use lina::{Mat2, vec2};
    ///
    /// let mat = Mat2::identity().map(|e: i32| e + 1);
    ///
    /// assert_eq!(mat.row(0), vec2(2, 1));
    /// assert_eq!(mat.row(1), vec2(1, 2));
    /// ```
    pub fn map<U: Scalar, F: FnMut(T) -> U>(self, mut f: F) -> Matrix<U, C, R> {
        Matrix(self.0.map(|col| col.map(&mut f)))
    }
}

impl<T: Scalar, const N: usize> Matrix<T, N, N> {
    /// Returns the identity matrix with all elements 0, except the diagonal
    /// which is all 1.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mat = Mat3::identity();
    /// assert_eq!(mat.row(0), vec3(1.0, 0.0, 0.0));
    /// assert_eq!(mat.row(1), vec3(0.0, 1.0, 0.0));
    /// assert_eq!(mat.row(2), vec3(0.0, 0.0, 1.0));
    /// ```
    pub fn identity() -> Self {
        let mut m = Self::zero();
        m.set_diagonal([T::one(); N]);
        m
    }

    /// Returns the diagonal of this matrix.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mat = Mat3::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    /// assert_eq!(mat.diagonal(), vec3(1, 5, 9));
    /// ```
    pub fn diagonal(&self) -> Vector<T, N> {
        array_from_index(|i| self[i][i]).into()
    }

    /// Sets the diagonal to the given vector.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    /// mat.set_diagonal(vec3(2, 1, 0));
    ///
    /// assert_eq!(mat.row(0), vec3(2, 2, 3));
    /// assert_eq!(mat.row(1), vec3(4, 1, 6));
    /// assert_eq!(mat.row(2), vec3(7, 8, 0));
    /// ```
    pub fn set_diagonal(&mut self, v: impl Into<Vector<T, N>>) {
        let v = v.into();
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
