use std::{fmt, ops::{self, Index, IndexMut}};

use crate::{Scalar, Vector, util::{array_from_index, zip_map}};


/// A `C`×`R` matrix with element type `T` (`C` many columns, `R` many rows).
/// Column-major memory layout.
///
/// *Note*: the `Debug` output (via `{:?}`) prints the matrix in row-major
/// order, i.e. row-by-row. This is more intuitive when reading matrices. You
/// can also use the "alternate" flat `#` (i.e. `{:#?}`) which avoids that
/// confusion by using one actual line per matrix row.
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

    /// Pairs up the same elements from `self` and `other`, applies the given
    /// function to each and returns the resulting matrix. Useful for
    /// element-wise operations.
    ///
    /// ```
    /// use lina::{Mat3f, vec3};
    ///
    /// let a = Mat3f::from_rows([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0],
    /// ]);
    /// let b = Mat3f::identity();
    /// let c = a.zip_map(b, |elem_a, elem_b| elem_a * elem_b);   // element-wise multiplication
    ///
    /// assert_eq!(c.row(0), vec3(1.0, 0.0, 0.0));
    /// assert_eq!(c.row(1), vec3(0.0, 5.0, 0.0));
    /// assert_eq!(c.row(2), vec3(0.0, 0.0, 9.0));
    /// ```
    pub fn zip_map<U, O, F>(self, other: Matrix<U, C, R>, mut f: F) -> Matrix<O, C, R>
    where
        U: Scalar,
        O: Scalar,
        F: FnMut(T, U) -> O,
    {
        Matrix(
            zip_map(self.0, other.0, |lcol, rcol| {
                zip_map(lcol, rcol, |l, r| f(l, r))
            })
        )
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
        Self::from_diagonal([T::one(); N])
    }

    /// Returns a matrix with the given diagonal and all other elements set to 0.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mat = Mat3::from_diagonal([1, 2, 3]);
    ///
    /// assert_eq!(mat.row(0), vec3(1, 0, 0));
    /// assert_eq!(mat.row(1), vec3(0, 2, 0));
    /// assert_eq!(mat.row(2), vec3(0, 0, 3));
    /// ```
    pub fn from_diagonal(v: impl Into<Vector<T, N>>) -> Self {
        let mut m = Self::zero();
        m.set_diagonal(v);
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

impl<T: Scalar, const C: usize, const R: usize> ops::Add for Matrix<T, C, R> {
    type Output = Matrix<T, C, R>;
    fn add(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |l, r| l + r)
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::AddAssign for Matrix<T, C, R> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Sub for Matrix<T, C, R> {
    type Output = Matrix<T, C, R>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.zip_map(rhs, |l, r| l - r)
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::SubAssign for Matrix<T, C, R> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<T> for Matrix<T, C, R> {
    type Output = Matrix<T, C, R>;
    fn mul(self, rhs: T) -> Self::Output {
        self.map(|elem| elem * rhs)
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::MulAssign<T> for Matrix<T, C, R> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Div<T> for Matrix<T, C, R> {
    type Output = Matrix<T, C, R>;
    fn div(self, rhs: T) -> Self::Output {
        self.map(|elem| elem / rhs)
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::DivAssign<T> for Matrix<T, C, R> {
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T: Scalar + ops::Neg, const C: usize, const R: usize> ops::Neg for Matrix<T, C, R>
where
    <T as ops::Neg>::Output: Scalar,
{
    type Output = Matrix<<T as ops::Neg>::Output, C, R>;
    fn neg(self) -> Self::Output {
        self.map(|elem| -elem)
    }
}

impl<T: Scalar, const C: usize, const R: usize> fmt::Debug for Matrix<T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        /// Helper type to format an array as a single line, regardless of
        /// alternate flag.
        struct Row<T, const N: usize>([T; N]);

        impl<T: fmt::Debug, const N: usize> fmt::Debug for Row<T, N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "[")?;
                for (i, elem) in self.0.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    elem.fmt(f)?;
                }
                write!(f, "]")
            }
        }

        write!(f, "Matrix ")?;
        let rows = (0..R).map(|i| Row(self.row(i).into()));
        f.debug_list().entries(rows).finish()
    }
}


#[cfg(test)]
mod tests;
