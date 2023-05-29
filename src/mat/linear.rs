use std::{array, fmt, ops};
use bytemuck::{Pod, Zeroable};

use crate::{Point, Scalar, Vector, Float, cross, HcMatrix, HcPoint};


/// A `C`×`R` matrix with element type `T` (`C` many columns, `R` many rows).
/// Column-major memory layout.
///
/// This type does not implement `ops::Index[Mut]`. Instead, there are two main
/// ways to access elements. Use whatever you prefer, but keep in mind that
/// code is read more often than it is written, so the first option is likely
/// better as it avoids ambiguity.
/// - [`Self::row`] and [`Self::col`]: `matrix.row(2).col(0)`
/// - [`Self::elem`]: `matrix.elem(2, 0)`
///
///
/// *Note*: the `Debug` output (via `{:?}`) prints the matrix in row-major
/// order, i.e. row-by-row. This is more intuitive when reading matrices. You
/// can also use the "alternate" flag `#` (i.e. `{:#?}`) which avoids that
/// confusion by using one actual line per matrix row.
///
///
/// # Matrices as transformations
///
/// Matrices in computer graphics are usually used to represent and carry out
/// *transformations*. In its simplest form, a matrix can represent a
/// *linear* transformation, which includes rotation and scaling, but *not*
/// translation. To learn more about this, I strongly recommend watching
/// [3blue1brown's series "Essence of Linear Algebra"][3b1b-lina], in
/// particular [Chapter 3: Linear transformations and matrices][3b1b-transform].
///
/// ## Homogeneous coordinates & non-linear transformations
///
/// In computer graphics, non-linear transformations (like translations and
/// perspective projection) are often important as well. To be able to
/// represent those in a matrix (along with linear transformations), we use
/// [*homogeneous coordinates*][hc-wiki] (instead of standard cartesian
/// coordinates). To do that, we increase the dimension of our vectors and
/// matrices by 1, e.g. having a 4D vector and 4x4 matrices to talk about 3D
/// space. This allows matrices to represent *affine* and perspective
/// projection transformations.
///
/// ## Transforming a point or vector
///
/// Mathematically, to apply a transformation to a vector/point, you multiply
/// the matrix with the vector/point: `matrix * vec`. The relevant operator is
/// defined, so you can just do that in Rust as well.
///
/// However, you can also use methods to transform vectors/points:
///
/// - [`Matrix::transform_vec`] and [`Matrix::transform_point`]
/// - [`Matrix::transform_hc_vec`] and [`Matrix::transform_hc_point`]
///
/// The `_hc` (homogeneous coordinate) versions take a vector/point of dimension
/// `N - 1` and correctly perform the perspective divide after the
/// transformation.
///
///
/// ## Combining transformations
///
/// Oftentimes you want to apply multiple transformations to a set of points or
/// vectors. You can save processing time by combining all
/// transformation-matrices into a single matrix. That's the beauty and
/// convenience of representing transformations as matrix: it's always possible
/// to combine all of them into a single matrix.
///
/// Mathematically, this composition is *matrix multiplication*: `A * B` results
/// in a matrix that represents the combined transformation of *first* `B`,
/// *then* `A`. Matrix multiplication is not commutative, i.e. the order of
/// operands matters. And it's also in an non-intuitive order, with the
/// rightmost transformation being applied first. For that reason, you can also
/// use [`Matrix::and_then`] instead of the overloaded operator `*`. Use what
/// you think is easier to read.
///
///
/// [3b1b-lina]: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
/// [3b1b-transform]: https://www.youtube.com/watch?v=kYB8IZa5AuE
/// [hc-wiki]: https://en.wikipedia.org/wiki/Homogeneous_coordinates#Use_in_computer_graphics_and_computer_vision
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Matrix<T: Scalar, const C: usize, const R: usize>(pub(super) [[T; R]; C]);

/// A 3×3 matrix.
pub type Mat3<T> = Matrix<T, 3, 3>;
/// A 2×2 matrix.
pub type Mat2<T> = Matrix<T, 2, 2>;

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
    /// assert_eq!(m.row(0).to_array(), [1, 2, 3]);
    /// assert_eq!(m.row(1).to_array(), [4, 5, 6]);
    /// ```
    pub fn from_rows<V>(rows: [V; R]) -> Self
    where
        V: Into<[T; C]>,
    {
        let mut out = Self::zero();
        for (i, row) in IntoIterator::into_iter(rows).enumerate() {
            out.set_row(i, row);
        }
        out
    }

    /// Returns a matrix with the specified columns. This matches the memory
    /// layout but is usually more difficult to read in code. Consider using
    /// [`Matrix::from_rows`] instead.
    ///
    /// ```
    /// use lina::{Matrix, vec2};
    ///
    /// let m = <Matrix<_, 2, 3>>::from_cols([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    ///
    /// assert_eq!(m.row(0).to_array(), [1, 4]);
    /// assert_eq!(m.row(1).to_array(), [2, 5]);
    /// assert_eq!(m.row(2).to_array(), [3, 6]);
    /// ```
    pub fn from_cols<V>(cols: [V; C]) -> Self
    where
        V: Into<[T; R]>,
    {
        Self(cols.map(|v| v.into()))
    }

    /// Returns the column with index `idx`.
    pub fn col(&self, index: usize) -> Col<'_, T, C, R> {
        Col {
            matrix: self,
            index,
        }
    }

    /// Sets the column with index `idx` to the given values.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::identity();
    /// mat.set_col(1, vec3(2, 4, 6));
    ///
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [1, 2, 0],
    ///     [0, 4, 0],
    ///     [0, 6, 1],
    /// ]));
    /// ```
    pub fn set_col(&mut self, idx: usize, v: impl Into<[T; R]>) {
        self.0[idx] = v.into().into();
    }

    /// Returns the row with index `idx`.
    pub fn row(&self, index: usize) -> Row<'_, T, C, R> {
        Row {
            matrix: self,
            index,
        }
    }

    /// Sets the row with index `idx` to the given values.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::identity();
    /// mat.set_row(1, vec3(2, 4, 6));
    ///
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [1, 0, 0],
    ///     [2, 4, 6],
    ///     [0, 0, 1],
    /// ]));
    /// ```
    pub fn set_row(&mut self, idx: usize, v: impl Into<[T; C]>) {
        let v = v.into();
        for i in 0..C {
            self.0[i][idx] = v[i];
        }
    }

    /// Returns the element at the given (row, column). Panics if either index
    /// is out of bounds.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mat = Mat3::from_rows([
    ///     [1, 0, 8],
    ///     [0, 9, 0],
    ///     [0, 7, 1],
    /// ]);
    ///
    /// assert_eq!(mat.elem(1, 1), 9);
    /// assert_eq!(mat.elem(0, 2), 8);
    /// assert_eq!(mat.elem(2, 1), 7);
    ///
    /// ```
    pub fn elem(&self, row: usize, col: usize) -> T {
        self.0[col][row]
    }

    /// Overwrites the element in the given (row, column) with the given value.
    /// Panics if `row` or `col` is out of bounds.
    ///
    /// ```
    /// use lina::{Mat3, vec3};
    ///
    /// let mut mat = Mat3::identity();
    /// mat.set_elem(1, 1, 9);
    /// mat.set_elem(0, 2, 8);
    /// mat.set_elem(2, 1, 7);
    ///
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [1, 0, 8],
    ///     [0, 9, 0],
    ///     [0, 7, 1],
    /// ]));
    /// ```
    pub fn set_elem(&mut self, row: usize, col: usize, value: T) {
        self.0[col][row] = value;
    }

    /// Returns an iterator over all entries of this matrix, in column-major order.
    ///
    /// ```
    /// let m = lina::Matrix::from_rows([
    ///     [1, 2],
    ///     [3, 4],
    ///     [5, 6],
    /// ]);
    /// let mut it = m.iter();
    ///
    /// assert_eq!(it.next(), Some(1));
    /// assert_eq!(it.next(), Some(3));
    /// assert_eq!(it.next(), Some(5));
    /// assert_eq!(it.next(), Some(2));
    /// assert_eq!(it.next(), Some(4));
    /// assert_eq!(it.next(), Some(6));
    /// assert_eq!(it.next(), None);
    /// ```
    pub fn iter(self) -> impl Iterator<Item = T> {
        self.0.into_iter().flat_map(|col| col)
    }

    /// Returns the transposed version of this matrix (swapping rows and
    /// columns). Also see [`Matrix::transpose`].
    ///
    /// ```
    /// use lina::{Matrix, vec2};
    ///
    /// let m = Matrix::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    /// let t = m.transposed();
    ///
    /// assert_eq!(t, Matrix::from_rows([
    ///     [1, 4],
    ///     [2, 5],
    ///     [3, 6],
    /// ]));
    /// ```
    #[must_use = "to transpose in-place, use `Matrix::transpose`, not `transposed`"]
    pub fn transposed(self) -> Matrix<T, R, C> {
        Matrix::from_rows(self.0)
    }

    /// Transforms the given point or vector with this matrix (numerically just
    /// a simple matrix-vector-multiplication).
    ///
    /// This function accepts `C`-dimensional [`Point`]s, [`Vector`]s and
    /// [`HcPoint`]s. For the latter, only its `coords` part are transformed
    /// with the `weight` untouched. To be clear: as this matrix represents a
    /// linear transformation in cartesian coordinates, no perspective divide
    /// is performed. Use [`HcMatrix`] if you want to represent non-linear
    /// transformations.
    ///
    /// Instead of using this function, you can also use the `*` operator
    /// overload, if you prefer. It does exactly the same.
    pub fn transform<'a, X>(&'a self, x: X) -> <&'a Self as ops::Mul<X>>::Output
    where
        &'a Self: ops::Mul<X>,
    {
        self * x
    }

    /// Combines the transformations of two matrices into a single
    /// transformation matrix. First the transformation of `self` is applied,
    /// then the one of `second`. In the language of math, this is just matrix
    /// multiplication: `second * self`. You can also use the overloaded `*`
    /// operator instead, but this method exists for those who find the matrix
    /// multiplication order unintuitive and this method call easier to read.
    ///
    /// ```
    /// use lina::Mat2;
    ///
    /// // Rotation by 90° counter clock wise.
    /// let rotate = Mat2::from_rows([
    ///     [0, -1],
    ///     [1,  0],
    /// ]);
    ///
    /// // Scale x-axis by 2, y-axis by 3.
    /// let scale = Mat2::from_rows([
    ///     [2, 0],
    ///     [0, 3],
    /// ]);
    ///
    /// assert_eq!(rotate.and_then(scale), Mat2::from_rows([
    ///     [0, -2],
    ///     [3,  0],
    /// ]));
    /// assert_eq!(scale.and_then(rotate), Mat2::from_rows([
    ///     [0, -3],
    ///     [2,  0],
    /// ]));
    /// ```
    pub fn and_then<const R2: usize>(self, second: Matrix<T, R, R2>) -> Matrix<T, C, R2> {
        second * self
    }

    pub fn to_homogeneous(&self) -> HcMatrix<T, C, R> {
        HcMatrix::from_parts(*self, Vector::zero(), Vector::zero(), T::one())
    }

    /// Applies the given function to each element and returns the resulting new
    /// matrix.
    ///
    /// ```
    /// use lina::{Mat2, vec2};
    ///
    /// let mat = Mat2::identity().map(|e: i32| e + 1);
    ///
    /// assert_eq!(mat, Mat2::from_rows([
    ///     [2, 1],
    ///     [1, 2],
    /// ]));
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
    /// assert_eq!(c, Mat3f::from_rows([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 5.0, 0.0],
    ///     [0.0, 0.0, 9.0],
    /// ]));
    /// ```
    pub fn zip_map<U, O, F>(self, other: Matrix<U, C, R>, mut f: F) -> Matrix<O, C, R>
    where
        U: Scalar,
        O: Scalar,
        F: FnMut(T, U) -> O,
    {
        Matrix(array::from_fn(|i| array::from_fn(|j| f(self.0[i][j], other.0[i][j]))))
    }

    /// Returns a byte slice of this matrix, representing the raw column-major
    /// data. Useful to pass to graphics APIs.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
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
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ]));
    /// ```
    pub fn identity() -> Self {
        Self::from_diagonal([T::one(); N])
    }

    /// Returns a matrix with the given diagonal and all other elements set to 0.
    ///
    /// ```
    /// use lina::Mat3;
    ///
    /// let mat = Mat3::from_diagonal([1, 2, 3]);
    ///
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [1, 0, 0],
    ///     [0, 2, 0],
    ///     [0, 0, 3],
    /// ]));
    /// ```
    pub fn from_diagonal(v: impl Into<[T; N]>) -> Self {
        let mut m = Self::zero();
        m.set_diagonal(v);
        m
    }

    /// Returns the diagonal of this matrix.
    ///
    /// ```
    /// use lina::Mat3;
    ///
    /// let mat = Mat3::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    /// assert_eq!(mat.diagonal(), [1, 5, 9]);
    /// ```
    pub fn diagonal(&self) -> [T; N] {
        array::from_fn(|i| self.0[i][i])
    }

    /// Sets the diagonal to the given values.
    ///
    /// ```
    /// use lina::Mat3;
    ///
    /// let mut mat = Mat3::from_rows([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    /// mat.set_diagonal([2, 1, 0]);
    ///
    /// assert_eq!(mat, Mat3::from_rows([
    ///     [2, 2, 3],
    ///     [4, 1, 6],
    ///     [7, 8, 0],
    /// ]));
    /// ```
    pub fn set_diagonal(&mut self, v: impl Into<[T; N]>) {
        let v = v.into();
        for i in 0..N {
            self.0[i][i] = v[i];
        }
    }

    /// Transposes this matrix in-place. Also see [`Matrix::transposed`].
    pub fn transpose(&mut self) {
        *self = self.transposed();
    }

    /// Checks whether this matrix is *symmetric*, i.e. whether transposing
    /// does *not* change the matrix.
    ///
    /// ```
    /// use lina::{Mat3f, Mat2};
    ///
    /// assert!(Mat3f::identity().is_symmetric());
    /// assert!(!Mat2::from_rows([[1, 2], [3, 4]]).is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        for c in 1..N {
            for r in 0..c {
                if self.0[c][r] != self.0[r][c] {
                    return false;
                }
            }
        }

        true
    }

    /// Returns the *trace* of the matrix, i.e. the sum of all elements on the
    /// diagonal.
    ///
    /// ```
    /// use lina::{Mat2, Mat3f};
    ///
    /// assert_eq!(Mat3f::identity().trace(), 3.0);
    /// assert_eq!(Mat2::from_rows([[1, 2], [3, 4]]).trace(), 5);
    /// ```
    pub fn trace(&self) -> T {
        self.diagonal().as_ref().iter().fold(T::zero(), |a, b| a + *b)
    }
}

impl<T: Float> Matrix<T, 1, 1> {
    /// Returns the determinant of this matrix. This number gives a general idea
    /// of whether the transformation represented by this matrix squishes or
    /// stretches space. If this is 0, the transformation's output space has a
    /// lower dimension than the input space and is thus non-invertible. See
    /// [this video][1] for a good explanation.
    ///
    /// [1]: https://www.youtube.com/watch?v=Ip3X9LOh2dk
    pub fn determinant(self) -> T {
        self.0[0][0]
    }

    /// Returns the inverse of this matrix, if it exists. If the determinant of
    /// this matrix is 0, then it is not invertible and `None` is returned. The
    /// inverted matrix "undoes" the transformation of `self`.
    pub fn inverted(self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            return None;
        }

        Some(Self::identity() / det)
    }
}

impl<T: Float> Matrix<T, 2, 2> {
    /// Returns the determinant of this matrix. This number gives a general idea
    /// of whether the transformation represented by this matrix squishes or
    /// stretches space. If this is 0, the transformation's output space has a
    /// lower dimension than the input space and is thus non-invertible. See
    /// [this video][1] for a good explanation.
    ///
    /// [1]: https://www.youtube.com/watch?v=Ip3X9LOh2dk
    pub fn determinant(self) -> T {
        self.0[0][0] * self.0[1][1] - self.0[0][1] * self.0[1][0]
    }

    /// Returns the inverse of this matrix, if it exists. If the determinant of
    /// this matrix is 0, then it is not invertible and `None` is returned. The
    /// inverted matrix "undoes" the transformation of `self`.
    pub fn inverted(self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            return None;
        }

        let m = Self::from_rows([
            [ self.row(1).col(1), -self.row(0).col(1)],
            [-self.row(1).col(0),  self.row(0).col(0)],
        ]);
        Some(m / det)
    }
}

impl<T: Float> Matrix<T, 3, 3> {
    /// Returns the determinant of this matrix. This number gives a general idea
    /// of whether the transformation represented by this matrix squishes or
    /// stretches space. If this is 0, the transformation's output space has a
    /// lower dimension than the input space and is thus non-invertible. See
    /// [this video][1] for a good explanation.
    ///
    /// [1]: https://www.youtube.com/watch?v=Ip3X9LOh2dk
    pub fn determinant(self) -> T {
        T::zero()
            + self.0[0][0] * (self.0[1][1] * self.0[2][2] - self.0[2][1] * self.0[1][2])
            + self.0[1][0] * (self.0[2][1] * self.0[0][2] - self.0[0][1] * self.0[2][2])
            + self.0[2][0] * (self.0[0][1] * self.0[1][2] - self.0[1][1] * self.0[0][2])
    }

    /// Returns the inverse of this matrix, if it exists. If the determinant of
    /// this matrix is 0, then it is not invertible and `None` is returned. The
    /// inverted matrix "undoes" the transformation of `self`.
    pub fn inverted(self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            return None;
        }

        let m = Self::from_rows([
            cross(self.col(1).to_vec(), self.col(2).to_vec()),
            cross(self.col(2).to_vec(), self.col(0).to_vec()),
            cross(self.col(0).to_vec(), self.col(1).to_vec()),
        ]);
        Some(m / det)
    }
}

impl<T: Float> Matrix<T, 4, 4> {
    /// Returns the determinant of this matrix. This number gives a general idea
    /// of whether the transformation represented by this matrix squishes or
    /// stretches space. If this is 0, the transformation's output space has a
    /// lower dimension than the input space and is thus non-invertible. See
    /// [this video][1] for a good explanation.
    ///
    /// [1]: https://www.youtube.com/watch?v=Ip3X9LOh2dk
    pub fn determinant(self) -> T {
        super::inv4::det(&self)
    }

    /// Returns the inverse of this matrix, if it exists. If the determinant of
    /// this matrix is 0, then it is not invertible and `None` is returned. The
    /// inverted matrix "undoes" the transformation of `self`.
    pub fn inverted(self) -> Option<Self> {
        super::inv4::inv(&self)
    }
}


// =============================================================================================
// ===== Non-mathematical trait impls
// =============================================================================================

/// The inner array implements `Zeroable` and `Matrix` is just a newtype wrapper
/// around that array with `repr(transparent)`.
unsafe impl<T: Scalar + Zeroable, const C: usize, const R: usize> Zeroable for Matrix<T, C, R> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[[T; R]; C]`. And `bytemuck` itself has an impl for arrays
/// where `T: Pod`.
unsafe impl<T: Scalar + Pod, const C: usize, const R: usize> Pod for Matrix<T, C, R> {}

impl<T: Scalar, const C: usize, const R: usize> fmt::Debug for Matrix<T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix ")?;
        super::debug_matrix_impl(f, C, R, |r, c| self.elem(r, c))
    }
}


// =============================================================================================
// ===== Mathematical trait impls
// =============================================================================================

impl<T: Scalar, const C: usize, const R: usize> ops::Add for Matrix<T, C, R> {
    type Output = Self;
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
    type Output = Self;
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
    type Output = Self;
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
    type Output = Self;
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

// Scalar multiplication: `scalar * matrix`. Unfortunately, due to Rust's orphan
// rules, this cannot be implemented generically. So we just implement it for
// core primitive types.
macro_rules! impl_scalar_mul {
    ($($ty:ident),*) => {
        $(
            impl<const C: usize, const R: usize> ops::Mul<Matrix<$ty, C, R>> for $ty {
                type Output = Matrix<$ty, C, R>;
                fn mul(self, rhs: Matrix<$ty, C, R>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl<T: Scalar, const C: usize, const R: usize> std::iter::Sum<Self> for Matrix<T, C, R> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}


// =============================================================================================
// ===== Matrix * matrix multiplication (composition)
// =============================================================================================

/// `matrix * matrix` multiplication. You can also use [`Matrix::and_then`].
impl<T: Scalar, const C: usize, const R: usize, const S: usize> ops::Mul<Matrix<T, C, S>>
    for Matrix<T, S, R>
{
    type Output = Matrix<T, C, R>;
    fn mul(self, rhs: Matrix<T, C, S>) -> Self::Output {
        // This is the straight-forward n³ algorithm. Using more sophisticated
        // algorithms with sub cubic runtime is not worth it for small
        // matrices. However, this can certainly be micro-optimized. In
        // particular, using SSE seems like a good idea, but "requires" all
        // columns to be properly aligned in memory.
        //
        // TODO: try to optimize
        let mut out = Self::Output::zero();
        for c in 0..C {
            for r in 0..R {
                for s in 0..S {
                    out.0[c][r] += self.0[s][r] * rhs.0[c][s];
                }
            }
        }
        out
    }
}


// =============================================================================================
// ===== Matrix * vector multiplication (transformations)
// =============================================================================================

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<Vector<T, C>> for &Matrix<T, C, R> {
    type Output = Vector<T, R>;
    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        array::from_fn(|row| {
            (0..C)
                .map(|col| self.elem(row, col) * rhs[col])
                .fold(T::zero(), |acc, e| acc + e)
        }).into()
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<Point<T, C>> for &Matrix<T, C, R> {
    type Output = Point<T, R>;
    fn mul(self, rhs: Point<T, C>) -> Self::Output {
        (self * rhs.to_vec()).to_point()
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<HcPoint<T, C>> for &Matrix<T, C, R> {
    type Output = HcPoint<T, R>;
    fn mul(self, rhs: HcPoint<T, C>) -> Self::Output {
        HcPoint::new(self * Vector::from(rhs.coords), rhs.weight)
    }
}


// =============================================================================================
// ===== `Row` and `Col` proxies
// =============================================================================================

/// Proxy type representing one row of a matrix.
#[derive(Clone, Copy)]
pub struct Row<'a, T: Scalar, const C: usize, const R: usize> {
    matrix: &'a Matrix<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> Row<'a, T, C, R> {
    /// Indexes into this row with the given column index, returning the element.
    pub fn col(self, col: usize) -> T {
        self.matrix.0[col][self.index]
    }

    /// Returns this row as array.
    pub fn to_array(self) -> [T; C] {
        self.into()
    }

    /// Returns this row as vector.
    pub fn to_vec(self) -> Vector<T, C> {
        self.into()
    }

    /// Returns this row as point.
    pub fn to_point(self) -> Point<T, C> {
        self.into()
    }
}

/// Proxy type representing one column of a matrix.
#[derive(Clone, Copy)]
pub struct Col<'a, T: Scalar, const C: usize, const R: usize> {
    matrix: &'a Matrix<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> Col<'a, T, C, R> {
    /// Indexes into this column with the given row index, returning the element.
    pub fn row(self, row: usize) -> T {
        self.matrix.0[self.index][row]
    }

    /// Returns this column as array.
    pub fn to_array(self) -> [T; R] {
        self.into()
    }

    /// Returns this column as vector.
    pub fn to_vec(self) -> Vector<T, R> {
        self.into()
    }

    /// Returns this column as point.
    pub fn to_point(self) -> Point<T, R> {
        self.into()
    }
}

impl<'a, T: Scalar, const C: usize, const R: usize> From<Row<'a, T, C, R>> for [T; C] {
    fn from(src: Row<'a, T, C, R>) -> Self {
        array::from_fn(|i| src.matrix.0[i][src.index])
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> From<Row<'a, T, C, R>> for Vector<T, C> {
    fn from(src: Row<'a, T, C, R>) -> Self {
        src.to_array().into()
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> From<Row<'a, T, C, R>> for Point<T, C> {
    fn from(src: Row<'a, T, C, R>) -> Self {
        src.to_array().into()
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> fmt::Debug for Row<'a, T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::debug_list_one_line(&self.to_array(), f)
    }
}

impl<'a, T: Scalar, const C: usize, const R: usize> From<Col<'a, T, C, R>> for [T; R] {
    fn from(src: Col<'a, T, C, R>) -> Self {
        src.matrix.0[src.index]
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> From<Col<'a, T, C, R>> for Vector<T, R> {
    fn from(src: Col<'a, T, C, R>) -> Self {
        src.to_array().into()
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> From<Col<'a, T, C, R>> for Point<T, R> {
    fn from(src: Col<'a, T, C, R>) -> Self {
        src.to_array().into()
    }
}
impl<'a, T: Scalar, const C: usize, const R: usize> fmt::Debug for Col<'a, T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::debug_list_one_line(&self.to_array(), f)

    }
}
