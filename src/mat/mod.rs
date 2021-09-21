use std::{fmt, ops::{self, Index, IndexMut, RangeInclusive}};

use bytemuck::{Pod, Zeroable};

use crate::{Float, Point3, Radians, Scalar, Vec3, Vector, cross, dot, util::{array_from_index, zip_map}};


/// A `C`×`R` matrix with element type `T` (`C` many columns, `R` many rows).
/// Column-major memory layout.
///
/// *Note*: the `Debug` output (via `{:?}`) prints the matrix in row-major
/// order, i.e. row-by-row. This is more intuitive when reading matrices. You
/// can also use the "alternate" flat `#` (i.e. `{:#?}`) which avoids that
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
/// ## Homogeneous coordinates & affine transformations
///
/// In computer graphics, non-linear transformations (like translations and
/// perspective projection) are often important as well. To be able to
/// represent those in a matrix (along with linear transformations), we use
/// [*homogeneous coordinates*][hc-wiki] (instead of standard cartesian
/// coordinates). To do that, we increase the dimension of our vectors and
/// matrices by 1, e.g. having a 4D vector and 4x4 matrices to talk about 3D
/// space. This allows matrices to represent *affine* transformations.
///
/// There are a couple of associated functions returning transformation
/// matrices. Be mindful of the prefix `hc_` or `cc_` which denotes the intended
/// coordinate system. If you're working in 3D space, you usually want to use
/// `hc_` versions for 4x4 matrices and `cc_` versions for 3x3 matrices.
///
/// ## Transforming a point or vector
///
/// TODO
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
/// *then* `A`. Yes, matrix multiplication is not commutative, i.e. the order of
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
    /// assert_eq!(m.row(0), vec2(1, 4));
    /// assert_eq!(m.row(1), vec2(2, 5));
    /// assert_eq!(m.row(2), vec2(3, 6));
    /// ```
    pub fn from_cols<V>(cols: [V; C]) -> Self
    where
        V: Into<Vector<T, R>>,
    {
        Self(cols.map(|v| v.into().into()))
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
    /// assert_eq!(t.row(0), vec2(1, 4));
    /// assert_eq!(t.row(1), vec2(2, 5));
    /// assert_eq!(t.row(2), vec2(3, 6));
    /// ```
    #[must_use = "to transpose in-place, use `Matrix::transpose`, not `transposed`"]
    pub fn transposed(self) -> Matrix<T, R, C> {
        Matrix::from_rows(self.0)
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
                if self[c][r] != self[r][c] {
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

    /// Returns a transformation matrix for homogeneous coordinates that scales
    /// all axis by `factor`. For the cartesian coordinate version, see
    /// [`Matrix::cc_scale`]. Example for `Mat4` (with `f` being `factor`):
    ///
    /// ```text
    /// ⎡ f 0 0 0 ⎤
    /// ⎢ 0 f 0 0 ⎥
    /// ⎢ 0 0 f 0 ⎥
    /// ⎣ 0 0 0 1 ⎦
    /// ```
    ///
    /// ```
    /// use lina::{Mat4f, vec4};
    ///
    /// let m = Mat4f::hc_scale(3.5);
    ///
    /// assert_eq!(m.row(0), vec4(3.5, 0.0, 0.0, 0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 3.5, 0.0, 0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, 3.5, 0.0));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, 0.0, 1.0));
    /// ```
    pub fn hc_scale(factor: T) -> Self {
        let mut diag = [factor; N];
        diag[N - 1] = T::one();
        Self::from_diagonal(diag)
    }

    /// Returns a transformation matrix for homogeneous coordinates that scales
    /// axis according to `factors`. For the cartesian coordinate version, see
    /// [`Matrix::cc_nonuniform_scale`]. Example for `Mat4` (with `factors`
    /// being `[x, y, z]`):
    ///
    /// ```text
    /// ⎡ x 0 0 0 ⎤
    /// ⎢ 0 y 0 0 ⎥
    /// ⎢ 0 0 z 0 ⎥
    /// ⎣ 0 0 0 1 ⎦
    /// ```
    ///
    /// ```
    /// use lina::{Mat4f, vec4};
    ///
    /// let m = Mat4f::hc_nonuniform_scale([2.0f32, 3.0, 8.0]);
    ///
    /// assert_eq!(m.row(0), vec4(2.0, 0.0, 0.0, 0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 3.0, 0.0, 0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, 8.0, 0.0));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, 0.0, 1.0));
    /// ```
    pub fn hc_nonuniform_scale(factors: impl Into<Vector<T, { N - 1 }>>) -> Self {
        let mut diag = [T::one(); N];
        let factors = factors.into();
        for i in 0..N - 1 {
            diag[i] = factors[i];
        }
        Self::from_diagonal(diag)
    }

    /// Returns a transformation matrix for homogeneous coordinates that
    /// translates according to `v`. Example for `Mat4` (with `v` being
    /// `[x, y, z]`):
    ///
    /// ```text
    /// ⎡ 1 0 0 a ⎤
    /// ⎢ 0 1 0 b ⎥
    /// ⎢ 0 0 1 c ⎥
    /// ⎣ 0 0 0 1 ⎦
    /// ```
    ///
    /// ```
    /// use lina::{Mat4f, vec4};
    ///
    /// let m = Mat4f::hc_translate([2.0f32, 3.0, 8.0]);
    ///
    /// assert_eq!(m.row(0), vec4(1.0, 0.0, 0.0, 2.0));
    /// assert_eq!(m.row(1), vec4(0.0, 1.0, 0.0, 3.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, 1.0, 8.0));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, 0.0, 1.0));
    /// ```
    pub fn hc_translate(v: impl Into<Vector<T, { N - 1}>>) -> Self {
        let mut m = Self::identity();
        let v = v.into();
        for i in 0..N - 1 {
            m[N - 1][i] = v[i];
        }
        m
    }

    /// Returns a transformation matrix for cartesian coordinates that scales
    /// all axis by `factor`. For the homogeneous coordinate version, see
    /// [`Matrix::hc_scale`]. Example for `Mat4` (with `f` being `factor`):
    ///
    /// ```text
    /// ⎡ f 0 0 0 ⎤
    /// ⎢ 0 f 0 0 ⎥
    /// ⎢ 0 0 f 0 ⎥
    /// ⎣ 0 0 0 f ⎦
    /// ```
    ///
    /// ```
    /// use lina::{Mat4f, vec4};
    ///
    /// let m = Mat4f::cc_scale(3.5);
    ///
    /// assert_eq!(m.row(0), vec4(3.5, 0.0, 0.0, 0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 3.5, 0.0, 0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, 3.5, 0.0));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, 0.0, 3.5));
    /// ```
    pub fn cc_scale(factor: T) -> Self {
        Self::from_diagonal([factor; N])
    }

    /// Returns a transformation matrix for cartesian coordinates that scales
    /// axis according to `factors`. For the homogeneous coordinate version,
    /// see [`Matrix::hc_nonuniform_scale`]. Equivalent to [`Matrix::from_diagonal`].
    /// Example for `Mat4` (with `factors` being `[x, y, z, w]`):
    ///
    /// ```text
    /// ⎡ x 0 0 0 ⎤
    /// ⎢ 0 y 0 0 ⎥
    /// ⎢ 0 0 z 0 ⎥
    /// ⎣ 0 0 0 w ⎦
    /// ```
    ///
    /// ```
    /// use lina::{Mat4f, vec4};
    ///
    /// let m = Mat4f::cc_nonuniform_scale([2.0f32, 3.0, 8.0, 5.0]);
    ///
    /// assert_eq!(m.row(0), vec4(2.0, 0.0, 0.0, 0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 3.0, 0.0, 0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, 8.0, 0.0));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, 0.0, 5.0));
    /// ```
    pub fn cc_nonuniform_scale(factors: impl Into<Vector<T, N>>) -> Self {
        Self::from_diagonal(factors)
    }
}

impl<T: Scalar> Mat4<T> {
    /// Returns a matrix for transformation of 3D homogeneous world space points
    /// into camera/view space.
    ///
    /// In view space, the camera is at the origin, +x points right, +y points
    /// up. This view space is right-handed, and thus, +z points outside of the
    /// screen and -z points into the screen. Please see [the part on
    /// handedness in the crate docs](../#choice-of-view-space--handedness) for
    /// more information. The returned matrix only translates and rotates,
    /// meaning that sizes and angles are unchanged compared to world space.
    ///
    /// The following camera properties have to be given:
    /// - `eye`: the position of the camera.
    /// - `direction`: the direction the camera is looking in. **Must not** be
    ///   the zero vector. Does not have to be normalized.
    /// - `up`: a usually artificial vector defining "up" in camera
    ///   space. **Must not** be the zero vector and **must not** be linearly
    ///   dependent to `direction` (i.e. they must not point into the same or
    ///   exactly opposite directions). Does not have to be normalized.
    ///
    /// To avoid float precision problems, `direction` and `up` should not have
    /// a tiny length and should not point in *almost* the same direction.
    ///
    ///
    /// ## A note on the `up` vector
    ///
    /// There are two main types of cameras in games that are distinguished by
    /// whether or not they can "fly a loop" (think airplane game with
    /// out-of-cockpit camera).
    ///
    /// In most games, this looping ability is not necessary: in those games, if
    /// you move the mouse/controller all the way up or down, the camera stops
    /// turning once you look almost straight down or up. Those are usually
    /// games with a clear "down" direction (e.g. gravity). In these cases, you
    /// usually just pass `(0, 0, 1)` (or `(0, 1, 0)` if you prefer your +y up)
    /// as `up` vector and make sure the player cannot look exactly up or down.
    /// The latter you can achieve by just having a min and max vertical angle
    /// (e.g. 1° and 179°).
    ///
    /// In games where a looping camera is required, you have to maintain and
    /// evolve the `up` vector over time. For example, if the player moves the
    /// mouse/controller you don't just adjust the look direction, but also the
    /// up vector.
    pub fn look_into(eye: Point3<T>, direction: Vec3<T>, up: Vec3<T>) -> Self
    where
        T: Float,
    {
        // This function is unlikely to be called often, so we improve developer
        // experience by having these checks and normalizations.
        assert!(!direction.is_zero(), "direction vector must not be the zero vector");
        assert!(!up.is_zero(), "up vector must not be the zero vector");
        let dir = direction.normalized();

        let right = cross(dir, up);
        assert!(!right.is_zero(), "direction and up vector must be linearly independent");

        let r = right.normalized();
        let u = cross(r, dir);
        let d = dir;
        let eye = eye.to_vec();

        // This is the inverse of this:
        //
        // ⎡ r.x  u.x  d.x  eye.x ⎤
        // ⎢ r.y  u.x  d.x  eye.x ⎥
        // ⎢ r.z  u.x  d.x  eye.x ⎥
        // ⎣   0    0    0      1 ⎦
        Self::from_rows([
            [      r.x,       r.y,       r.z, -dot(eye, r)],
            [      u.x,       u.y,       u.z, -dot(eye, u)],
            [     -d.x,      -d.y,      -d.z,  dot(eye, d)],
            [T::zero(), T::zero(), T::zero(),     T::one()],
        ])
    }

    /// Returns a matrix for perspective transformation from view space to NDC
    /// (using homogeneous coordinates).
    ///
    /// View space is assumed to be right-handed, i.e. +y pointing up and -z
    /// pointing into the screen (satisfied by [`Matrix::look_into`]). In NDC,
    /// `x/w` and `y/w` are in range -1 to 1  and denote the horizontal and
    /// vertical screen position, respectively. The +x axis points to the right,
    /// the +y axis points up. `z` represents the depth and `z/w` is in range
    /// `depth_range_out`.
    ///
    /// **Function inputs**:
    ///
    /// - `vertical_fov`: the vertical field of view of your projection. Has to
    ///   be less than half a turn (π radians or 180°)!
    ///
    /// - `aspect_ratio`: `width / height` of your target surface, e.g. screen
    ///   or application window. Has to be positive!
    ///
    /// - `depth_range_in`: the near and far plane of the projection, in world
    ///   or view space (equivalent since the view matrix does not change
    ///   distances). The far plane may be ∞ (e.g. `f32::INFINITY`), which is
    ///   handled properly by this function.
    ///
    /// - `depth_range_out`: the `z` range after the transformation. For `a..=b`,
    ///   `a` is what the near plane is mapped to and `b` is what the far plane
    ///   is mapped to. Usually, only the following values make sense:
    ///   - `0.0..=1.0` as default for WebGPU, Direct3D, Metal, Vulkan.
    ///   - `-1.0..=1.0` as default for OpenGL.
    ///   - `1.0..=0.0` for *reverse z* projection. Using this together with a
    ///     floating point depth buffer is **strongly recommended** as it vastly
    ///     improves depth precision.
    ///
    /// If your view space has left-handed or if you want your +y axis pointing
    /// down in NDC (e.g. for Vulkan), you just have to multiply the matrix
    /// returned by this function with a matrix flipping the z or y sign. (Mind
    /// the order of multiplication.)
    ///
    /// - Left-handed view space: use
    ///   `Mat4::perspective(...) * Mat4::from_diagonal([1.0, 1.0, -1.0, 1.0])`
    ///   as projection matrix.
    /// - +y pointing down in NDC: use
    ///   `Mat4::from_diagonal([1.0, -1.0, 1.0, 1.0]) * Mat4::perspective(...)`
    ///   as projection matrix.
    ///
    ///
    /// # Example
    ///
    /// (The finite near and far plane values in this example are probably not
    /// what you would want for your application.)
    ///
    /// ```
    /// use lina::{Degrees, Mat4f, vec4};
    ///
    /// let m = Mat4f::perspective(Degrees(90.0), 2.0, 0.1..=f32::INFINITY, 1.0..=0.0);
    /// assert_eq!(m.row(0), vec4(0.5, 0.0,  0.0, 0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 1.0,  0.0, 0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0,  0.0, 0.1));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0, -1.0, 0.0));
    ///
    /// let m = Mat4f::perspective(Degrees(90.0), 1.0, 0.1..=100.0, 0.0..=1.0);
    /// assert_eq!(m.row(0), vec4(1.0, 0.0,       0.0,        0.0));
    /// assert_eq!(m.row(1), vec4(0.0, 1.0,       0.0,        0.0));
    /// assert_eq!(m.row(2), vec4(0.0, 0.0, -1.001001, -0.1001001));
    /// assert_eq!(m.row(3), vec4(0.0, 0.0,      -1.0,        0.0));
    /// ```
    ///
    ///
    pub fn perspective(
        vertical_fov: impl Into<Radians<T>>,
        aspect_ratio: T,
        depth_range_in: RangeInclusive<T>,
        depth_range_out: RangeInclusive<T>,
    ) -> Self
    where
        T: Float,
    {
        let vertical_fov = vertical_fov.into();
        assert!(vertical_fov.0 < T::PI(), "`vertical_fov` has to be < π radians/180°");
        assert!(vertical_fov.0 > T::zero(), "`vertical_fov` has to be > 0");
        assert!(aspect_ratio > T::zero(), "`aspect_ratio` needs to be positive");

        let t = (vertical_fov / (T::one() + T::one())).tan();
        let sy = T::one() / t;
        let sx = sy / aspect_ratio;

        // We calculate the a and b elements of the matrix according to the
        // depth mapping (from `depth_range_in` to `depth_range_out`). This
        // article might help with understanding how these formulas came to be:
        // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
        let (near_in, far_in) = depth_range_in.into_inner();
        let (near_out, far_out) = depth_range_out.into_inner();
        let (a, b);
        if far_in.is_infinite() {
            // Curiously, this does not depend on the sign of `far_in`.
            a = -far_out;
            b = -near_in * (far_out - near_out);
        } else {
            a = (far_in * far_out - near_in * near_out) / (near_in - far_in);
            b = (near_in * far_in * (far_out - near_out)) / (near_in - far_in);
        }

        Self::from_rows([
            [       sx, T::zero(), T::zero(), T::zero()],
            [T::zero(),        sy, T::zero(), T::zero()],
            [T::zero(), T::zero(),         a,         b],
            [T::zero(), T::zero(), -T::one(), T::zero()],
        ])
    }
}


/// The inner array implements `Zeroable` and `Matrix` is just a newtype wrapper
/// around that array with `repr(transparent)`.
unsafe impl<T: Scalar + Zeroable, const C: usize, const R: usize> Zeroable for Matrix<T, C, R> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[[T; R]; C]`. And `bytemuck` itself has an impl for arrays
/// where `T: Pod`.
unsafe impl<T: Scalar + Pod, const C: usize, const R: usize> Pod for Matrix<T, C, R> {}

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
                    out[c][r] += self[s][r] * rhs[c][s];
                }
            }
        }
        out
    }
}

/// `matrix * vector` multiplication. **Important**: does not consider
/// homogeneous coordinates and thus does not divide by `w`!
impl<T: Scalar, const C: usize, const R: usize> ops::Mul<Vector<T, C>> for Matrix<T, C, R> {
    type Output = Vector<T, R>;
    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        // TODO: check generated assembly and optimize if necessary.
        array_from_index(|i| dot(self.row(i), rhs)).into()
    }
}


#[cfg(test)]
mod tests;
