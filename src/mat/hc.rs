use std::{array, ops, fmt};

use bytemuck::{Zeroable, Pod};

use crate::{Scalar, Matrix, Vector, Point, HcPoint};



/// TODO
///
/// ## `fmt::Debug` output
///
/// Setting the alternative flag `#` for debug output is recommended to have a
/// properly formatted matrix. In order to avoid ambiguity when using
/// single-line mode, a `row<i>` is prefixed to each row.
///
/// ```
/// use lina::HcMat2;
///
/// let m = HcMat2::from_rows([
///     [1, 2, 3],
///     [4, 5, 6],
///     [7, 8, 9],
/// ]);
///
/// // Formatting without `#` alternate flag (one line)
/// assert_eq!(
///     format!("{m:?}"),
///     "HcMatrix [row0 [1, 2, 3], row1 [4, 5, 6], row2 [7, 8, 9]]",
/// );
///
/// // Formatting with `#` alternate flag (multi line)
/// assert_eq!(format!("{m:#?}"), concat!(
///     "HcMatrix [\n",
///     "    [1, 2, 3],\n",
///     "    [4, 5, 6],\n",
///     "    [7, 8, 9],\n",
///     "]",
/// ));
/// ```
///
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct HcMatrix<T, const C: usize, const R: usize>(HcMatrixStorage<T, C, R>);


/// A 3×3 homogeneous transformation matrix.
pub type HcMat3<T> = HcMatrix<T, 3, 3>;
/// A 2×2 homogeneous transformation matrix.
pub type HcMat2<T> = HcMatrix<T, 2, 2>;

/// A 3×3 homogeneous transformation matrix with `f32` elements.
pub type HcMat3f = HcMat3<f32>;
/// A 3×3 homogeneous transformation matrix with `f63` elements.
pub type HcMat3d = HcMat3<f64>;

/// A 2×2 homogeneous transformation matrix with `f32` elements.
pub type HcMat2f = HcMat2<f32>;
/// A 2×2 homogeneous transformation matrix with `f62` elements.
pub type HcMat2d = HcMat2<f64>;


impl<T: Scalar, const C: usize, const R: usize> HcMatrix<T, C, R> {
    pub fn zero() -> Self {
        let col = NPlusOneArray([T::zero(); R], T::zero());
        Self(HcMatrixStorage(NPlusOneArray([col; C], col)))
    }

    pub fn from_parts(
        linear: Matrix<T, C, R>,
        translation: Vector<T, R>,
        projection: Vector<T, C>,
        q: T,
    ) -> Self {
        Self(HcMatrixStorage(NPlusOneArray(
            array::from_fn(|c| NPlusOneArray(linear.col(c).to_array(), projection[c])),
            NPlusOneArray(translation.into(), q),
        )))
    }

    pub fn elem(&self, row: usize, col: usize) -> T {
        self.0.0[col][row]
    }

    pub fn set_elem(&mut self, row: usize, col: usize, v: T) {
        self.0.0[col][row] = v;
    }

    pub fn row(&self, row: usize) -> HcRow<'_, T, C, R> {
        HcRow { matrix: self, index: row }
    }

    pub fn col(&self, col: usize) -> HcCol<'_, T, C, R> {
        HcCol { matrix: self, index: col }
    }

    pub fn linear_part(&self) -> Matrix<T, C, R> {
        Matrix::from_cols(self.0.0.0.map(|c| c.0))
    }

    pub fn translation_part(&self) -> Vector<T, R> {
        self.0.0.1.0.into()
    }

    pub fn projection_part(&self) -> Vector<T, C> {
        array::from_fn(|i| self.elem(R, i)).into()
    }

    pub fn q(&self) -> T {
        self.elem(R, C)
    }

    /// Transforms the given point with this matrix.
    ///
    /// This function accepts `C`-dimensional [`Point`]s and [`HcPoint`]s. For
    /// `HcPoint`, this is just a simple matrix-vector-multiplication. For
    /// `Point`, the input is first converted into a `HcPoint` (1-extended),
    /// then transformed, and finally converted back to `Point` via
    /// [`HcPoint::to_point`] (performing a perspective divide).
    ///
    /// You cannot transform [`Vector`]s with this, as that generally doesn't
    /// make sense. A vector is a displacement in space. Translations do not
    /// affect it and any projective transformation does not make any sense for
    /// vectors. If you have an affine transformation (a `HcMatrix` with last
    /// row `0, 0, .., 0, 1`), you call `m.linear_part() * vec` to only apply
    /// the linear part of the transformation to the vector.
    ///
    /// Instead of using this function, you can also use the `*` operator
    /// overload, if you prefer. It does exactly the same.
    ///
    /// # Example
    ///
    /// ```
    /// use lina::{HcMat3f, HcMat2f, HcPoint, point2, point3};
    ///
    /// // Scale and translate
    /// let m = HcMat2f::from_rows([
    ///     [2.5, 0.0, 0.3],
    ///     [0.0, 2.5, 0.7],
    ///     [0.0, 0.0, 1.0],
    /// ]);
    /// assert_eq!(
    ///     m.transform(point2(2.0, 4.0)),
    ///     point2(5.3, 10.7),
    /// );
    /// assert_eq!(
    ///     m.transform(HcPoint::new([2.0, 4.0], 1.0)),
    ///     HcPoint::new([5.3, 10.7], 1.0),
    /// );
    ///
    ///
    /// // Projection onto the z = 1 plane.
    /// let m = HcMat3f::from_rows([
    ///     [1.0, 0.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0],
    /// ]);
    /// assert_eq!(
    ///     m.transform(point3(4.0, 3.0, 2.0)),
    ///     point3(2.0, 1.5, 1.0),
    /// );
    /// assert_eq!(
    ///     m.transform(HcPoint::new([4.0, 3.0, 2.0], 1.0)),
    ///     HcPoint::new([4.0, 3.0, 2.0], 2.0),
    /// );
    /// ```
    pub fn transform<'a, P>(&'a self, p: P) -> <&'a Self as ops::Mul<P>>::Output
    where
        &'a Self: ops::Mul<P>,
    {
        self * p
    }

    /// Combines the transformations of two matrices into a single
    /// transformation matrix. First the transformation of `self` is applied,
    /// then the one of `second`. In the language of math, this is just matrix
    /// multiplication: `second * self`. You can also use the overloaded `*`
    /// operator instead, but this method exists for those who find the matrix
    /// multiplication order unintuitive and this method call easier to read.
    ///
    /// ```
    /// use lina::HcMat2;
    ///
    /// // Translate by (3, 7)
    /// let translate = HcMat2::from_rows([
    ///     [1, 0, 3],
    ///     [0, 1, 7],
    ///     [0, 0, 1],
    /// ]);
    ///
    /// // Project onto the y = 1 line.
    /// let project = HcMat2::from_rows([
    ///     [1, 0, 0],
    ///     [0, 1, 0],
    ///     [0, 1, 0],
    /// ]);
    ///
    /// assert_eq!(translate.and_then(project), HcMat2::from_rows([
    ///     [1, 0, 3],
    ///     [0, 1, 7],
    ///     [0, 1, 7],
    /// ]));
    /// assert_eq!(project.and_then(translate), HcMat2::from_rows([
    ///     [1, 3, 0],
    ///     [0, 8, 0],
    ///     [0, 1, 0],
    /// ]));
    /// ```
    pub fn and_then<const R2: usize>(self, second: HcMatrix<T, R, R2>) -> HcMatrix<T, C, R2> {
        second * self
    }

    pub fn transposed(&self) -> HcMatrix<T, R, C> {
        let mut out = HcMatrix::zero();
        for c in 0..=C {
            for r in 0..=R {
                out.set_elem(c, r, self.elem(r, c));
            }
        }
        out
    }

    pub fn map<U: Scalar, F: FnMut(T) -> U>(self, mut f: F) -> HcMatrix<U, C, R> {
        let mut out = HcMatrix::zero();
        for c in 0..=C {
            for r in 0..=R {
                out.set_elem(r, c, f(self.elem(r, c)));
            }
        }
        out
    }

    /// Returns a byte slice of this matrix, representing the raw column-major
    /// data. Useful to pass to graphics APIs.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[cfg(not(feature = "nightly"))]
macro_rules! inc {
    (1) => { 2 };
    (2) => { 3 };
    (3) => { 4 };
}

macro_rules! gen_inc_methods {
    ($( ($c:tt, $r:tt) ),+ $(,)?) => {
        #[cfg(feature = "nightly")]
        gen_inc_methods!(@imp [, const C: usize, const R: usize], C, C + 1, R, R + 1);

        $(
            #[cfg(not(feature = "nightly"))]
            gen_inc_methods!(@imp [], $c, inc!($c), $r, inc!($r));
        )+
    };
    (@imp [$($const_params:tt)*],$c:expr, $cpo:expr, $r:expr, $rpo:expr) => {
        impl<T: Scalar $($const_params)*> HcMatrix<T, $c, $r> {
            pub fn from_rows<V: Into<[T; $cpo]>>(rows: [V; $rpo]) -> Self {
                let mut out = Self::zero();
                for (r, row) in rows.into_iter().enumerate() {
                    out.set_row(r, row.into());
                }
                out
            }

            pub fn from_cols<V: Into<[T; $rpo]>>(cols: [V; $cpo]) -> Self {
                Self(cols.map(Into::into).into())
            }

            pub fn set_row(&mut self, index: usize, row: [T; $cpo]) {
                for c in 0..=$c {
                    self.set_elem(index, c, row[c]);
                }
            }

            pub fn set_col(&mut self, index: usize, col: [T; $rpo]) {
                for r in 0..=$r {
                    self.set_elem(r, index, col[r]);
                }
            }
        }

        impl<T: Scalar $($const_params)*> AsRef<[[T; $rpo]; $cpo]> for HcMatrixStorage<T, $c, $r> {
            fn as_ref(&self) -> &[[T; $rpo]; $cpo] {
                bytemuck::cast_ref(self)
            }
        }

        impl<T: Scalar $($const_params)*> AsMut<[[T; $rpo]; $cpo]> for HcMatrixStorage<T, $c, $r> {
            fn as_mut(&mut self) -> &mut [[T; $rpo]; $cpo] {
                bytemuck::cast_mut(self)
            }
        }

        impl<T: Scalar $($const_params)*> From<[[T; $rpo]; $cpo]> for HcMatrixStorage<T, $c, $r> {
            fn from(value: [[T; $rpo]; $cpo]) -> Self {
                bytemuck::cast(value)
            }
        }
    }
}

gen_inc_methods!(
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
);

macro_rules! gen_quadratic_inc_methods {
    ($( $n:tt ),+) => {
        #[cfg(feature = "nightly")]
        gen_quadratic_inc_methods!(@imp [, const N: usize], N, N + 1);

        $(
            #[cfg(not(feature = "nightly"))]
            gen_quadratic_inc_methods!(@imp [], $n, inc!($n));
        )+
    };
    (@imp [$($const_params:tt)*], $n:expr, $npo:expr) => {
        impl<T: Scalar $($const_params)*> HcMatrix<T, $n, $n> {
            pub fn from_diagonal(diagonal: [T; $npo]) -> Self {
                let mut out = Self::zero();
                out.set_diagonal(diagonal);
                out
            }

            pub fn set_diagonal(&mut self, diagonal: [T; $npo]) {
                for i in 0..=$n {
                    self.set_elem(i, i, diagonal[i]);
                }
            }

            pub fn diagonal(&self) -> [T; $npo] {
                array::from_fn(|i| self.elem(i, i))
            }
        }
    }
}

gen_quadratic_inc_methods!(1, 2, 3);

impl<T: Scalar, const N: usize> HcMatrix<T, N, N> {
    pub fn identity() -> Self {
        Self::from_diagonal_parts([T::one(); N], T::one())
    }

    pub fn from_diagonal_parts(linear: [T; N], q: T) -> Self {
        let mut out = Self::zero();
        out.set_diagonal_parts(linear, q);
        out
    }

    pub fn diagonal_parts(&self) -> ([T; N], T) {
        (array::from_fn(|i| self.elem(i, i)), self.elem(N, N))
    }

    pub fn set_diagonal_parts(&mut self, linear: [T; N], q: T) {
        for i in 0..N {
            self.set_elem(i, i, linear[i]);
        }
        self.set_elem(N, N, q);
    }


    pub fn transpose(&mut self) {
        *self = self.transposed();
    }

    pub fn is_symmetric(&self) -> bool {
        for c in 1..=N {
            for r in 0..c {
                if self.elem(c, r) != self.elem(r, c) {
                    return false;
                }
            }
        }

        true
    }

    pub fn trace(&self) -> T {
        let (l, q) = self.diagonal_parts();
        q + l.into_iter().fold(q, |acc, e| acc + e)
    }
}


// =============================================================================================
// ===== Non-mathematical trait impls
// =============================================================================================

// HcMatrix is just a wrapper around `HcMatrixStorage`
unsafe impl<T: Scalar + Zeroable, const C: usize, const R: usize> Zeroable for HcMatrix<T, C, R> {}
unsafe impl<T: Scalar + Pod, const C: usize, const R: usize> Pod for HcMatrix<T, C, R> {}

impl<T: Scalar, const C: usize, const R: usize> fmt::Debug for HcMatrix<T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HcMatrix ")?;
        super::debug_matrix_impl(f, C + 1, R + 1, |r, c| self.elem(r, c))
    }
}

// =============================================================================================
// ===== Mathematical trait impls
// =============================================================================================


// =============================================================================================
// ===== Matrix * matrix multiplication (composition)
// =============================================================================================

/// `matrix * matrix` multiplication. You can also use [`Matrix::and_then`].
impl<T: Scalar, const C: usize, const R: usize, const S: usize> ops::Mul<HcMatrix<T, C, S>>
    for HcMatrix<T, S, R>
{
    type Output = HcMatrix<T, C, R>;
    fn mul(self, rhs: HcMatrix<T, C, S>) -> Self::Output {
        // This is the straight-forward n³ algorithm. Using more sophisticated
        // algorithms with sub cubic runtime is not worth it for small
        // matrices. However, this can certainly be micro-optimized. In
        // particular, using SSE seems like a good idea, but "requires" all
        // columns to be properly aligned in memory.
        //
        // TODO: try to optimize
        let mut out = Self::Output::zero();
        for c in 0..=C {
            for r in 0..=R {
                let mut e = T::zero();
                for s in 0..=S {
                    e += self.elem(r, s) * rhs.elem(s, c);
                }
                out.set_elem(r, c, e);
            }
        }
        out
    }
}

// =============================================================================================
// ===== Matrix * vector multiplication (transformations)
// =============================================================================================

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<Point<T, C>> for &HcMatrix<T, C, R> {
    type Output = Point<T, R>;
    fn mul(self, rhs: Point<T, C>) -> Self::Output {
        (self * rhs.to_hc_point()).to_point()
    }
}

impl<T: Scalar, const C: usize, const R: usize> ops::Mul<HcPoint<T, C>> for &HcMatrix<T, C, R> {
    type Output = HcPoint<T, R>;
    fn mul(self, rhs: HcPoint<T, C>) -> Self::Output {
        let dot = |row| (0..=C)
            .map(|col| self.elem(row, col) * rhs[col])
            .fold(T::zero(), |acc, e| acc + e);

        HcPoint::new(array::from_fn(dot), dot(C))
    }
}


// =============================================================================================
// ===== Storage utilities
// =============================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
struct HcMatrixStorage<T, const C: usize, const R: usize>(NPlusOneArray<NPlusOneArray<T, R>, C>);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
struct NPlusOneArray<T, const N: usize>([T; N], T);

// These are all fine: The only stored things are `T` and the bounds on `T`
// already make sure most of the requirements for these traits are met.
// Further, due to `repr(C)` and the struct layout, there are no padding
// bytes.
unsafe impl<T: Scalar + Zeroable, const C: usize, const R: usize> Zeroable
    for HcMatrixStorage<T, C, R> {}
unsafe impl<T: Scalar + Pod, const C: usize, const R: usize> Pod for HcMatrixStorage<T, C, R> {}
unsafe impl<T: Scalar + Zeroable, const N: usize> Zeroable for NPlusOneArray<T, N> {}
unsafe impl<T: Scalar + Pod, const N: usize> Pod for NPlusOneArray<T, N> {}

impl<T, const N: usize> ops::Index<usize> for NPlusOneArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match () {
            () if index < N => &self.0[index],
            () if index == N => &self.1,
            _ => panic!("index ({index}) out of bounds ({})", N + 1),
        }
    }
}

impl<T, const N: usize> ops::IndexMut<usize> for NPlusOneArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match () {
            () if index < N => &mut self.0[index],
            () if index == N => &mut self.1,
            _ => panic!("index ({index}) out of bounds ({})", N + 1),
        }
    }
}

impl<T: Pod + Scalar, const N: usize> AsRef<[T]> for NPlusOneArray<T, N>
where
    [T]: Pod,
{
    fn as_ref(&self) -> &[T] {
        bytemuck::cast_ref(self)
    }
}


// =============================================================================================
// ===== `Row` and `Col` proxies
// =============================================================================================

/// Proxy type representing one row of a homogeneous matrix.
#[derive(Clone, Copy)]
pub struct HcRow<'a, T: Scalar, const C: usize, const R: usize> {
    matrix: &'a HcMatrix<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> HcRow<'a, T, C, R> {
    /// Indexes into this row with the given column index, returning the element.
    pub fn col(self, col: usize) -> T {
        self.matrix.0.0[col][self.index]
    }
}

/// Proxy type representing one column of a homogeneous matrix.
#[derive(Clone, Copy)]
pub struct HcCol<'a, T: Scalar, const C: usize, const R: usize> {
    matrix: &'a HcMatrix<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> HcCol<'a, T, C, R> {
    /// Indexes into this column with the given row index, returning the element.
    pub fn row(self, row: usize) -> T {
        self.matrix.0.0[self.index][row]
    }
}


impl<'a, T: Scalar, const C: usize, const R: usize> fmt::Debug for HcRow<'a, T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::debug_list_one_line((0..=C).map(|c| self.col(c)), f)
    }
}

impl<'a, T: Scalar, const C: usize, const R: usize> fmt::Debug for HcCol<'a, T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::util::debug_list_one_line((0..=R).map(|r| self.row(r)), f)
    }
}


macro_rules! gen_col_row_impls {
    ($( ($c:tt, $r:tt) ),+ $(,)?) => {
        #[cfg(feature = "nightly")]
        gen_col_row_impls!(@imp [, const C: usize, const R: usize], C, C + 1, R, R + 1);

        $(
            #[cfg(not(feature = "nightly"))]
            gen_col_row_impls!(@imp [], $c, inc!($c), $r, inc!($r));
        )+
    };
    (@imp [$($const_params:tt)*], $c:expr, $cpo:expr, $r:expr, $rpo:expr) => {
        // Row
        impl<'a, T: Scalar $($const_params)*> HcRow<'a, T, $c, $r> {
            /// Returns this row as array.
            pub fn to_array(self) -> [T; $cpo] {
                self.into()
            }

            /// Returns this row as vector.
            pub fn to_vec(self) -> Vector<T, {$cpo}> {
                self.into()
            }

            /// Returns this row as point.
            pub fn to_point(self) -> Point<T, {$cpo}> {
                self.into()
            }
        }
        impl<'a, T: Scalar $($const_params)*> From<HcRow<'a, T, $c, $r>> for [T; $cpo] {
            fn from(src: HcRow<'a, T, $c, $r>) -> Self {
                array::from_fn(|i| src.matrix.0.0[i][src.index])
            }
        }
        impl<'a, T: Scalar $($const_params)*> From<HcRow<'a, T, $c, $r>> for Vector<T, {$cpo}> {
            fn from(src: HcRow<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }
        impl<'a, T: Scalar $($const_params)*> From<HcRow<'a, T, $c, $r>> for Point<T, {$cpo}> {
            fn from(src: HcRow<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }

        // Col
        impl<'a, T: Scalar $($const_params)*> HcCol<'a, T, $c, $r> {
            /// Returns this column as array.
            pub fn to_array(self) -> [T; $rpo] {
                self.into()
            }

            /// Returns this column as vector.
            pub fn to_vec(self) -> Vector<T, {$rpo}> {
                self.into()
            }

            /// Returns this column as point.
            pub fn to_point(self) -> Point<T, {$rpo}> {
                self.into()
            }
        }

        impl<'a, T: Scalar $($const_params)*> From<HcCol<'a, T, $c, $r>> for [T; $rpo] {
            fn from(src: HcCol<'a, T, $c, $r>) -> Self {
                array::from_fn(|i| src.matrix.0.0[src.index][i])
            }
        }

        impl<'a, T: Scalar $($const_params)*> From<HcCol<'a, T, $c, $r>> for Vector<T, {$rpo}> {
            fn from(src: HcCol<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }
        impl<'a, T: Scalar $($const_params)*> From<HcCol<'a, T, $c, $r>> for Point<T, {$rpo}> {
            fn from(src: HcCol<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }
    }
}

gen_col_row_impls!(
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3),
);
