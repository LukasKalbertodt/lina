use std::{array, ops, fmt, marker::PhantomData};

use bytemuck::{Zeroable, Pod};

use crate::{Float, Scalar, Matrix, Vector, Point, HcPoint, Space, WorldSpace};



/// A `C + 1`×`R + 1` homogeneous transformation matrix (`C + 1` many columns,
/// `R + 1` many rows) representing a transformation from `Src` to `Dst`.
///
/// This transformation can be interpreted in two ways:
/// - As a potentially non-linear transformation of Cartesian coordinates
///   (your normal XYZ space), or
/// - as a linear transformation of [homogeneous coordinates][hc-wiki].
///
/// In the vast majority of cases, in 3D application anyway, it is more useful
/// to think about the former interpretation, as you are usually interested in
/// your normal XYZ space. So in addition to linear transformations (which can
/// already be fully represented by [`Matrix`]), translations and projective
/// transformation can be represented by `HcMatrix`. The term "affine
/// transformation" describes the set of linear transformation and
/// translations.
///
/// As with [`Matrix`], you can use the `*` operator or [`HcMatrix::transform`]
/// to transform points with this matrix. And you can also use `*` or
/// [`HcMatrix::and_then`] to combine transformations. For more general
/// information about this, check [the `Matrix` docs][Matrix].
///
/// [hc-wiki]: https://en.wikipedia.org/wiki/Homogeneous_coordinates#Use_in_computer_graphics_and_computer_vision
///
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
/// let m = <HcMat2<_>>::from_rows([
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
#[repr(C)]
pub struct HcMatrix<
    T: Scalar,
    const C: usize,
    const R: usize,
    Src: Space = WorldSpace,
    Dst: Space = WorldSpace,
>(HcMatrixStorage<T, C, R>, PhantomData<(Src, Dst)>);


/// A homogeneous transformation matrix storing 4×4 elements.
pub type HcMat3<T, Src = WorldSpace, Dst = WorldSpace> = HcMatrix<T, 3, 3, Src, Dst>;
/// A homogeneous transformation matrix storing 3×3 elements.
pub type HcMat2<T, Src = WorldSpace, Dst = WorldSpace> = HcMatrix<T, 2, 2, Src, Dst>;

/// A homogeneous transformation matrix storing 4×4 `f32` elements.
pub type HcMat3f<Src = WorldSpace, Dst = WorldSpace> = HcMat3<f32, Src, Dst>;
/// A homogeneous transformation matrix storing 3×3 `f32` elements.
pub type HcMat2f<Src = WorldSpace, Dst = WorldSpace> = HcMat2<f32, Src, Dst>;


impl<
    T: Scalar,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> HcMatrix<T, C, R, Src, Dst> {
    fn new_impl(data: HcMatrixStorage<T, C, R>) -> Self {
        Self(data, PhantomData)
    }

    /// Returns a matrix with all elements being 0.
    ///
    /// Note: the zero homogeneous matrix is technically not a valid
    /// transformation as all resulting points have 0 coordinates and a 0
    /// weight, which is not a valid homogeneous point.
    pub fn zero() -> Self {
        let col = NPlusOneArray([T::zero(); R], T::zero());
        Self::new_impl(HcMatrixStorage(NPlusOneArray([col; C], col)))
    }

    /// Creates a homogeneous matrix from individual parts.
    ///
    /// You can't, however, treat these parts as completely independent. This
    /// function mainly exists to create homogeneous matrices of any size, as
    /// `from_rows` and similar methods only work for small dimensions.
    pub fn from_parts(
        linear: Matrix<T, C, R, Src, Dst>,
        translation: Vector<T, R, Src>,
        projection: Vector<T, C, Src>,
        q: T,
    ) -> Self {
        Self::new_impl(HcMatrixStorage(NPlusOneArray(
            array::from_fn(|c| NPlusOneArray(linear.col(c).to_array(), projection[c])),
            NPlusOneArray(translation.into(), q),
        )))
    }

    /// Returns the element in the given `row` and `col`umn.
    pub fn elem(&self, row: usize, col: usize) -> T {
        self.0.0[col][row]
    }

    /// Sets the element in the given `row` and `col`umn to `v`
    pub fn set_elem(&mut self, row: usize, col: usize, v: T) {
        self.0.0[col][row] = v;
    }

    /// Returns the row with the given index.
    pub fn row(&self, row: usize) -> HcRow<'_, T, C, R> {
        HcRow { matrix: &self.0, index: row }
    }

    /// Returns the column with the given index.
    pub fn col(&self, col: usize) -> HcCol<'_, T, C, R> {
        HcCol { matrix: &self.0, index: col }
    }

    /// Returns the linear part of this matrix, i.e. the last row and column
    /// removed.
    pub fn linear_part(&self) -> Matrix<T, C, R, Src, Dst> {
        Matrix::from_cols(self.0.0.0.map(|c| c.0))
    }

    /// Reinterprets this matrix to be a transformation into the space `New`.
    pub fn with_target_space<New: Space>(self) -> HcMatrix<T, C, R, Src, New> {
        HcMatrix::new_impl(self.0)
    }

    /// Reinterprets this matrix to be a transformation from the space `New`.
    pub fn with_source_space<New: Space>(self) -> HcMatrix<T, C, R, New, Dst> {
        HcMatrix::new_impl(self.0)
    }

    /// Reinterprets this matrix to be a transformation from the space `NewSrc`
    /// into the space `NewDst`.
    pub fn with_spaces<NewSrc: Space, NewDst: Space>(self) -> HcMatrix<T, C, R, NewSrc, NewDst> {
        HcMatrix::new_impl(self.0)
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
    /// let translate = <HcMat2<_>>::from_rows([
    ///     [1, 0, 3],
    ///     [0, 1, 7],
    ///     [0, 0, 1],
    /// ]);
    ///
    /// // Project onto the y = 1 line.
    /// let project = <HcMat2<_>>::from_rows([
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
    pub fn and_then<const R2: usize, Dst2: Space>(
        self,
        second: HcMatrix<T, R, R2, Dst, Dst2>,
    ) -> HcMatrix<T, C, R2, Src, Dst2> {
        second * self
    }

    pub fn transposed<NewSrc: Space, NewDst: Space>(&self) -> HcMatrix<T, R, C, NewSrc, NewDst> {
        let mut out = HcMatrix::zero();
        for c in 0..=C {
            for r in 0..=R {
                out.set_elem(c, r, self.elem(r, c));
            }
        }
        out
    }

    /// Returns an iterator over all entries of this matrix, in column-major
    /// order.
    ///
    /// ```
    /// let m = <lina::HcMatrix<_, 1, 2>>::from_rows([
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
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        (0..(C + 1) * (R + 1)).map(|idx| self.elem(idx % (R + 1), idx / (R + 1)))
    }

    /// Applies the given function to each element and returns the resulting new
    /// matrix.
    pub fn map<U: Scalar, F: FnMut(T) -> U>(&self, mut f: F) -> HcMatrix<U, C, R, Src, Dst> {
        let mut out = HcMatrix::zero();
        for c in 0..=C {
            for r in 0..=R {
                out.set_elem(r, c, f(self.elem(r, c)));
            }
        }
        out
    }

    /// Pairs up the same elements from `self` and `other`, applies the given
    /// function to each and returns the resulting matrix. Useful for
    /// element-wise operations.
    ///
    /// ```
    /// use lina::{HcMat2f, vec3};
    ///
    /// let a = <HcMat2f>::from_rows([
    ///     [1.0, 2.0, 3.0],
    ///     [4.0, 5.0, 6.0],
    ///     [7.0, 8.0, 9.0],
    /// ]);
    /// let b = <HcMat2f>::identity();
    /// let c = a.zip_map(&b, |elem_a, elem_b| elem_a * elem_b);   // element-wise multiplication
    ///
    /// assert_eq!(c, HcMat2f::from_rows([
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 5.0, 0.0],
    ///     [0.0, 0.0, 9.0],
    /// ]));
    /// ```
    pub fn zip_map<U, O, F>(
        &self,
        other: &HcMatrix<U, C, R, Src, Dst>,
        mut f: F,
    ) -> HcMatrix<O, C, R, Src, Dst>
    where
        U: Scalar,
        O: Scalar,
        F: FnMut(T, U) -> O,
    {
        let mut out = HcMatrix::zero();
        for c in 0..=C {
            for r in 0..=R {
                out.set_elem(r, c, f(self.elem(r, c), other.elem(r, c)));
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

macro_rules! impl_det_inv {
    ($d:expr, $dpo:expr) => {
        impl<T: Float, Src: Space, Dst: Space> HcMatrix<T, $d, $d, Src, Dst> {
            #[doc = include_str!("determinant_docs.md")]
            pub fn determinant(&self) -> T {
                self.to_mat().determinant()
            }

            #[doc = include_str!("inverted_docs.md")]
            pub fn inverted(&self) -> Option<HcMatrix<T, $d, $d, Dst, Src>> {
                let inv = self.to_mat().inverted()?;
                Some(<HcMatrix<T, $d, $d, Dst, Src>>::from_cols(inv.0))
            }

            fn to_mat(&self) -> Matrix<T, $dpo, $dpo, Src, Dst> {
                Matrix::from_cols(array::from_fn(|c| self.col(c).to_array()))
            }
        }
    };
}

impl_det_inv!(1, 2);
impl_det_inv!(2, 3);
impl_det_inv!(3, 4);


macro_rules! inc {
    (1) => { 2 };
    (2) => { 3 };
    (3) => { 4 };
}

macro_rules! gen_inc_methods {
    ($( ($c:tt, $r:tt) ),+ $(,)?) => {
        $(
            gen_inc_methods!(@imp [], $c, inc!($c), $r, inc!($r));
        )+
    };
    (@imp [$($const_params:tt)*],$c:expr, $cpo:expr, $r:expr, $rpo:expr) => {
        impl<T: Scalar $($const_params)*, Src: Space, Dst: Space> HcMatrix<T, $c, $r, Src, Dst> {
            pub fn from_rows<V: Into<[T; $cpo]>>(rows: [V; $rpo]) -> Self {
                let mut out = Self::zero();
                for (r, row) in rows.into_iter().enumerate() {
                    out.set_row(r, row.into());
                }
                out
            }

            pub fn from_cols<V: Into<[T; $rpo]>>(cols: [V; $cpo]) -> Self {
                Self::new_impl(cols.map(Into::into).into())
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
        $(
            gen_quadratic_inc_methods!(@imp [], $n, inc!($n));
        )+
    };
    (@imp [$($const_params:tt)*], $n:expr, $npo:expr) => {
        impl<T: Scalar $($const_params)*, Src: Space, Dst: Space> HcMatrix<T, $n, $n, Src, Dst> {
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

impl<T: Scalar, const N: usize, Src: Space, Dst: Space> HcMatrix<T, N, N, Src, Dst> {
    /// Returns the identity matrix (1s along diagonal, everything else 0s).
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

    /// Returns the *trace* of the matrix, i.e. the sum of all elements on the
    /// diagonal.
    pub fn trace(&self) -> T {
        let (l, q) = self.diagonal_parts();
        q + l.into_iter().fold(q, |acc, e| acc + e)
    }
}


// =============================================================================================
// ===== Non-mathematical trait impls
// =============================================================================================

// HcMatrix is just a wrapper around `HcMatrixStorage`
unsafe impl<
    T: Scalar + Zeroable,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> Zeroable for HcMatrix<T, C, R, Src, Dst> {}
unsafe impl<
    T: Scalar + Pod,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> Pod for HcMatrix<T, C, R, Src, Dst> {}

impl<
    T: Scalar,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> fmt::Debug for HcMatrix<T, C, R, Src, Dst> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HcMatrix ")?;
        super::debug_matrix_impl(f, C + 1, R + 1, |r, c| self.elem(r, c))
    }
}

// =============================================================================================
// ===== Mathematical trait impls
// =============================================================================================

super::shared_trait_impls!(HcMatrix);
super::impl_scalar_mul!(HcMatrix => f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);


// =============================================================================================
// ===== Matrix * matrix multiplication (composition)
// =============================================================================================

/// `matrix * matrix` multiplication. You can also use [`Matrix::and_then`].
impl<
    T: Scalar,
    const C: usize,
    const R: usize,
    const S: usize,
    Src: Space,
    Mid: Space,
    Dst: Space,
> ops::Mul<HcMatrix<T, C, S, Src, Mid>> for HcMatrix<T, S, R, Mid, Dst> {
    type Output = HcMatrix<T, C, R, Src, Dst>;
    fn mul(self, rhs: HcMatrix<T, C, S, Src, Mid>) -> Self::Output {
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

/// See [`HcMatrix::transform`].
impl<
    T: Scalar,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> ops::Mul<Point<T, C, Src>> for &HcMatrix<T, C, R, Src, Dst> {
    type Output = Point<T, R, Dst>;
    fn mul(self, rhs: Point<T, C, Src>) -> Self::Output {
        (self * rhs.to_hc_point()).to_point()
    }
}

/// See [`HcMatrix::transform`].
impl<
    T: Scalar,
    const C: usize,
    const R: usize,
    Src: Space,
    Dst: Space,
> ops::Mul<HcPoint<T, C, Src>> for &HcMatrix<T, C, R, Src, Dst> {
    type Output = HcPoint<T, R, Dst>;
    fn mul(self, rhs: HcPoint<T, C, Src>) -> Self::Output {
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
    matrix: &'a HcMatrixStorage<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> HcRow<'a, T, C, R> {
    /// Indexes into this row with the given column index, returning the element.
    pub fn col(self, col: usize) -> T {
        self.matrix.0[col][self.index]
    }
}

/// Proxy type representing one column of a homogeneous matrix.
#[derive(Clone, Copy)]
pub struct HcCol<'a, T: Scalar, const C: usize, const R: usize> {
    matrix: &'a HcMatrixStorage<T, C, R>,
    index: usize,
}

impl<'a, T: Scalar, const C: usize, const R: usize> HcCol<'a, T, C, R> {
    /// Indexes into this column with the given row index, returning the element.
    pub fn row(self, row: usize) -> T {
        self.matrix.0[self.index][row]
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
        $(
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
            pub fn to_vec<S: Space>(self) -> Vector<T, {$cpo}, S> {
                self.into()
            }

            /// Returns this row as point.
            pub fn to_point<S: Space>(self) -> Point<T, {$cpo}, S> {
                self.into()
            }
        }
        impl<'a, T: Scalar $($const_params)*> From<HcRow<'a, T, $c, $r>> for [T; $cpo] {
            fn from(src: HcRow<'a, T, $c, $r>) -> Self {
                array::from_fn(|i| src.matrix.0[i][src.index])
            }
        }
        impl<'a, T: Scalar $($const_params)*, S: Space> From<HcRow<'a, T, $c, $r>>
            for Vector<T, {$cpo}, S>
        {
            fn from(src: HcRow<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }
        impl<'a, T: Scalar $($const_params)*, S: Space> From<HcRow<'a, T, $c, $r>>
            for Point<T, {$cpo}, S>
        {
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
            pub fn to_vec<S: Space>(self) -> Vector<T, {$rpo}, S> {
                self.into()
            }

            /// Returns this column as point.
            pub fn to_point<S: Space>(self) -> Point<T, {$rpo}, S> {
                self.into()
            }
        }

        impl<'a, T: Scalar $($const_params)*> From<HcCol<'a, T, $c, $r>> for [T; $rpo] {
            fn from(src: HcCol<'a, T, $c, $r>) -> Self {
                array::from_fn(|i| src.matrix.0[src.index][i])
            }
        }

        impl<'a, T: Scalar $($const_params)*, S: Space> From<HcCol<'a, T, $c, $r>>
            for Vector<T, {$rpo}, S>
        {
            fn from(src: HcCol<'a, T, $c, $r>) -> Self {
                src.to_array().into()
            }
        }
        impl<'a, T: Scalar $($const_params)*, S: Space> From<HcCol<'a, T, $c, $r>>
            for Point<T, {$rpo}, S>
        {
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
