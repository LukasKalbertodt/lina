use std::{array, ops};

use bytemuck::{Zeroable, Pod};

use crate::{Scalar, Matrix, Vector};




#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct HcMatrix<T, const C: usize, const R: usize>(HcMatrixStorage<T, C, R>);


/// A 4×4 matrix.
pub type HcMat4<T> = HcMatrix<T, 4, 4>;
/// A 3×3 matrix.
pub type HcMat3<T> = HcMatrix<T, 3, 3>;
/// A 2×2 matrix.
pub type HcMat2<T> = HcMatrix<T, 2, 2>;

/// A 4×4 matrix with `f32` elements.
pub type HcMat4f = HcMat4<f32>;
/// A 4×4 matrix with `f64` elements.
pub type HcMat4d = HcMat4<f64>;

/// A 3×3 matrix with `f32` elements.
pub type HcMat3f = HcMat3<f32>;
/// A 3×3 matrix with `f63` elements.
pub type HcMat3d = HcMat3<f64>;

/// A 2×2 matrix with `f32` elements.
pub type HcMat2f = HcMat2<f32>;
/// A 2×2 matrix with `f62` elements.
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

    #[cfg(feature = "nightly")]
    pub fn from_rows<V: Into<[T; R + 1]>>(cols: [V; C + 1]) -> Self {
        Self(cols.map(Into::into).into())
    }

    #[cfg(feature = "nightly")]
    pub fn from_cols<V: Into<[T; R + 1]>>(cols: [V; C + 1]) -> Self {
        Self(cols.map(Into::into).into())
    }

    pub fn elem(&self, row: usize, col: usize) -> T {
        self.0.0[col][row]
    }

    pub fn set_elem(&mut self, row: usize, col: usize, v: T) {
        self.0.0[col][row] = v;
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

    // col/row
    // set_col/row
    // iter

    // and_then
    // zip_map
    //
}

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

    #[cfg(feature = "nightly")]
    pub fn from_diagonal(diagonal: [T; N + 1]) -> Self {
        let mut out = Self::zero();
        out.set_diagonal(diagonal);
        out
    }

    #[cfg(feature = "nightly")]
    pub fn set_diagonal(&mut self, diagonal: [T; N + 1]) {
        for i in 0..=N {
            self.set_elem(i, i, diagonal[i]);
        }
    }

    #[cfg(feature = "nightly")]
    pub fn diagonal(&self) -> [T; N + 1] {
        array::from_fn(|i| self.elem(i, i))
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

// TODO: invert & det


// These are all fine: The only stored things are `T` and the bounds on `T`
// already make sure most of the requirements for these traits are met.
// Further, due to `repr(C)` and the struct layout, there are no padding
// bytes.
unsafe impl<T, const C: usize, const R: usize> Zeroable for HcMatrix<T, C, R>
where
    T: Scalar + Zeroable,
{}
unsafe impl<T, const C: usize, const R: usize> Pod for HcMatrix<T, C, R>
where
    T: Scalar + Pod,
{}
unsafe impl<T, const C: usize, const R: usize> Zeroable for HcMatrixStorage<T, C, R>
where
    T: Scalar + Zeroable,
{}
unsafe impl<T, const C: usize, const R: usize> Pod for HcMatrixStorage<T, C, R>
where
    T: Scalar + Pod,
{}


// =============================================================================================
// ===== Storage utilities
// =============================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct HcMatrixStorage<T, const C: usize, const R: usize>(NPlusOneArray<NPlusOneArray<T, R>, C>);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
struct NPlusOneArray<T, const N: usize>([T; N], T);

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

#[cfg(feature = "nightly")]
impl<T: Scalar, const C: usize, const R: usize> AsRef<[[T; R + 1]; C + 1]> for HcMatrixStorage<T, C, R> {
    fn as_ref(&self) -> &[[T; R + 1]; C + 1] {
        bytemuck::cast_ref(self)
    }
}

#[cfg(feature = "nightly")]
impl<T: Scalar, const C: usize, const R: usize> AsMut<[[T; R + 1]; C + 1]> for HcMatrixStorage<T, C, R> {
    fn as_mut(&mut self) -> &mut [[T; R + 1]; C + 1] {
        bytemuck::cast_mut(self)
    }
}

#[cfg(feature = "nightly")]
impl<T: Scalar, const C: usize, const R: usize> From<[[T; R + 1]; C + 1]> for HcMatrixStorage<T, C, R> {
    fn from(value: [[T; R + 1]; C + 1]) -> Self {
        bytemuck::cast(value)
    }
}
