use std::{array, ops, fmt};

use bytemuck::{Zeroable, Pod};

use crate::{Scalar, Matrix, Vector};
use super::debug_matrix_impl;




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
    // iter

    // and_then
    // zip_map
    //
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

// TODO: invert & det

// =============================================================================================
// ===== Non-mathematical trait impls
// =============================================================================================

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
impl<T: Scalar, const C: usize, const R: usize> fmt::Debug for HcMatrix<T, C, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HcMatrix ")?;
        debug_matrix_impl(f, C, R, |r, c| self.elem(r, c))
    }
}

// =============================================================================================
// ===== Mathematical trait impls
// =============================================================================================


// =============================================================================================
// ===== Matrix * vector multiplication (transformations)
// =============================================================================================


// =============================================================================================
// ===== Matrix * matrix multiplication (composition)
// =============================================================================================


// =============================================================================================
// ===== Storage utilities
// =============================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
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
