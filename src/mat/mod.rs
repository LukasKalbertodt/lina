use std::fmt;

use crate::Scalar;

mod inv4;
pub(crate) mod linear;
pub(crate) mod hc;


/// Helper to implement `fmt::Debug` for matrices.
fn debug_matrix_impl<T: Scalar>(
    f: &mut fmt::Formatter,
    cols: usize,
    rows: usize,
    elem: impl Fn(usize, usize) -> T,
) -> fmt::Result {
    /// Helper type to format a matrix row.
    struct DebugRow<F> {
        row_index: usize,
        cols: usize,
        elem: F,
    }

    impl<F: Fn(usize) -> T, T: fmt::Debug> fmt::Debug for DebugRow<F> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if f.alternate() {
                write!(f, "[")?;
            } else {
                write!(f, "row{} [", self.row_index)?;
            }

            for c in 0..self.cols {
                if c != 0 {
                    write!(f, ", ")?;
                }
                (self.elem)(c).fmt(f)?;
            }
            write!(f, "]")
        }
    }

    let mut list = f.debug_list();
    for r in 0..rows {
        list.entry(&DebugRow {
            row_index: r,
            cols,
            elem: |c| elem(r, c),
        });
    }
    list.finish()
}

macro_rules! impl_math_traits {
    ($ty:ident) => {
        impl<T: Scalar, const C: usize, const R: usize> ops::Add for $ty<T, C, R> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                self.zip_map(rhs, |l, r| l + r)
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::AddAssign for $ty<T, C, R> {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::Sub for $ty<T, C, R> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                self.zip_map(rhs, |l, r| l - r)
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::SubAssign for $ty<T, C, R> {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::Mul<T> for $ty<T, C, R> {
            type Output = Self;
            fn mul(self, rhs: T) -> Self::Output {
                self.map(|elem| elem * rhs)
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::MulAssign<T> for $ty<T, C, R> {
            fn mul_assign(&mut self, rhs: T) {
                *self = *self * rhs;
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::Div<T> for $ty<T, C, R> {
            type Output = Self;
            fn div(self, rhs: T) -> Self::Output {
                self.map(|elem| elem / rhs)
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> ops::DivAssign<T> for $ty<T, C, R> {
            fn div_assign(&mut self, rhs: T) {
                *self = *self / rhs;
            }
        }

        impl<T: Scalar + ops::Neg, const C: usize, const R: usize> ops::Neg for $ty<T, C, R>
        where
            <T as ops::Neg>::Output: Scalar,
        {
            type Output = $ty<<T as ops::Neg>::Output, C, R>;
            fn neg(self) -> Self::Output {
                self.map(|elem| -elem)
            }
        }

        impl<T: Scalar, const C: usize, const R: usize> std::iter::Sum<Self> for $ty<T, C, R> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + x)
            }
        }
    };
}
use impl_math_traits;

// Scalar multiplication: `scalar * matrix`. Unfortunately, due to Rust's orphan
// rules, this cannot be implemented generically. So we just implement it for
// core primitive types.
macro_rules! impl_scalar_mul {
    ($ty:ident => $($scalar:ident),*) => {
        $(
            impl<const C: usize, const R: usize> ops::Mul<$ty<$scalar, C, R>> for $scalar {
                type Output = $ty<$scalar, C, R>;
                fn mul(self, rhs: $ty<$scalar, C, R>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}
use impl_scalar_mul;


#[cfg(test)]
mod tests;
