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

macro_rules! shared_trait_impls {
    ($ty:ident) => {
        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > Clone for $ty<T, C, R, Src, Dst> {
            fn clone(&self) -> Self {
                Self(self.0, PhantomData)
            }
        }
        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > Copy for $ty<T, C, R, Src, Dst> {}

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > PartialEq for $ty<T, C, R, Src, Dst> {
            fn eq(&self, other: &Self) -> bool {
                self.0.eq(&other.0)
            }
        }
        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > Eq for $ty<T, C, R, Src, Dst> {}

        impl<
            T: Scalar + std::hash::Hash,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > std::hash::Hash for $ty<T, C, R, Src, Dst> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state)
            }
        }


        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::Add for $ty<T, C, R, Src, Dst> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                self.zip_map(&rhs, |l, r| l + r)
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::AddAssign for $ty<T, C, R, Src, Dst> {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::Sub for $ty<T, C, R, Src, Dst> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                self.zip_map(&rhs, |l, r| l - r)
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::SubAssign for $ty<T, C, R, Src, Dst> {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::Mul<T> for $ty<T, C, R, Src, Dst> {
            type Output = Self;
            fn mul(self, rhs: T) -> Self::Output {
                self.map(|elem| elem * rhs)
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::MulAssign<T> for $ty<T, C, R, Src, Dst> {
            fn mul_assign(&mut self, rhs: T) {
                *self = *self * rhs;
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::Div<T> for $ty<T, C, R, Src, Dst> {
            type Output = Self;
            fn div(self, rhs: T) -> Self::Output {
                self.map(|elem| elem / rhs)
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::DivAssign<T> for $ty<T, C, R, Src, Dst> {
            fn div_assign(&mut self, rhs: T) {
                *self = *self / rhs;
            }
        }

        impl<
            T: Scalar + ops::Neg,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > ops::Neg for $ty<T, C, R, Src, Dst>
        where
            <T as ops::Neg>::Output: Scalar,
        {
            type Output = $ty<<T as ops::Neg>::Output, C, R, Src, Dst>;
            fn neg(self) -> Self::Output {
                self.map(|elem| -elem)
            }
        }

        impl<
            T: Scalar,
            const C: usize,
            const R: usize,
            Src: Space,
            Dst: Space,
        > std::iter::Sum<Self> for $ty<T, C, R, Src, Dst> {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), |acc, x| acc + x)
            }
        }
    };
}
use shared_trait_impls;

// Scalar multiplication: `scalar * matrix`. Unfortunately, due to Rust's orphan
// rules, this cannot be implemented generically. So we just implement it for
// core primitive types.
macro_rules! impl_scalar_mul {
    ($ty:ident => $($scalar:ident),*) => {
        $(
            impl<
                const C: usize,
                const R: usize,
                Src: Space,
                Dst: Space,
            > ops::Mul<$ty<$scalar, C, R, Src, Dst>> for $scalar {
                type Output = $ty<$scalar, C, R, Src, Dst>;
                fn mul(self, rhs: $ty<$scalar, C, R, Src, Dst>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}
use impl_scalar_mul;


#[cfg(test)]
mod tests;
