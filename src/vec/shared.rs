macro_rules! shared_methods {
    ($ty:ident, $tys_lower:literal) => {
        #[doc = concat!(
            "Applies the given function to each component and returns the resulting ",
            $tys_lower,
            ". Very similar to `[T; N]::map`.",
        )]
        pub fn map<R, F: FnMut(T) -> R>(self, f: F) -> $ty<R, N> {
            $ty(self.0.map(f))
        }
    };
}


macro_rules! shared_methods2 {
    ($ty:ident, $tys_lower:literal) => {
        #[doc = concat!("Returns a 2D ", $tys_lower, " from the given coordinates.")]
        pub fn new(x: T, y: T) -> Self {
            Self([x, y])
        }
    }
}

macro_rules! shared_methods3 {
    ($ty:ident, $tys_lower:literal) => {
        #[doc = concat!("Returns a 3D ", $tys_lower, " from the given coordinates.")]
        pub fn new(x: T, y: T, z: T) -> Self {
            Self([x, y, z])
        }
    }
}

macro_rules! shared_impls {
    ($ty:ident, $tys_lower:literal, $debug:literal) => {
        use std::{
            fmt,
            ops::{self, Index, IndexMut},
        };

        impl<T, const N: usize> Index<usize> for $ty<T, N> {
            type Output = T;
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<T, const N: usize> IndexMut<usize> for $ty<T, N> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<T, const N: usize> From<[T; N]> for $ty<T, N> {
            fn from(src: [T; N]) -> Self {
                Self(src)
            }
        }

        impl<T, const N: usize> Into<[T; N]> for $ty<T, N> {
            fn into(self) -> [T; N] {
                self.0
            }
        }

        impl<T, const N: usize> AsRef<[T; N]> for $ty<T, N> {
            fn as_ref(&self) -> &[T; N] {
                &self.0
            }
        }

        impl<T, const N: usize> AsMut<[T; N]> for $ty<T, N> {
            fn as_mut(&mut self) -> &mut [T; N] {
                &mut self.0
            }
        }

        impl<T: fmt::Debug, const N: usize> fmt::Debug for $ty<T, N> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{} [", $debug)?;
                for (i, e) in self.0.iter().enumerate() {
                    if i != 0 {
                        write!(f, ", ")?;
                    }
                    e.fmt(f)?;
                }
                write!(f, "]")
            }
        }

        impl<T: ops::Neg, const N: usize> ops::Neg for $ty<T, N> {
            type Output = $ty<T::Output, N>;
            fn neg(self) -> Self::Output {
                self.map(|c| -c)
            }
        }

        /// Scalar multipliation: `vector * scalar`.
        impl<S: Clone, T: ops::Mul<S>, const N: usize> ops::Mul<S> for $ty<T, N> {
            type Output = $ty<T::Output, N>;
            fn mul(self, rhs: S) -> Self::Output {
                self.map(|c| c * rhs.clone())
            }
        }

        /// Scalar multipliation: `vector *= scalar`.
        impl<S: Clone, T: ops::MulAssign<S>, const N: usize> ops::MulAssign<S> for $ty<T, N> {
            fn mul_assign(&mut self, rhs: S) {
                for c in &mut self.0 {
                    *c *= rhs.clone();
                }
            }
        }

        /// Scalar division: `vector / scalar`.
        impl<S: Clone, T: ops::Div<S>, const N: usize> ops::Div<S> for $ty<T, N> {
            type Output = $ty<T::Output, N>;
            fn div(self, rhs: S) -> Self::Output {
                self.map(|c| c / rhs.clone())
            }
        }

        /// Scalar division: `vector /= scalar`.
        impl<S: Clone, T: ops::DivAssign<S>, const N: usize> ops::DivAssign<S> for $ty<T, N> {
            fn div_assign(&mut self, rhs: S) {
                for c in &mut self.0 {
                    *c /= rhs.clone();
                }
            }
        }
    };
}
