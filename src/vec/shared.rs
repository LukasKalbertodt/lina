macro_rules! shared_methods {
    ($ty:ident, $tys_lower:literal, $ctor2:literal, $ctor3:literal) => {
        /// Returns a
        #[doc = concat!(" ", $tys_lower, " ")]
        /// with dimension `N + 1` by adding `new` as new component.
        ///
        /// ```
        #[doc = concat!("use lina::{", $ctor2, ", ", $ctor3, "};")]
        ///
        #[doc = concat!("assert_eq!(", $ctor2, "(2i32, 4).extend(9), ", $ctor3, "(2, 4, 9));")]
        /// ```
        pub fn extend(self, new: T) -> $ty<T, { N + 1 }> {
            let mut out = [T::zero(); N + 1];
            for i in 0..N {
                out[i] = self[i];
            }
            out[N] = new;
            out.into()
        }

        /// Returns a
        #[doc = concat!(" ", $tys_lower, " ")]
        /// with dimension `N - 1` by removing the last dimension.
        ///
        /// ```
        #[doc = concat!("use lina::{", $ctor2, ", ", $ctor3, "};")]
        ///
        #[doc = concat!("assert_eq!(", $ctor3, "(2i32, 4, 9).truncate(), ", $ctor2, "(2, 4));")]
        /// ```
        pub fn truncate(self) -> $ty<T, { N - 1 }> {
            let mut out = [T::zero(); N - 1];
            for i in 0..N - 1 {
                out[i] = self[i];
            }
            out.into()
        }

        /// Applies the given function to each component and returns the resulting
        #[doc = concat!(" ", $tys_lower, ".")]
        /// Very similar to `[T; N]::map`.
        ///
        /// ```
        #[doc = concat!("use lina::", $ctor3, ";")]
        ///
        #[doc = concat!("let r = ", $ctor3, "(1.0, 2.0, 3.0).map(|c| c * 2.0);")]
        #[doc = concat!("assert_eq!(r, ", $ctor3, "(2.0, 4.0, 6.0));")]
        /// ```
        pub fn map<R: Scalar, F: FnMut(T) -> R>(self, f: F) -> $ty<R, N> {
            $ty(self.0.map(f))
        }

        /// Pairs up the components of `self` and `other`, applies the given
        /// function to each pair and returns the resulting
        #[doc = concat!(" ", $tys_lower, ".")]
        ///
        /// This can be used for all "component-wise" operations, like
        /// multiplication:
        ///
        /// ```
        #[doc = concat!("use lina::", $ctor3, ";")]
        ///
        #[doc = concat!("let a = ", $ctor3, "(1, 2, 3);")]
        #[doc = concat!("let b = ", $ctor3, "(4, 5, 6);")]
        /// let r = a.zip_map(b, |ac, bc| ac * bc);
        #[doc = concat!("assert_eq!(r, ", $ctor3, "(4, 10, 18));")]
        /// ```
        pub fn zip_map<U, R, F>(self, other: $ty<U, N>, f: F) -> $ty<R, N>
        where
            U: Scalar,
            R: Scalar,
            F: FnMut(T, U) -> R,
        {
            $ty(crate::util::zip_map(self.0, other.0, f))
        }

        #[doc = concat!("Returns a byte slice of this ", $tys_lower, ", ")]
        /// representing the full raw data. Useful to pass to graphics APIs.
        pub fn as_bytes(&self) -> &[u8] {
            bytemuck::bytes_of(self)
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

        impl<T: Scalar, const N: usize> Index<usize> for $ty<T, N> {
            type Output = T;
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<T: Scalar, const N: usize> IndexMut<usize> for $ty<T, N> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<T: Scalar, const N: usize> From<[T; N]> for $ty<T, N> {
            fn from(src: [T; N]) -> Self {
                Self(src)
            }
        }

        impl<T: Scalar, const N: usize> Into<[T; N]> for $ty<T, N> {
            fn into(self) -> [T; N] {
                self.0
            }
        }

        impl<T: Scalar, const N: usize> AsRef<[T; N]> for $ty<T, N> {
            fn as_ref(&self) -> &[T; N] {
                &self.0
            }
        }

        impl<T: Scalar, const N: usize> AsMut<[T; N]> for $ty<T, N> {
            fn as_mut(&mut self) -> &mut [T; N] {
                &mut self.0
            }
        }

        impl<T: Scalar, const N: usize> fmt::Debug for $ty<T, N> {
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
    };
}
