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
        #[cfg(feature = "nightly")]
        pub fn extend(self, new: T) -> $ty<T, { N + 1 }, S> {
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
        #[cfg(feature = "nightly")]
        pub fn truncate(self) -> $ty<T, { N - 1 }, S> {
            let mut out = [T::zero(); N - 1];
            for i in 0..N - 1 {
                out[i] = self[i];
            }
            out.into()
        }

        #[doc = concat!(" Reinterprets this ", $tys_lower, " as being in the")]
        /// space `Target` instead of `S`. Before calling this, make sure this
        /// operation makes semantic sense and don't just use it to get rid of
        /// compiler errors.
        pub fn in_space<Target: Space>(self) -> $ty<T, N, Target> {
            $ty(self.0, PhantomData)
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
        pub fn map<R: Scalar, F: FnMut(T) -> R>(self, f: F) -> $ty<R, N, S> {
            self.0.map(f).into()
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
        pub fn zip_map<U, R, F>(self, other: $ty<U, N, S>, mut f: F) -> $ty<R, N, S>
        where
            U: Scalar,
            R: Scalar,
            F: FnMut(T, U) -> R,
        {
            std::array::from_fn(|i| f(self.0[i], other.0[i])).into()
        }

        #[doc = concat!("Returns a byte slice of this ", $tys_lower, ", ")]
        /// representing the full raw data. Useful to pass to graphics APIs.
        pub fn as_bytes(&self) -> &[u8] {
            bytemuck::bytes_of(self)
        }

        #[doc = concat!("Converts this ", $tys_lower, " to an array. This")]
        /// is equivalent to using the corresponding `From<Self> for [T; N]`
        /// impl.
        pub fn to_array(self) -> [T; N] {
            self.into()
        }

        #[doc = concat!("Returns an iterator over all components of this ", $tys_lower, ",")]
        /// yielding them in the obvious order, starting with `x`.
        pub fn iter(self) -> impl Iterator<Item = T> {
            self.to_array().into_iter()
        }
    };
}


macro_rules! shared_methods2 {
    ($ty:ident, $tys_lower:literal) => {
        #[doc = concat!("Returns a 2D ", $tys_lower, " from the given coordinates.")]
        pub fn new(x: T, y: T) -> Self {
            [x, y].into()
        }
    }
}

macro_rules! shared_methods3 {
    ($ty:ident, $tys_lower:literal) => {
        #[doc = concat!("Returns a 3D ", $tys_lower, " from the given coordinates.")]
        pub fn new(x: T, y: T, z: T) -> Self {
            [x, y, z].into()
        }
    }
}

macro_rules! shared_impls {
    ($ty:ident, $tys_lower:literal, $debug:literal) => {
        use std::{
            fmt,
            ops::{self, Index, IndexMut},
        };

        impl<T: Scalar, const N: usize, S: Space> Index<usize> for $ty<T, N, S> {
            type Output = T;
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<T: Scalar, const N: usize, S: Space> IndexMut<usize> for $ty<T, N, S> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<T: Scalar, const N: usize, S: Space> From<[T; N]> for $ty<T, N, S> {
            fn from(src: [T; N]) -> Self {
                Self(src, std::marker::PhantomData)
            }
        }

        impl<T: Scalar, const N: usize, S: Space> From<$ty<T, N, S>> for [T; N] {
            fn from(src: $ty<T, N, S>) -> Self {
                src.0
            }
        }

        impl<T: Scalar, const N: usize, S: Space> AsRef<[T; N]> for $ty<T, N, S> {
            fn as_ref(&self) -> &[T; N] {
                &self.0
            }
        }

        impl<T: Scalar, const N: usize, S: Space> AsMut<[T; N]> for $ty<T, N, S> {
            fn as_mut(&mut self) -> &mut [T; N] {
                &mut self.0
            }
        }

        impl<T: Scalar, const N: usize, S: Space> fmt::Debug for $ty<T, N, S> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", $debug)?;
                crate::util::debug_list_one_line(&self.0, f)
            }
        }
    };
}
