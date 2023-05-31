use std::{marker::PhantomData, fmt, ops};

use bytemuck::{Zeroable, Pod};

use crate::{Scalar, Space, WorldSpace, Point};


/// A point in `N`-dimensional space represented by *homogeneous coordinates*
/// (with `N + 1` values).
///
/// Any given point can not only be represented in the familiar cartesian
/// coordinates, but also in so called homogeneous coordinates. In fact, it has
/// infinitely many representations in homogeneous coordinates! To represent a
/// point in N-dimensional space, homogeneous coordinates store N+1 values: N
/// normal coordinates and a *weight*. The cartesian coordinates `(x, y, z)`
/// and the homogeneous coordinates `(W·x, W·y, W·z; W)` (for any `W != 0`) all
/// represent the same point in 3D space! E.g., `(1, 2, 3)`, `(1, 2, 3; 1)` and
/// `(2, 4, 6; 2)` represent the same point.
///
/// Homogeneous coordinates are used to represent affine and projective
/// transformations of N-dimensional points as (N+1)x(N+1) matrix. You usually
/// only ever encounter them in that context.
///
/// The most important functions of this type is to convert from and to
/// [`Point`] via the `From` impls or [`Self::to_point`].
///
/// ```
/// use lina::{HcPoint, point2};
///
/// let mut hc = HcPoint::from(point2(7.0, 5.0));
/// assert_eq!(hc.x, 7.0);
/// assert_eq!(hc.y, 5.0);
/// assert_eq!(hc.weight, 1.0);
///
/// hc.weight = 2.0;
/// assert_eq!(hc.to_point(), point2(3.5, 2.5));
/// ```
#[repr(C)]
pub struct HcPoint<T: Scalar, const N: usize, S: Space = WorldSpace> {
    pub(crate) coords: [T; N],
    pub weight: T,
    _dummy: PhantomData<S>,
}

/// A point in 2-dimensional space represented by homogeneous coordinates
/// (with 3 values).
pub type HcPoint2<T, S = WorldSpace> = HcPoint<T, 2, S>;

/// A point in 3-dimensional space represented by homogeneous coordinates
/// (with 4 values).
pub type HcPoint3<T, S = WorldSpace> = HcPoint<T, 3, S>;


impl<T: Scalar, const N: usize, S: Space> HcPoint<T, N, S> {
    /// Returns a representation of the origin in N-dimensional space.
    pub fn origin() -> Self {
        Self::new(std::array::from_fn(|_| T::zero()), T::one())
    }

    /// Creates a new point with the given coordinates and weight.
    pub fn new(coords: impl Into<[T; N]>, weight: T) -> Self {
        Self {
            coords: coords.into(),
            weight,
            _dummy: PhantomData,
        }
    }

    /// Converts this homogeneous coordinates point into a cartesian coordinate
    /// point (by dividing all components by the weight).
    ///
    /// ```
    /// use lina::{HcPoint, point3};
    ///
    /// let hc = HcPoint::new([1.0, 2.0, 3.0], 2.0);
    /// assert_eq!(hc.to_point(), point3(0.5, 1.0, 1.5));
    /// ```
    pub fn to_point(self) -> Point<T, N, S> {
        // TODO: maybe optimize?
        // - Check if `weight` is 1 and just return?
        // - For floats, calculate 1 / weight once and then multiply by that
        //   (or does the compiler do that for us?)
        self.coords.map(|s| s / self.weight).into()
    }

    /// Reinterprets this points as being in the space `Target` instead of `S`.
    /// Before calling this, make sure this operation makes semantic sense and
    /// don't just use it to get rid of compiler errors.
    pub const fn in_space<Target: Space>(self) -> HcPoint<T, N, Target> {
        HcPoint {
            coords: self.coords,
            weight: self.weight,
            _dummy: PhantomData,
        }
    }


    /// Casts `self` to using `f32` as scalar.
    pub fn to_f32(self) -> HcPoint<f32, N, S>
    where
        T: num_traits::NumCast,
    {
        HcPoint {
            coords: self.coords.map(|s| num_traits::cast(s).unwrap()),
            weight: num_traits::cast(self.weight).unwrap(),
            _dummy: PhantomData,
        }
    }

    /// Casts `self` to using `f64` as scalar.
    pub fn to_f64(self) -> HcPoint<f64, N, S>
    where
        T: num_traits::NumCast,
    {
        HcPoint {
            coords: self.coords.map(|s| num_traits::cast(s).unwrap()),
            weight: num_traits::cast(self.weight).unwrap(),
            _dummy: PhantomData,
        }
    }

    /// Returns a byte slice of this point representing the full raw data
    /// (components + weight). Useful to pass to graphics APIs.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}


// =============================================================================================
// ===== Trait impls
// =============================================================================================

impl<T: Scalar + std::hash::Hash, const N: usize, S: Space> std::hash::Hash for HcPoint<T, N, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coords.hash(state);
        self.weight.hash(state);
    }
}
impl<T: Scalar, const N: usize, S: Space> PartialEq for HcPoint<T, N, S> {
    fn eq(&self, other: &Self) -> bool {
        self.weight.eq(&other.weight) && self.coords.eq(&other.coords)
    }
}
impl<T: Scalar + Eq, const N: usize, S: Space> Eq for HcPoint<T, N, S> {}

impl<T: Scalar, const N: usize, S: Space> Clone for HcPoint<T, N, S> {
    fn clone(&self) -> Self {
        Self {
            coords: self.coords,
            weight: self.weight,
            _dummy: PhantomData,
        }
    }
}
impl<T: Scalar, const N: usize, S: Space> Copy for HcPoint<T, N, S> {}

/// Due to the `T: Zeroable` bound, all our fields are zeroable. `weight`
/// trivially is, `coords` is as `[T; N]` implements `Zeroable` as well. And
/// `PhantomData` is zero sized, so implements it as well. Further, due to
/// `repr(C)`, this type has no padding bytes.
unsafe impl<T: Scalar + Zeroable, const N: usize, S: Space> Zeroable for HcPoint<T, N, S> {}

/// This type, with the `T: Pod` bounds, satisfies all properties required by
/// `Pod`. All bit patterns are allowed, no padding bytes, trivially inhabiteda
/// nad `repr(C)`.
unsafe impl<T: Scalar + Pod, const N: usize, S: Space> Pod for HcPoint<T, N, S> {}

impl<T: Scalar, const N: usize, S: Space> ops::Index<usize> for HcPoint<T, N, S> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match () {
            () if index < N => &self.coords[index],
            () if index == N => &self.weight,
            _ => panic!("index ({index}) out of bounds ({})", N + 1),
        }
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::IndexMut<usize> for HcPoint<T, N, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match () {
            () if index < N => &mut self.coords[index],
            () if index == N => &mut self.weight,
            _ => panic!("index ({index}) out of bounds ({})", N + 1),
        }
    }
}

impl<T: Scalar, const N: usize, S: Space> From<Point<T, N, S>> for HcPoint<T, N, S> {
    fn from(src: Point<T, N, S>) -> Self {
        Self::new(src, T::one())
    }
}

impl<T: Scalar, const N: usize, S: Space> From<HcPoint<T, N, S>> for Point<T, N, S> {
    fn from(src: HcPoint<T, N, S>) -> Self {
        src.to_point()
    }
}

macro_rules! impl_np1 {
    ($n:expr, $np1:expr) => {
        impl<T: Scalar, S: Space> HcPoint<T, $n, S> {
            /// Converts this to an array of size N+1 with all the stored values
            /// (components + weight). This is equivalent to using the corresponding
            /// `From<Self> for [T; N + 1]` impl.
            pub fn to_array(self) -> [T; $np1] {
                self.into()
            }
        }

        impl<T: Scalar, S: Space> From<[T; $np1]> for HcPoint<T, $n, S> {
            fn from(src: [T; $np1]) -> Self {
                let coords = std::array::from_fn(|i| src[i]);
                Self::new(coords, src[$n])
            }
        }

        impl<T: Scalar, S: Space> From<HcPoint<T, $n, S>> for [T; $np1] {
            fn from(src: HcPoint<T, $n, S>) -> Self {
                std::array::from_fn(|i| if i == $n { src.weight } else { src.coords[i] })
            }
        }

        impl<T: Scalar, S: Space> AsRef<[T; $np1]> for HcPoint<T, $n, S> {
            fn as_ref(&self) -> &[T; $np1] {
                bytemuck::cast_ref(self)
            }
        }

        impl<T: Scalar, S: Space> AsMut<[T; $np1]> for HcPoint<T, $n, S> {
            fn as_mut(&mut self) -> &mut [T; $np1] {
                bytemuck::cast_mut(self)
            }
        }
    };
}

impl_np1!(2, 3);


impl<T: Scalar, const N: usize, S: Space> fmt::Debug for HcPoint<T, N, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HcPoint[")?;
        for (i, e) in self.coords.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            e.fmt(f)?;
        }
        write!(f, "; ")?;
        self.weight.fmt(f)?;
        write!(f, "]")
    }
}
