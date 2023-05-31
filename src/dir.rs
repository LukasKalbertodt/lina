use std::{hash::Hash, fmt, ops};

use bytemuck::{Zeroable, Pod};
use num_traits::NumCast;

use crate::{Space, Vector, Scalar, WorldSpace, SphericalDir, Float, SphericalPos, Vec2, Vec3, Point};



/// A direction in `N`-dimensional space, represented by a unit vector.
///
/// The API of this type is fairly limited to ensure it always contains a unit
/// vector.
#[repr(transparent)]
pub struct Dir<T: Scalar, const N: usize, S: Space>(Vector<T, N, S>);

/// A 2-dimensional direction.
pub type Dir2<T, S = WorldSpace> = Dir<T, 2, S>;
/// A 3-dimensional direction.
pub type Dir3<T, S = WorldSpace> = Dir<T, 3, S>;

/// A 2-dimensional direction with scalar type `f32`.
pub type Dir2f<S = WorldSpace> = Dir2<f32, S>;
/// A 3-dimensional direction with scalar type `f32`.
pub type Dir3f<S = WorldSpace> = Dir3<f32, S>;

/// A 2-dimensional direction with scalar type `f64`.
pub type Dir2d<S = WorldSpace> = Dir2<f64, S>;
/// A 3-dimensional direction with scalar type `f64`.
pub type Dir3d<S = WorldSpace> = Dir3<f64, S>;

impl<T: Scalar, const N: usize, S: Space> Dir<T, N, S> {
    /// Returns the direction of the given vector. If `v` is the null vector,
    /// this function panics.
    pub fn from_vec(v: Vector<T, N, S>) -> Self
    where
        T: Float,
    {
        v.into()
    }

    /// Returns the wrapped input vector, assuming it is normalized.
    pub fn from_unit_vec_unchecked(v: Vector<T, N, S>) -> Self {
        Self(v)
    }

    /// Returns the unit vector that represents this direction.
    pub fn to_unit_vec(&self) -> Vector<T, N, S> {
        self.0
    }

    /// Returns the point corresponding to this unit vector (a point on the unit sphere).
    pub fn to_point(&self) -> Point<T, N, S> {
        self.0.to_point()
    }

    /// Reinterprets this direction as being in the space `Target` instead of
    /// `S`. Before calling this, make sure this operation makes semantic sense
    /// and don't just use it to get rid of compiler errors.
    pub fn in_space<Target: Space>(self) -> Dir<T, N, Target> {
        Dir(self.0.in_space())
    }

    /// Casts `self` to using `f32` as scalar.
    pub fn to_f32(self) -> Dir<f32, N, S>
    where
        T: NumCast,
    {

        Dir(self.0.map(|s| num_traits::cast(s).unwrap()))
    }

    /// Casts `self` to using `f64` as scalar.
    pub fn to_f64(self) -> Dir<f64, N, S>
    where
        T: NumCast,
    {
        Dir(self.0.map(|s| num_traits::cast(s).unwrap()))
    }

    /// Returns a byte slice of the unit vector representing this direction.
    /// Useful to pass to graphics APIs.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Returns the components of this unit vector as arary.
    pub fn to_array(self) -> [T; N] {
        self.into()
    }
}

impl<T: Scalar, S: Space> Dir2<T, S> {
    /// Creates a `Dir` without checking that the given values form a unit
    /// vector.
    pub const fn new_unchecked(x: T, y: T) -> Self {
        Self(Vec2::new(x, y))
    }

    /// Returns `(1, 0, 0)`.
    pub fn unit_x() -> Self {
        Self(Vector::unit_x())
    }

    /// Returns `(0, 1, 0)`.
    pub fn unit_y() -> Self {
        Self(Vector::unit_y())
    }

    /// Returns the x coordinate.
    pub fn x(&self) -> T {
        self.0.x
    }

    /// Returns the y coordinate.
    pub fn y(&self) -> T {
        self.0.y
    }
}

impl<T: Scalar, S: Space> Dir3<T, S> {
    /// Creates a `Dir` without checking that the given values form a unit
    /// vector.
    pub const fn new_unchecked(x: T, y: T, z: T) -> Self {
        Self(Vec3::new(x, y, z))
    }

    /// Returns `(1, 0, 0)`.
    pub fn unit_x() -> Self {
        Self(Vector::unit_x())
    }

    /// Returns `(0, 1, 0)`.
    pub fn unit_y() -> Self {
        Self(Vector::unit_y())
    }

    /// Returns `(0, 0, 1)`.
    pub fn unit_z() -> Self {
        Self(Vector::unit_z())
    }

    /// Returns the x coordinate.
    pub fn x(&self) -> T {
        self.0.x
    }

    /// Returns the y coordinate.
    pub fn y(&self) -> T {
        self.0.y
    }

    /// Returns the z coordinate.
    pub fn z(&self) -> T {
        self.0.z
    }
}

impl<T: Float, S: Space> From<SphericalDir<T, S>> for Dir3<T, S> {
    fn from(src: SphericalDir<T, S>) -> Self {
        Self(src.to_unit_vec())
    }
}

impl<T: Float, S: Space> From<Dir3<T, S>> for SphericalDir<T, S> {
    fn from(src: Dir3<T, S>) -> Self {
        src.0.into()
    }
}

/// Simply disregards the radius.
impl<T: Float, S: Space> From<SphericalPos<T, S>> for Dir3<T, S> {
    fn from(src: SphericalPos<T, S>) -> Self {
        Self(src.without_radius().to_unit_vec())
    }
}

/// If the null vector is passed, this function panics.
impl<T: Float, const N: usize, S: Space> From<Vector<T, N, S>> for Dir<T, N, S> {
    fn from(v: Vector<T, N, S>) -> Self {
        let l = v.length();
        assert!(!l.is_zero(), "zero vector passed to `Dir::from`");
        Self(v / l)
    }
}

impl<T: Float, const N: usize, S: Space> From<Dir<T, N, S>> for Vector<T, N, S> {
    fn from(src: Dir<T, N, S>) -> Self {
        src.0
    }
}


impl<T: Scalar, S: Space> fmt::Debug for Dir3<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dir")?;
        crate::util::debug_list_one_line(&self.0.0, f)
    }
}

// This is just a `repr(transparent)` wrapper around `Vector`.
unsafe impl<T: Scalar + Zeroable, const N: usize, S: Space> Zeroable for Dir<T, N, S> {}
unsafe impl<T: Scalar + Pod, const N: usize, S: Space> Pod for Dir<T, N, S> {}

impl<T: Scalar, const N: usize, S: Space> Clone for Dir<T, N, S> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<T: Scalar, const N: usize, S: Space> Copy for Dir<T, N, S> {}
impl<T: Scalar, const N: usize, S: Space> PartialEq for Dir<T, N, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: Scalar + Hash, const N: usize, S: Space> Hash for Dir<T, N, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}


impl<T: Scalar, const N: usize, S: Space> ops::Mul<T> for Dir<T, N, S> {
    type Output = Vector<T, N, S>;
    fn mul(self, rhs: T) -> Self::Output {
        self.0 * rhs
    }
}

macro_rules! impl_scalar_mul {
    ($($ty:ident),*) => {
        $(
            impl<const N: usize, S: Space> ops::Mul<Dir<$ty, N, S>> for $ty {
                type Output = Vector<$ty, N, S>;
                fn mul(self, rhs: Dir<$ty, N, S>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl<T: Scalar, const N: usize, S: Space> ops::Div<T> for Dir<T, N, S> {
    type Output = Vector<T, N, S>;
    fn div(self, rhs: T) -> Self::Output {
        self.0 / rhs
    }
}

impl<T: Scalar + ops::Neg, const N: usize, S: Space> ops::Neg for Dir<T, N, S>
where
    <T as ops::Neg>::Output: Scalar,
{
    type Output = Dir<<T as ops::Neg>::Output, N, S>;
    fn neg(self) -> Self::Output {
        Dir(-self.0)
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::Index<usize> for Dir<T, N, S> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Scalar, const N: usize, S: Space> From<Dir<T, N, S>> for [T; N] {
    fn from(src: Dir<T, N, S>) -> Self {
        src.0.0
    }
}
