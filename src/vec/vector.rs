use bytemuck::{Pod, Zeroable};

use crate::{
    named_scalar::{HasX, HasY, HasZ, HasW},
    num::{One, Zero},
};


/// An `N`-dimensional vector with scalar type `T`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Vector<T, const N: usize>(pub(crate) [T; N]);

/// A 2-dimensional vector.
pub type Vec2<T> = Vector<T, 2>;

/// A 3-dimensional vector.
pub type Vec3<T> = Vector<T, 3>;

/// A 4-dimensional vector.
pub type Vec4<T> = Vector<T, 4>;

/// A 2-dimensional vector with scalar type `f32`.
pub type Vec2f = Vec2<f32>;

/// A 3-dimensional vector with scalar type `f32`.
pub type Vec3f = Vec3<f32>;

/// A 4-dimensional vector with scalar type `f32`.
pub type Vec4f = Vec4<f32>;


/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Zeroable, const N: usize> Zeroable for Vector<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Pod, const N: usize> Pod for Vector<T, N> {}

impl<T, const N: usize> Vector<T, N> {
    /// Returns the zero vector.
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self([(); N].map(|_| T::zero()))
    }

    /// Returns a unit vector in x direction.
    ///
    /// ```
    /// assert_eq!(lina::Vec2f::unit_x(), lina::vec2(1.0, 0.0));
    /// ```
    pub fn unit_x() -> Self
    where
        T: Zero + One,
        Self: HasX<Scalar = T>,
    {
        let mut out = Self::zero();
        *out.x_mut() = T::one();
        out
    }

    /// Returns a unit vector in y direction.
    ///
    /// ```
    /// assert_eq!(lina::Vec3f::unit_y(), lina::vec3(0.0, 1.0, 0.0));
    /// ```
    pub fn unit_y() -> Self
    where
        T: Zero + One,
        Self: HasY<Scalar = T>,
    {
        let mut out = Self::zero();
        *out.y_mut() = T::one();
        out
    }

    /// Returns a unit vector in z direction.
    ///
    /// ```
    /// assert_eq!(lina::Vec3f::unit_z(), lina::vec3(0.0, 0.0, 1.0));
    /// ```
    pub fn unit_z() -> Self
    where
        T: Zero + One,
        Self: HasZ<Scalar = T>,
    {
        let mut out = Self::zero();
        *out.z_mut() = T::one();
        out
    }

    /// Returns a unit vector in w direction.
    ///
    /// ```
    /// assert_eq!(lina::Vec4f::unit_w(), lina::vec4(0.0, 0.0, 0.0, 1.0));
    /// ```
    pub fn unit_w() -> Self
    where
        T: Zero + One,
        Self: HasW<Scalar = T>,
    {
        let mut out = Self::zero();
        *out.w_mut() = T::one();
        out
    }

    shared_methods!(Vector, "vector");
}

impl<T> Vector<T, 2> {
    shared_methods2!(Vector, "vector");
}

impl<T> Vector<T, 3> {
    shared_methods3!(Vector, "vector");
}

impl<T> Vector<T, 4> {
    /// Returns a 4D vector from the given coordinates.
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self([x, y, z, w])
    }
}

shared_impls!(Vector, "vector", "Vec");

/// Shorthand for `Vec2::new(...)`.
pub fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vec2::new(x, y)
}

/// Shorthand for `Vec3::new(...)`.
pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3::new(x, y, z)
}

/// Shorthand for `Vec4::new(...)`.
pub fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4::new(x, y, z, w)
}
