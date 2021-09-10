use bytemuck::{Pod, Zeroable};

use crate::{
    Point, Scalar, Real,
    named_scalar::{HasX, HasY, HasZ, HasW},
    util::zip_map,
};


/// An `N`-dimensional vector with scalar type `T`. This represents
/// a *displacement* in space.
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

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Returns the zero vector.
    pub fn zero() -> Self {
        Self([(); N].map(|_| T::zero()))
    }

    /// Returns a unit vector in x direction.
    ///
    /// ```
    /// assert_eq!(lina::Vec2f::unit_x(), lina::vec2(1.0, 0.0));
    /// ```
    pub fn unit_x() -> Self
    where
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
        Self: HasW<Scalar = T>,
    {
        let mut out = Self::zero();
        *out.w_mut() = T::one();
        out
    }

    /// Converts this vector into a point without changing the component values.
    /// Semantically equivalent to `Point::origin() + self`. Please think twice
    /// before using this method as it blindly changes the semantics of your
    /// value.
    pub fn to_point(self) -> Point<T, N> {
        Point(self.0)
    }

    /// Returns the *squared* length of this vector. This is faster than
    /// [`length`][Self::length] as no sqrt operation is required. And since
    /// the square root is continious, if you only need to compare two lengths,
    /// you can use this method.
    pub fn length2(&self) -> T {
        self.0.iter().map(|&c| c * c).fold(T::zero(), |acc, e| acc + e)
    }

    /// Returns the length of this vector. If you only need to compare two
    /// lengths, [`length2`][Self::length2] is faster.
    pub fn length(&self) -> T
    where
        T: Real,
    {
        self.length2().sqrt()
    }

    /// Returns a normalized version of this vector (i.e. a vector with the same
    /// direction, but length 1). Also see [`normalize`][Self::normalize].
    pub fn normalized(mut self) -> Self
    where
        T: Real,
    {
        self.normalize();
        self
    }

    /// Normalizes the vector *in place* (i.e. maintain the direction but make
    /// it length 1). Also see [`normalized`][Self::normalized].
    pub fn normalize(&mut self)
    where
        T: Real,
    {
        *self = *self / self.length();
    }

    shared_methods!(Vector, "vector");
}

impl<T: Scalar> Vector<T, 2> {
    shared_methods2!(Vector, "vector");
}

impl<T: Scalar> Vector<T, 3> {
    shared_methods3!(Vector, "vector");
}

impl<T: Scalar> Vector<T, 4> {
    /// Returns a 4D vector from the given coordinates.
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self([x, y, z, w])
    }
}

shared_impls!(Vector, "vector", "Vec");

/// Shorthand for `Vec2::new(...)`.
pub fn vec2<T: Scalar>(x: T, y: T) -> Vec2<T> {
    Vec2::new(x, y)
}

/// Shorthand for `Vec3::new(...)`.
pub fn vec3<T: Scalar>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3::new(x, y, z)
}

/// Shorthand for `Vec4::new(...)`.
pub fn vec4<T: Scalar>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4::new(x, y, z, w)
}

impl<T: ops::Add<U>, U, const N: usize> ops::Add<Vector<U, N>> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        Vector(zip_map(self.0, rhs.0, |l, r| l + r))
    }
}

impl<T: ops::AddAssign<U>, U, const N: usize> ops::AddAssign<Vector<U, N>> for Vector<T, N> {
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T: ops::Sub<U>, U, const N: usize> ops::Sub<Vector<U, N>> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        Vector(zip_map(self.0, rhs.0, |l, r| l - r))
    }
}

impl<T: ops::SubAssign<U>, U, const N: usize> ops::SubAssign<Vector<U, N>> for Vector<T, N> {
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T: Scalar + ops::Neg, const N: usize> ops::Neg for Vector<T, N> {
    type Output = Vector<<T as ops::Neg>::Output, N>;
    fn neg(self) -> Self::Output {
        self.map(|c| -c)
    }
}

/// Scalar multipliation: `vector * scalar`.
impl<S: Clone, T: Scalar + ops::Mul<S>, const N: usize> ops::Mul<S> for Vector<T, N> {
    type Output = Vector<<T as ops::Mul<S>>::Output, N>;
    fn mul(self, rhs: S) -> Self::Output {
        self.map(|c| c * rhs.clone())
    }
}

/// Scalar multipliation: `vector *= scalar`.
impl<S: Clone, T: ops::MulAssign<S>, const N: usize> ops::MulAssign<S> for Vector<T, N> {
    fn mul_assign(&mut self, rhs: S) {
        for c in &mut self.0 {
            *c *= rhs.clone();
        }
    }
}

/// Scalar division: `vector / scalar`.
impl<S: Clone, T: Scalar + ops::Div<S>, const N: usize> ops::Div<S> for Vector<T, N> {
    type Output = Vector<<T as ops::Div<S>>::Output, N>;
    fn div(self, rhs: S) -> Self::Output {
        self.map(|c| c / rhs.clone())
    }
}

/// Scalar division: `vector /= scalar`.
impl<S: Clone, T: ops::DivAssign<S>, const N: usize> ops::DivAssign<S> for Vector<T, N> {
    fn div_assign(&mut self, rhs: S) {
        for c in &mut self.0 {
            *c /= rhs.clone();
        }
    }
}
