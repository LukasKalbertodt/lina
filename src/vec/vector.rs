use bytemuck::{Pod, Zeroable};

use crate::{
    Point, Scalar, Float,
    named_scalar::{HasX, HasY, HasZ, HasW},
};


/// An `N`-dimensional vector with scalar type `T`. This represents
/// a *displacement* in space.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Vector<T: Scalar, const N: usize>(pub(crate) [T; N]);

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

/// A 2-dimensional vector with scalar type `f64`.
pub type Vec2d = Vec2<f64>;
/// A 3-dimensional vector with scalar type `f64`.
pub type Vec3d = Vec3<f64>;
/// A 4-dimensional vector with scalar type `f64`.
pub type Vec4d = Vec4<f64>;


/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Scalar + Zeroable, const N: usize> Zeroable for Vector<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Scalar + Pod, const N: usize> Pod for Vector<T, N> {}

impl<T: Scalar, const N: usize> Vector<T, N> {
    /// Returns the zero vector.
    pub fn zero() -> Self {
        Self([(); N].map(|_| T::zero()))
    }

    /// Returns `true` if this vector is the zero vector (all components 0).
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|c| c.is_zero())
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
        T: Float,
    {
        self.length2().sqrt()
    }

    /// Returns a normalized version of this vector (i.e. a vector with the same
    /// direction, but length 1). Also see [`normalize`][Self::normalize].
    #[must_use = "to normalize in-place, use `Vector::normalize`, not `normalized`"]
    pub fn normalized(mut self) -> Self
    where
        T: Float,
    {
        self.normalize();
        self
    }

    /// Normalizes the vector *in place* (i.e. maintain the direction but make
    /// it length 1). Also see [`normalized`][Self::normalized].
    pub fn normalize(&mut self)
    where
        T: Float,
    {
        *self = *self / self.length();
    }

    /// Returns the average of all given vectors or `None` if the given iterator
    /// is empty.
    ///
    /// ```
    /// use lina::{Vector, vec2};
    ///
    /// let avg = Vector::average([vec2(0.0, 8.0), vec2(1.0, 6.0)]);
    /// assert_eq!(avg, Some(vec2(0.5, 7.0)));
    /// ```
    pub fn average(vectors: impl IntoIterator<Item = Self>) -> Option<Self> {
        let mut it = vectors.into_iter();
        let mut total = it.next()?;
        let mut count = T::one();
        for v in it {
            total += v;
            count = count + T::one();
        }

        Some(total / count)
    }

    shared_methods!(Vector, "vector", "vec2", "vec3");
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

impl<T: Scalar, const N: usize> ops::Add<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip_map(rhs, |l, r| l + r)
    }
}

impl<T: Scalar, const N: usize> ops::AddAssign<Vector<T, N>> for Vector<T, N> {
    fn add_assign(&mut self, rhs: Vector<T, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T: Scalar, const N: usize> ops::Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        self.zip_map(rhs, |l, r| l - r)
    }
}

impl<T: Scalar, const N: usize> ops::SubAssign<Vector<T, N>> for Vector<T, N> {
    fn sub_assign(&mut self, rhs: Vector<T, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T: Scalar + ops::Neg, const N: usize> ops::Neg for Vector<T, N>
where
    <T as ops::Neg>::Output: Scalar,
{
    type Output = Vector<<T as ops::Neg>::Output, N>;
    fn neg(self) -> Self::Output {
        self.map(|c| -c)
    }
}

/// Scalar multipliation: `vector * scalar`.
impl<T: Scalar, const N: usize> ops::Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn mul(self, rhs: T) -> Self::Output {
        self.map(|c| c * rhs.clone())
    }
}

// Scalar multiplication: `scalar * vector`. Unfortunately, due to Rust's orphan
// rules, this cannot be implemented generically. So we just implement it for
// core primitive types.
macro_rules! impl_scalar_mul {
    ($($ty:ident),*) => {
        $(
            impl<const N: usize> ops::Mul<Vector<$ty, N>> for $ty {
                type Output = Vector<$ty, N>;
                fn mul(self, rhs: Vector<$ty, N>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(f32, f64, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

/// Scalar multipliation: `vector *= scalar`.
impl<T: Scalar, const N: usize> ops::MulAssign<T> for Vector<T, N> {
    fn mul_assign(&mut self, rhs: T) {
        for c in &mut self.0 {
            *c *= rhs.clone();
        }
    }
}

/// Scalar division: `vector / scalar`.
impl<T: Scalar, const N: usize> ops::Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn div(self, rhs: T) -> Self::Output {
        self.map(|c| c / rhs.clone())
    }
}

/// Scalar division: `vector /= scalar`.
impl<T: Scalar, const N: usize> ops::DivAssign<T> for Vector<T, N> {
    fn div_assign(&mut self, rhs: T) {
        for c in &mut self.0 {
            *c /= rhs;
        }
    }
}

impl<T: Scalar, const N: usize> std::iter::Sum<Self> for Vector<T, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}
