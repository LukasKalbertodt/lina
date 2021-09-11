use std::{
    fmt::Debug,
    ops::{AddAssign, SubAssign, MulAssign, DivAssign},
};
use bytemuck::Pod;
use num_traits::Num;

mod angle;
mod vec;
mod util;
pub mod named_scalar;

pub use self::{
    angle::{Angle, Degrees, Radians},
    vec::{
        point::{Point, Point2, Point2f, Point3, Point3f, point2, point3},
        vector::{Vector, Vec2, Vec2f, Vec3, Vec3f, Vec4, Vec4f, vec2, vec3, vec4},
    },
};


/// A scalar type in the context of this library.
///
/// This is the bare minimum `lina` requires for most operations. It is somewhat
/// restricting (e.g. excluding big nums), but this library does not aim to be
/// super generic. There are better ones for that purpose. These requirements
/// make sense for games and similar applications.
///
/// This is implemented for at least these basic types:
///
/// - Floats: `f32` and `f64`
/// - Signed integers: `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
/// - Unsigned integers: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
pub trait Scalar:
    Num + Clone + Copy + Debug + AddAssign + SubAssign + MulAssign + DivAssign + Pod
{}

impl<T> Scalar for T
where
    T: Num + Clone + Copy + Debug + AddAssign + SubAssign + MulAssign + DivAssign + Pod,
{}

/// A floating point scalar.
///
/// This is similar to [`Scalar`] as it defines coarse requirements for using
/// functions of this library. It is used whenever `Scalar` is not sufficient,
/// which is basically whenever a function does not make sense for integers.
/// This trait is implemented for at least `f32` and `f64`.
pub trait Float: Scalar + num_traits::Float + num_traits::FloatConst {}

impl<T> Float for T
where
    T: Scalar + num_traits::Float + num_traits::FloatConst,
{}

/// Returns the [cross product][wiki] `a ⨯ b`, a vector perpendicular to both
/// input vectors.
///
/// ```
/// use lina::{cross, vec3};
///
/// assert_eq!(
///     cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)),
///     vec3(0.0, 0.0, 1.0),
/// );
/// assert_eq!(
///     cross(vec3(2.0, 0.0, 2.0), vec3(2.0, 0.0, -2.0)),
///     vec3(0.0, 8.0, 0.0),
/// );
/// ```
///
/// [wiki]: https://en.wikipedia.org/wiki/Cross_product
pub fn cross<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Returns the [dot product][wiki] `a · b`, a scalar value.
///
/// The dot product is equal to the product of the vectors lengths/magnitudes
/// and the cosine of the angle between the two vectors. So if both input
/// vectors are normalized, the dot product is exactly `cos(a)` with `a` being
/// the angle between the two vectors.
///
/// Another way to think about the dot product is to imagine one vector being
/// projected onto the other one. The dot product is incredible useful in many
/// scenarios.
///
/// This function panics if `N = 0` as dot products of 0-dimensional vectors
/// make little sense.
///
/// ```
/// use lina::{dot, vec2};
///
/// assert_eq!(dot(vec2(0, 0), vec2(1, 1)), 0);     // dot product of zero vectors are 0
/// assert_eq!(dot(vec2(-2, 0), vec2(3, 0)), -6);   // product of lengths times cos(180°) = -1
/// assert_eq!(dot(vec2(8, 0), vec2(0, 5)), 0);     // angle is 90°, cos(90°) = 0
/// assert_eq!(dot(vec2(1, 1), vec2(4, 0)), 4);
/// ```
///
/// [wiki]: https://en.wikipedia.org/wiki/Dot_product
pub fn dot<T: Scalar, const N: usize>(a: Vector<T, N>, b: Vector<T, N>) -> T {
    if N == 0 {
        panic!("the dot product of 0-dimensional vectors is not useful");
    }

    let mut out = a[0] * b[0];
    for i in 1..N {
        out += a[i] * b[i];
    }
    out
}

/// The [`atan2` function](https://en.wikipedia.org/wiki/Atan2).
///
/// This returns the angle between the positive x axis and the vector `[x, y]`
/// (mind the switched order of the function arguments).
pub fn atan2<T: Float>(y: T, x: T) -> Radians<T> {
    Radians(T::atan2(y, x))
}

/// Returns the angle between the two given vectors. Returns garbage if either
/// vector has length 0.
///
/// If you already know the vectors are normalized, it's faster to manually
/// calculate `Radians::acos(dot(a, b))`, as this skips calculating the
/// vectors' lengths.
///
///
/// ```
/// use lina::{angle_between, vec2, Radians};
/// use std::f32::consts::PI;
///
/// assert_eq!(angle_between(vec2(1.0, 0.0), vec2(3.0, 0.0)), Radians(0.0));
/// assert_eq!(angle_between(vec2(-2.0, 0.0), vec2(3.0, 0.0)), Radians(PI));       // 180°
/// assert_eq!(angle_between(vec2(0.2, 0.0), vec2(0.0, 7.3)), Radians(PI / 2.0));  // 90°
/// ```
pub fn angle_between<T: Float, const N: usize>(a: Vector<T, N>, b: Vector<T, N>) -> Radians<T> {
    Radians::acos(dot(a, b) / (a.length() * b.length()))
}
