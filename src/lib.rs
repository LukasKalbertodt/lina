//! Linear algebra library with a focus on computer graphics, heavily using
//! strong typing and const generics (while still allowing access via `.x`,
//! `.y`, `.z` and `.w`).
//!
//! `lina` leans heavily into strong typing by distinguishing points and
//! vectors, Cartesian and homogeneous coordinates, and even elements from
//! different semantic spaces. See [these docs][docs::strong_typing] for more
//! information.
//!
//!
//! # Quick start / overview
//!
//! - **Locations and displacements**
//!     - [`Point`] represents a location.
//!     - [`Vector`] represents a displacement.
//!     - [`HcPoint`] represents a point in homogeneous coordinates.
//!     - [`Dir`] represents a direction via unit vector.
//!     - Use [`SphericalPos`] and [`SphericalDir`] for spherical coordinates.
//! - **Transformations**
//!     - [`Matrix`] represents a linear transformation.
//!     - [`HcMatrix`] represents a potentially non-linear transformation in
//!       homogeneous coordinates.
//!     - Use [`transform`] to get common transformation matrices.
//!     - Use `*` or `and_then` to combine two matrices.
//!     - Use `*` or `transform` to transform a vector or point with a matrix.
//! - Operators are overloaded as you would expect.
//! - Many types have a [`Space`] parameter to use strong typing.
//!     - Use `in_space` or `with_spaces` methods to cast/reinterpret the space parameter.
//!     - Recommendation: make type aliases for specific types used in your app,
//!       e.g. `type HelioPoint = Point<f64, 3, HelioSpace>`.
//! - Most types have a `to_bytes` method to pass them to graphic APIs.
//! - Other features:
//!     - Strongly typed angles: [`Degrees`] and [`Radians`]
//!     - Useful functions: [`atan2`], [`clamp`], [`lerp`], [`slerp`], [`cross`], [`dot`], ...
//!     - [`ApproxEq`] for approximate float equality
//!
//! This example shows some basic usage:
//!
//! ```
//! use lina::{point3, vec3, transform, Degrees, Vector};
//!
//! // Basic vector/point usage
//! let player_pos = point3(4.0, 5.0, 1.8);
//! let fox_pos = point3(10.0, 3.0, 0.5);
//! let view_direction = vec3(1.3, 0.2, 0.0).normalized();
//! let speed = 1.5;
//! let new_player_pos = player_pos + view_direction * speed;
//!
//! println!("{:.2}m still to go!", player_pos.distance_from(fox_pos));
//!
//! // Create and compose transformation matrices
//! let view_matrix = transform::look_into(new_player_pos, view_direction, Vector::unit_z());
//! let proj_matrix = transform::perspective(
//!     Degrees(90.0),
//!     16.0 / 9.0,
//!     0.1..=f32::INFINITY,
//!     1.0..=0.0,
//! );
//! let view_proj = view_matrix.and_then(proj_matrix); // or `proj_matrix * view_matrix`
//!
//! // Transform points with matrices
//! let fox_on_screen = view_proj.transform(fox_pos);  // or `view_proj * fox_pos`
//! ```
//!
//! ## Const generics limitations
//!
//! To express the signature of some functions, `feature(generic_const_exprs)`
//! is required. Think of [`Vector::extend`] which returns a `Vector<T, N + 1>`.
//! The `+ 1` is the problem here as this is currently not yet allowed on
//! stable Rust. Most of these functions are related to [`HcPoint`] or
//! [`HcMatrix`].
//!
//! Not only is that feature not stabilized yet, it is also very unfinished and
//! broken. So unfortunately, `lina` cannot use it. Instead, these functions
//! are implemented for a small number of fixed dimensions via macro. But worry
//! not! For one, these are just a few functions without which `lina` can still
//! be used without a problem. Further, for the 3D graphics use case, all
//! relevant functions exist, as one is almost never converned with anything
//! with more than 4 dimensions. So the only relevant disadvantage of this is
//! that the docs look less nice, as there are repetitions of the same
//! function.
//!
//! One further consequence of this is the choice that the const parameters of
//! `HcPoint` and `HcMatrix` don't reflect the number of values/rows/columns,
//! but the the dimension of the space in which the point lives/which the
//! transformation transforms. E.g. `HcPoint<3>` represents a 3D point, by
//! storing 4 numbers.
//!

use std::{
    fmt::Debug,
    ops::{self, AddAssign, SubAssign, MulAssign, DivAssign, RangeInclusive},
};
use bytemuck::Pod;
use num_traits::Num;

mod angle;
mod approx;
mod dir;
mod mat;
mod space;
mod spherical;
mod util;
mod vec;
pub mod docs;
pub mod named_scalar;
pub mod transform;

pub use self::{
    angle::{Degrees, Radians},
    approx::ApproxEq,
    dir::{Dir, Dir2, Dir2f, Dir3, Dir3f},
    mat::{
        linear::{Matrix, Mat2, Mat2f, Mat3, Mat3f},
        hc::{HcMatrix, HcMat2, HcMat2f, HcMat3, HcMat3f},
    },
    space::{Space, ModelSpace, WorldSpace, ViewSpace, ProjSpace},
    spherical::{SphericalDir, SphericalPos},
    vec::{
        hc::{HcPoint, HcPoint2, HcPoint2f, HcPoint3, HcPoint3f},
        point::{Point, Point2, Point2f, Point3, Point3f, point2, point3},
        vector::{Vector, Vec2, Vec2f, Vec3, Vec3f, vec2, vec3},
    },
};

/// Helper utilities for matrices.
pub mod matrix {
    pub use crate::mat::{
        linear::{Col, Row},
        hc::{HcCol, HcRow},
    };
}


/// A scalar type in the context of this library.
///
/// This is the bare minimum `lina` requires for most operations. It is somewhat
/// restricting (e.g. excluding big nums), but this library does not aim to be
/// super generic. There are better ones for that purpose. These requirements
/// make sense for games and similar applications.
///
/// This is implemented for at least these types:
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
pub trait Float: Scalar + num_traits::Float + num_traits::FloatConst + approx::ApproxEq {
    fn two() -> Self {
        Self::one() + Self::one()
    }
    fn three() -> Self {
        Self::one() + Self::one() + Self::one()
    }
    fn four() -> Self {
        Self::one() + Self::one() + Self::one() + Self::one()
    }
}

impl<T> Float for T
where
    T: Scalar + num_traits::Float + num_traits::FloatConst + approx::ApproxEq,
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
pub fn cross<T: Scalar, S: Space>(
    a: impl Into<Vec3<T, S>>,
    b: impl Into<Vec3<T, S>>,
) -> Vec3<T, S> {
    let a = a.into();
    let b = b.into();
    Vec3::new(
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
pub fn dot<T: Scalar, const N: usize, S: Space>(
    a: impl Into<Vector<T, N, S>>,
    b: impl Into<Vector<T, N, S>>,
) -> T {
    assert!(N != 0, "the dot product of 0-dimensional vectors is not useful");

    let a = a.into();
    let b = b.into();

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
pub fn angle_between<T: Float, const N: usize, S: Space>(
    a: impl Into<Vector<T, N, S>>,
    b: impl Into<Vector<T, N, S>>,
) -> Radians<T> {
    let a = a.into();
    let b = b.into();

    debug_assert!(!a.is_zero());
    debug_assert!(!b.is_zero());

    let mut cos_angle = dot(a, b) / (a.length() * b.length());

    // Unfortunately, sometimes, due to float precision, `cos_angle` is
    // sometimes slightly above 1. And taking the `acos` of it then would
    // result in NaN.
    if cos_angle > T::one() {
        cos_angle = T::one();
    }
    Radians::acos(cos_angle)
}

/// Clamps `val` into the given range `min..=max`.
///
/// The trait bound *should* technically be `Ord`, but that's inconvenient when
/// dealing with floats. Panics when passed a NaN.
pub fn clamp<T: PartialOrd>(val: T, range: RangeInclusive<T>) -> T {
    let (min, max) = range.into_inner();
    assert!(
        min.partial_cmp(&max).is_some(),
        "non-comparable value (NaN?) in range passed to `clamp`",
    );
    assert!(
        val.partial_cmp(&min).is_some(),
        "non-comparable value (NaN?) passed as 'val' to `clamp`",
    );
    assert!(min < max, "'min' is larger than 'max'");

    match () {
        () if val < min => min,
        () if val > max => max,
        _ => val,
    }
}

/// Linearly interpolates between `a` and `b` with the given `factor`.
/// `factor = 0` is 100% `a`, `factor = 1` is 100% `b`.
///
/// If `factor` is outside of the range `0..=1`, the result might not make
/// sense. It is simply following the formula `(1 - factor) * a + factor * b`.
///
/// ```
/// use lina::{lerp, vec2};
///
/// assert_eq!(lerp(10.0, 20.0, 0.6), 16.0);
/// assert_eq!(lerp(vec2(10.0, -5.0), vec2(12.0, 5.0), 0.2), vec2(10.4, -3.0));
/// ```
pub fn lerp<F: Float, T>(a: T, b: T, factor: F) -> T
where
    T: ops::Mul<F, Output = T> + ops::Add<Output = T>,
{
    a * (F::one() - factor) + b * factor
}

/// *Spherical* linear interpolation between `a` and `b` with the given `factor`.
/// `factor = 0` is 100% `a`, `factor = 1` is 100% `b`.
///
/// This operation linearly interpolates the angle between the vectors, so to
/// speak. Or viewed differently: it linearly interpolates the sphere surface
/// path from one to the other vector. For more information, see here:
/// <https://en.wikipedia.org/wiki/Slerp>
///
/// The vectors must not be zero! They don't have to be normalized, but don't
/// ask me how to interpret the result if they don't have the same length. The
/// same is true for `factor`: usually only the range `0..=1` makes sense, but
/// this is not enforced. No idea if the results are useful.
///
/// ```
/// use lina::{slerp, vec3};
///
/// assert_eq!(
///     slerp(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 0.5),
///     vec3(0.7071067811865475, 0.7071067811865475, 0.0),  // sqrt(2) / 2
/// );
/// ```
pub fn slerp<T: Float, const N: usize, S: Space>(
    a: Vector<T, N, S>,
    b: Vector<T, N, S>,
    factor: T,
) -> Vector<T, N, S> {
    let angle = angle_between(a, b);

    // The general formula `sin(x * t) / sin(x)` is problematic for very small
    // `x` as for x=0, we would divide by 0. In math world, all is hunky dory:
    // in the limit x->0, the value approaches `t`. But with floats, this is
    // breaks.
    //
    // Luckily, in a fairly large region around x=0, floats do produce the
    // correct value `t`. By checking for `f32` and `f64`, the machine epsilon
    // is a good treshold. The general formula already yields exactly `t` for
    // `T::epsilon()`. So while the branch is unfortunate, but this actually
    // results in the correct value for all angles.
    let (factor_a, factor_b) = if angle.0 < T::epsilon() {
        (T::one() - factor, factor)
    } else {
        let sin_angle = angle.sin();
        (
            (angle * (T::one() - factor)).sin() / sin_angle,
            (angle * factor).sin() / sin_angle,
        )
    };

    a * factor_a + b * factor_b
}

/// Projects `v` onto `target`, returning `target · (v · target)`.
pub fn project_onto<T: Float, const N: usize, S: Space>(
    v: impl Into<Vector<T, N, S>>,
    target: impl Into<Vector<T, N, S>>,
) -> Vector<T, N, S> {
    let target = target.into();
    target * dot(v.into(), target)
}
