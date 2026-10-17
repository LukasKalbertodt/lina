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
//! - Most types have an `as_bytes` method to pass them to graphics APIs.
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
//! relevant functions exist, as one is almost never concerned with anything
//! with more than 4 dimensions. So the only relevant disadvantage of this is
//! that the docs look less nice, as there are repetitions of the same
//! function.
//!
//! One further consequence of this is the choice that the const parameters of
//! `HcPoint` and `HcMatrix` don't reflect the number of values/rows/columns,
//! but the dimension of the space in which the point lives/which the
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

#[cfg(test)]
mod tests;

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
/// projected onto the other one. The dot product is incredibly useful in many
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
/// This is precise for all angles, including ones very close to 0 or π
/// (unlike `acos(cos θ)`, the obvious formula).
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

    // `acos(cos θ)` is very imprecise for angles close to 0 or π; same goes for
    // `asin(sin θ)` for angles near π/2. So instead, we calculate
    // `|a| |b| sin θ` and `|a| |b| cos θ` independently and pass both to
    // `atan2`.
    //
    // `|a| |b| cos θ` is simply `dot(a, b)` of course. `|a| |b| sin θ` is the
    // magnitude of the wedge product `a ∧ b` (the generalization of the cross
    // product to all dimensions), whose components are all `aᵢ bⱼ - aⱼ bᵢ` with
    // `i < j`.
    fn wedge2_and_dot<T: Float, const N: usize, S: Space>(
        a: Vector<T, N, S>,
        b: Vector<T, N, S>,
    ) -> (T, T) {
        let mut wedge2 = T::zero();
        for i in 0..N {
            for j in (i + 1)..N {
                let m = a[i] * b[j] - a[j] * b[i];
                wedge2 += m * m;
            }
        }
        (wedge2, dot(a, b))
    }

    let (mut wedge2, mut dot_ab) = wedge2_and_dot(a, b);

    // `wedge2 + dot_ab²` equals `|a|² |b|²`. If that is extremely small or
    // large (or NaN), the squares above lost precision due to underflow or
    // overflowed. The angle doesn't depend on the lengths, so in that rare
    // case, we scale both vectors to have a largest component of 1 and try
    // again. Checking this before calculating the wedge product would make
    // the common case slower.
    let ab2 = wedge2 + dot_ab * dot_ab;
    let min = T::min_positive_value() / (T::epsilon() * T::epsilon());
    if !(ab2 >= min && ab2 <= T::max_value()) {
        let scaled = |v: Vector<T, N, S>| v / v.iter().fold(T::zero(), |max, c| max.max(c.abs()));
        (wedge2, dot_ab) = wedge2_and_dot(scaled(a), scaled(b));
    }

    atan2(wedge2.sqrt(), dot_ab)
}

/// Clamps `val` into the given range `min..=max`.
///
/// The trait bound *should* technically be `Ord`, but that's inconvenient when
/// dealing with floats. Panics if `min > max` or if `min` or `max` is NaN. If
/// `val` is NaN, this panics in debug mode and returns NaN in release mode.
///
/// ```
/// use lina::clamp;
///
/// assert_eq!(clamp(0.7, 0.0..=1.0), 0.7);
/// assert_eq!(clamp(-3, -2..=2), -2);
/// assert_eq!(clamp(3, -2..=2), 2);
/// assert_eq!(clamp(1.5, 0.3..=0.3), 0.3);
/// ```
pub fn clamp<T: PartialOrd>(val: T, range: RangeInclusive<T>) -> T {
    let (min, max) = range.into_inner();

    // The range is usually constant, or at least the same for all iterations
    // of a loop, so these checks are optimized away or moved out of the loop.
    assert!(
        min.partial_cmp(&max).is_some(),
        "non-comparable value (NaN?) in range passed to `clamp`",
    );
    assert!(min <= max, "'min' is larger than 'max'");

    // This, however, would be checked for every value and prevents
    // auto-vectorization of loops, so we only check it in debug mode.
    debug_assert!(
        val.partial_cmp(&min).is_some(),
        "non-comparable value (NaN?) passed as 'val' to `clamp`",
    );

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
/// The vectors must also not point in opposite directions, as then, there
/// are infinitely many shortest paths between them and the result is not
/// defined (this is checked with a `debug_assert`). For vectors pointing in
/// *almost* opposite directions, the result can be imprecise.
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
    debug_assert!(
        angle < Radians::half_turn(),
        "`slerp` called with vectors pointing in opposite directions",
    );

    // The general formula `sin(x * t) / sin(x)` is problematic for very small
    // `x` as for x=0, we would divide by 0. In math world, all is hunky dory:
    // in the limit x->0, the value approaches `t`. But with floats, this
    // breaks.
    //
    // Luckily, in a fairly large region around x=0, floats do produce the
    // correct value `t`. By checking for `f32` and `f64`, the machine epsilon
    // is a good threshold. The general formula already yields exactly `t` for
    // `T::epsilon()`. So while the branch is unfortunate, this actually
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
///
/// `target` must have unit length (i.e. be normalized)! Otherwise, the result
/// is scaled by `|target|²` and thus wrong.
///
/// ```
/// use lina::{project_onto, vec2};
///
/// assert_eq!(project_onto(vec2(3.0, 4.0), vec2(1.0, 0.0)), vec2(3.0, 0.0));
/// assert_eq!(project_onto(vec2(3.0, 4.0), vec2(0.0, -1.0)), vec2(0.0, 4.0));
/// ```
pub fn project_onto<T: Float, const N: usize, S: Space>(
    v: impl Into<Vector<T, N, S>>,
    target: impl Into<Vector<T, N, S>>,
) -> Vector<T, N, S> {
    let target = target.into();
    target * dot(v.into(), target)
}

/// Reflects `incoming` off a surface with the given `normal`, returning
/// `incoming - 2 · (incoming · normal) · normal`.
///
/// When `incoming` represents the direction of incoming light, then this
/// returns `outgoing` representing the direction of the light being reflected.
///
/// ```
/// use lina::{Dir3f, Vec3f};
///
/// let normal = <Dir3f>::unit_z();
/// let incoming = Vec3f::new(1.0, 1.5, -1.0);
/// let reflected = lina::reflect(incoming, normal);
/// lina::assert_approx_eq!(ulps <= 4; reflected, Vec3f::new(1.0, 1.5, 1.0));
/// ```
pub fn reflect<T: Float, const N: usize, S: Space>(
    incoming: Vector<T, N, S>,
    normal: Dir<T, N , S>,
) -> Vector<T, N, S> {
    incoming - normal.to_unit_vec() * dot(incoming, normal) * T::two()
}

/// Linearly remaps `x` from `range` to `0..=1`.
///
/// This is just `(x - min) / (max - min)` clamped to `0..=1`. Also see
/// [`smoothstep`]. If `min == max`, this is a step function returning 0 for
/// `x < min` and 1 otherwise.
///
/// ```text
/// 1.0 ┤                         ······
///     │                       ·
///     │                     ·
///     │                   ·
///     │                 ·
/// 0.0 ┼ ···············
///     └───────────────┬─────────┬─────────→ x
///                    min       max
/// ```
///
/// ```
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep(-2.8, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 2.9, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 3.0, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 3.1, 3.0..7.0), 0.025);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 5.0, 3.0..7.0), 0.5);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 6.0, 3.0..7.0), 0.75);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep( 7.0, 3.0..7.0), 1.0);
/// lina::assert_approx_eq!(ulps <= 8; lina::linearstep(98.3, 3.0..7.0), 1.0);
///
/// assert_eq!(lina::linearstep(2.9, 3.0..3.0), 0.0);
/// assert_eq!(lina::linearstep(3.0, 3.0..3.0), 1.0);
/// assert_eq!(lina::linearstep(3.1, 3.0..3.0), 1.0);
/// ```
///
pub fn linearstep<T: Float>(x: T, range: ops::Range<T>) -> T {
    // Without this special case, we would calculate `0 / 0` for `x == min`.
    // NaN is handled below, like for non-empty ranges.
    if range.start == range.end && !x.is_nan() {
        return if x < range.start { T::zero() } else { T::one() };
    }

    clamp((x - range.start) / (range.end - range.start), T::zero()..=T::one())
}


/// Smoothly remaps `x` from `range` to `0..=1`, via smooth Hermite
/// interpolation.
///
/// This is `t² · (3 - 2t)` where [`t = linearstep(x, range)`][linearstep]. See
/// <https://en.wikipedia.org/wiki/Smoothstep>. A little something like this:
///
/// ```text
/// 1.0 ┤                        ..······
///     │                      .˙
///     │                     ·
///     │                    ·
///     │                  .˙
/// 0.0 ┼ ···············˙˙
///     └───────────────┬──────────┬─────────→ x
///                    min        max
/// ```
///
/// ```
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep(-2.8, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 2.9, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 3.0, 3.0..7.0), 0.0);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 3.1, 3.0..7.0), 0.00184375);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 5.0, 3.0..7.0), 0.5);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 6.0, 3.0..7.0), 0.84375);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep( 7.0, 3.0..7.0), 1.0);
/// lina::assert_approx_eq!(ulps <= 20; lina::smoothstep(98.3, 3.0..7.0), 1.0);
///
/// assert_eq!(lina::smoothstep(2.9, 3.0..3.0), 0.0);
/// assert_eq!(lina::smoothstep(3.0, 3.0..3.0), 1.0);
/// ```
pub fn smoothstep<T: Float>(x: T, range: ops::Range<T>) -> T {
    let t = linearstep(x, range);
    t * t * (T::three() - T::two() * t)
}
