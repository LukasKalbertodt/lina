//! Yet another linear algebra library with a focus on computer graphics, thus
//! 2D, 3D and 4D vectors and matrices. TODO
//!
//!
//! # Viewing Pipeline
//!
//! In your 3D application, you somehow have to draw your scene to the screen.
//! There are two main approaches: ray-tracing and rasterization. The former is
//! generally still too expensive to do in real-time, thus rasterization is the
//! standard way to render games and other interactive 3D applications.
//!
//! Graphics APIs (like Vulkan, OpenGL, DirectX, WebGPU) basically require you
//! to transform every 3D point of your object into the so called *normalized
//! device coordinate system* (**NDC**). This is usually done in multiple
//! steps, going through a few different *coordinate systems* or *spaces* in
//! this order:
//!
//! - **Model space**: one object that has not been placed in your scene/world
//!   yet. For example, its center is usually around the origin (0, 0, 0) and
//!   is usually in upright position.
//!
//! - **World space**: this is the most intuitive space and where most of your
//!   game logic will happen. Here, all objects have been placed in the world by
//!   moving (translating), rotating and scaling them.
//!
//! - **View space**: an intermediate space where, by convention, the camera
//!   sits at the origin (0, 0, 0) and looks down the z axis (-z or +z, see
//!   below). This simplifies later calculations and makes some operations
//!   easier. The world has been translated and rotated, but not scaled or
//!   otherwise transformed. Angles from world space are preserved.
//!
//! - **NDC**: this is very close to screen space, with `x` and `y` describing
//!   the horizontal and vertical position on your screen (or rather, the
//!   application window), respectively. The `z` axis points into or out of your
//!   screen and is used for depth testing and such things.
//!
//! - **screen space**: 2D space with `x` and `y` being horizontal and vertical
//!   position in pixel units. Converting NDC to screen space is straight
//!   forward and is done internally by your graphics API.
//!
//! *Aside*: what about **Clip space?** This coordinate system is essentially
//! the same as NDC, but coordinates are still homogeneous (i.e. 4D) to simplify
//! clipping. See [this answer](https://gamedev.stackexchange.com/a/65798/85787)
//! for more details.
//!
//! Since the graphics APIs only start to interpret coordinates when you pass
//! them in NDC, everything before that is mostly up to you. The above spaces
//! are a good convention to stick to, though.
//!
//!
//! ## Transforming between spaces with matrices inside shaders
//!
//! In practice, you use matrices for almost all transformations from one space
//! to another. We do not discuss the *model → world* transformation here, as
//! this depends a lot on your application. The *world → view* and *view → NDC*
//! transformations are typically done like this:
//!
//! You place a virtual camera in your scene which has properties like a
//! position and the direction in which it looks. You also have "global"
//! properties like the field of view (FoV) you want to render with. From those
//! values, you create two matrices: the view matrix (world → view, via
//! [`transform::look_into`]) and the projection matrix (view → NDC, via
//! [`transform::perspective`]).
//!
//! You pass both of those matrices to your shader as uniform value or push
//! constant. Inside the vertex shader, you extend your 3D vertex position with
//! a `w` coordinate with value 1, giving you a 4D vector representing
//! homogeneous coordinates in 3D space. Next, multiple that 4D vector with the
//! view matrix, then with the projection matrix, giving you another 4D vector
//! which `w` component might not be 1. This is what you "return" from the
//! shader (e.g. assign to `gl_Position`). Your graphics API will then perform
//! the "perspective divide" automatically: divide `x`, `y` and `z` by `w`,
//! which are the final 3D NDC coordinates.
//!
//! It's also possible to pre-multiply both matrices in your application and
//! only pass the combined matrix to the shader. You can do that if you don't
//! perform any calculations in view space.
//!
//!
//! ## NDC differences between APIs
//!
//! Unfortunatley, the exact properties/requirements of NDC depend on the
//! graphic API you are using. The common properties are:
//!
//! - `x` and `y` are in range -1 to 1 (inclusive) and describe the horizontal
//!   and vertical position on the screen, respectively.
//! - The `+x` axis points from left to right.
//! - The `z` axis direction is not defined. It is used for depth-tests for
//!   example, but those can be configured for different comparisons (e.g. `<`
//!   vs. `>`). So this is up to the programmer.
//! - All points outside of the valid `x`, `y` or `z` ranges are clipped by
//!   default, i.e. basically removed.
//!
//! Differences concern the direction of the `y` axis and the `z` range.
//!
//! | API | +y is... | z range |
//! | --- | -------- | ------- |
//! | WebGPU, Direct3D & Metal | ... up (`[-1, -1]` is bottom left corner) | 0 to 1 |
//! | OpenGL | ... up (`[-1, -1]` is bottom left corner) | -1 to 1 |
//! | Vulkan | ... down (`[-1, -1]` is top left corner) | 0 to 1 |
//!
//! Since you are responsible for creating the point in NDC, your
//! view-projection matrix depends on the API you are using.
//!
//! The above values are the defaults for the respective APIs. Some APIs might
//! allow you to configure these things. For example, in OpenGL, the widely
//! supported [`ARB_clip_control`][gl-clip-control] allows you to change the +y
//! direction and z-range.
//!
//! [gl-clip-control]: https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_clip_control.txt
//!
//!
//! ## Choice of view space & handedness
//!
//! By convention, in view space, the camera sits at the origin, +x points to
//! the right, +y points up or down, +z points inside or outside the monitor.
//! In the majority of cases, +y points up in view space, so we will assume
//! that.
//!
//! That leaves the choice of z-direction. Usually, -z points inside the screen
//! and +z points out of the screen. This is called a *right-handed coordinate
//! system* because you can use your right hand's thumb (+x), index finger
//! (+y) and middle finger (+z) at right angles to one another to represent the
//! axis of this space. The alternative is the left-handed coordinate system
//! where +z points inside the screen and -z points out of it.
//!
//! What's important is that **the choice does not matter** as long as you use
//! an appropriate projection matrix in order to transform your points
//! correctly into the NDC of your graphics API. View and projection matrix
//! need to fit to one another and to your API's NDC. If you perform any
//! calculations in view space, you might merely need to know about the
//! handedness of your view space. But other than that, it's arbitrary.
//!
//! [`transform::look_into`] returns a view matrix that transforms into the
//! right-handed view space. Similarly, [`transform::perspective`] assumes a
//! right-handed view space.
//!
//! Left-handed versions of these functions are not offered in this library
//! because the choice is arbitrary and you can easily get a left-handed
//! version yourself. For the view matrix, just pass `-direction` as direction
//! to `look_into`. To get a projection matrix that works with a left-handed
//! view space, just flip the sign of your view space as a transformation
//! before the projection matrix. That means your projection matrix would be
//! `transform::perspective(...) * flip` where `flip` is this matrix (e.g. via
//! `Mat4::from_diagonal([1, 1, -1, 1])`:
//!
//! ```text
//! ⎡ 1  0  0  0 ⎤
//! ⎢ 0  1  0  0 ⎥
//! ⎢ 0  0 -1  0 ⎥
//! ⎣ 0  0  0  1 ⎦
//! ```
//!
//!


#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use std::{
    fmt::Debug,
    ops::{self, AddAssign, SubAssign, MulAssign, DivAssign},
};
use bytemuck::Pod;
use num_traits::Num;

mod angle;
mod mat;
mod vec;
mod util;
pub mod named_scalar;
pub mod transform;

pub use self::{
    angle::{Degrees, Radians},
    mat::{Matrix, Mat2, Mat2f, Mat2d, Mat3, Mat3f, Mat3d, Mat4, Mat4f, Mat4d},
    vec::{
        point::{Point, Point2, Point2f, Point2d, Point3, Point3f, Point3d, point2, point3},
        vector::{
            Vector, Vec2, Vec2f, Vec2d, Vec3, Vec3f, Vec3d, Vec4, Vec4f, Vec4d, vec2, vec3, vec4,
        },
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

/// Clamps `val` into the range `min..=max`.
///
/// The trait bound *should* technically be `Ord`, but that's inconvenient when
/// dealing with floats. When you pass a `NaN` you will get a strange result.
pub fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
    assert!(min < max);

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
