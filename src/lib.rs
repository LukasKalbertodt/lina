use std::{
    fmt::Debug,
    ops::{AddAssign, SubAssign, MulAssign, DivAssign},
};
use num_traits::Num;

mod vec;
mod util;
pub mod named_scalar;

pub use self::{
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
pub trait Scalar: Num + Clone + Copy + Debug + AddAssign + SubAssign + MulAssign + DivAssign {}

impl<T> Scalar for T
where
    T: Num + Clone + Copy + Debug + AddAssign + SubAssign + MulAssign + DivAssign,
{}

/// A scalar that approximates the real numbers.
///
/// This is similar to [`Scalar`] as it defines coarse requirements for using
/// functions of this library. It is used whenever `Scalar` is not sufficient,
/// which is basically whenever a function does not make sense for integers.
/// This trait is implemented for at least `f32` and `f64`.
pub trait Real: Scalar + num_traits::real::Real {}

impl<T> Real for T
where
    T: Scalar + num_traits::real::Real,
{}
