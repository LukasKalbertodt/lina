//! Helper types and traits to access specific coordinates of vectors of points
//! by name `x`, `y`, `z` and `w`.
//!
//! This only works with a small set of vectors/points with specific dimension.
//! Thus, it is not something you can rely on in generic contexts, but is
//! mainly for convenience and readable code.
//!
//! TODO: explain more.

use std::ops::{Deref, DerefMut};
use bytemuck::{Pod, Zeroable};

use crate::{Point, Vector};



/// Helper struct giving access to the individual components of a 2D vector or
/// point.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct View2<T> {
    pub x: T,
    pub y: T,
}

/// Helper struct giving access to the individual components of a 3D vector or
/// point.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct View3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Helper struct giving access to the individual components of a 4D vector or
/// point.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct View4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// `Zeroable` impls for "view" types are sound:
//
// - They are inhabited: structs plus bound `T: Zeroable`.
// - They only consists of `Zeroable` fields, thus zero bit pattern is fine.
unsafe impl<T: Zeroable> Zeroable for View2<T> {}
unsafe impl<T: Zeroable> Zeroable for View3<T> {}
unsafe impl<T: Zeroable> Zeroable for View4<T> {}

// `Pod` impls for "view" types are sound:
//
// - "The type must be inhabited": guaranteed by all being structs and the bound `T: Pod`.
// - "The type must not contain any padding bytes": this is true according to [1].
// - "The type needs to have all fields also be `Pod`": trivially true due to `T: Pod`.
// - "The type must allow any bit pattern": true based on the previous two facts.
// - "The type needs to be `repr(C)` or `repr(transparent)`": trivially true.
//
// [1] https://doc.rust-lang.org/reference/type-layout.html#reprc-structs
unsafe impl<T: Pod> Pod for View2<T> {}
unsafe impl<T: Pod> Pod for View3<T> {}
unsafe impl<T: Pod> Pod for View4<T> {}

// `Deref` and `DerefMut` impls to enable `.x` like field access.
macro_rules! impl_view_deref {
    ($ty:ident, $n:expr, $view_ty:ident) => {
        impl<T: Pod> Deref for $ty<T, $n> {
            type Target = $view_ty<T>;
            fn deref(&self) -> &Self::Target {
                bytemuck::cast_ref(self)
            }
        }
        impl<T: Pod> DerefMut for $ty<T, $n> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                bytemuck::cast_mut(self)
            }
        }
    };
}

impl_view_deref!(Vector, 2, View2);
impl_view_deref!(Vector, 3, View3);
impl_view_deref!(Vector, 4, View4);
impl_view_deref!(Point, 2, View2);
impl_view_deref!(Point, 3, View3);


/// Vectors or points that have an `x` component.
pub trait HasX {
    type Scalar;
    fn x(&self) -> &Self::Scalar;
    fn x_mut(&mut self) -> &mut Self::Scalar;
}

/// Vectors or points that have an `y` component.
pub trait HasY {
    type Scalar;
    fn y(&self) -> &Self::Scalar;
    fn y_mut(&mut self) -> &mut Self::Scalar;
}

/// Vectors or points that have an `z` component.
pub trait HasZ {
    type Scalar;
    fn z(&self) -> &Self::Scalar;
    fn z_mut(&mut self) -> &mut Self::Scalar;
}

/// Vectors or points that have an `w` component.
pub trait HasW {
    type Scalar;
    fn w(&self) -> &Self::Scalar;
    fn w_mut(&mut self) -> &mut Self::Scalar;
}

macro_rules! impl_has_axis {
    ($ty:ident, $d:expr, $trait:ident, $i:expr, $axis:ident, $axis_mut:ident) => {
        impl<T> $trait for $ty<T, $d> {
            type Scalar = T;
            fn $axis(&self) -> &Self::Scalar {
                &self[$i]
            }
            fn $axis_mut(&mut self) -> &mut Self::Scalar {
                &mut self[$i]
            }
        }
    };
}

impl_has_axis!(Vector, 2, HasX, 0, x, x_mut);
impl_has_axis!(Vector, 3, HasX, 0, x, x_mut);
impl_has_axis!(Vector, 4, HasX, 0, x, x_mut);
impl_has_axis!(Point, 2, HasX, 0, x, x_mut);
impl_has_axis!(Point, 3, HasX, 0, x, x_mut);

impl_has_axis!(Vector, 2, HasY, 1, y, y_mut);
impl_has_axis!(Vector, 3, HasY, 1, y, y_mut);
impl_has_axis!(Vector, 4, HasY, 1, y, y_mut);
impl_has_axis!(Point, 2, HasY, 1, y, y_mut);
impl_has_axis!(Point, 3, HasY, 1, y, y_mut);

impl_has_axis!(Vector, 3, HasZ, 2, z, z_mut);
impl_has_axis!(Vector, 4, HasZ, 2, z, z_mut);
impl_has_axis!(Point, 3, HasZ, 2, z, z_mut);

impl_has_axis!(Vector, 4, HasW, 3, w, w_mut);
