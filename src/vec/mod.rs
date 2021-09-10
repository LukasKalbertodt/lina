use std::ops::{Deref, DerefMut, Index, IndexMut};
use bytemuck::{Pod, Zeroable};

use crate::num::{One, Zero};


// ===============================================================================================
// ===== Stuff shared by `Vector` and `Point`
// ===============================================================================================

macro_rules! shared {
    ($ty:ident) => {
        impl<T, const N: usize> $ty<T, N> {
            pub fn map<R, F: FnMut(T) -> R>(self, f: F) -> $ty<R, N> {
                $ty(self.0.map(f))
            }

        }

        impl<T> $ty<T, 2> {
            pub fn new(x: T, y: T) -> Self {
                Self([x, y])
            }
        }

        impl<T> $ty<T, 3> {
            pub fn new(x: T, y: T, z: T) -> Self {
                Self([x, y, z])
            }
        }

        impl<T, const N: usize> Index<usize> for $ty<T, N> {
            type Output = T;
            fn index(&self, index: usize) -> &Self::Output {
                &self.0[index]
            }
        }

        impl<T, const N: usize> IndexMut<usize> for $ty<T, N> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.0[index]
            }
        }

        impl<T, const N: usize> From<[T; N]> for $ty<T, N> {
            fn from(src: [T; N]) -> Self {
                Self(src)
            }
        }
    };
}


// ===============================================================================================
// ===== Vector
// ===============================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Vector<T, const N: usize>([T; N]);

pub type Vec2<T> = Vector<T, 2>;
pub type Vec3<T> = Vector<T, 3>;
pub type Vec4<T> = Vector<T, 4>;

pub type Vec2f = Vec2<f32>;
pub type Vec3f = Vec3<f32>;
pub type Vec4f = Vec4<f32>;

/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Zeroable, const N: usize> Zeroable for Vector<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Pod, const N: usize> Pod for Vector<T, N> {}

impl<T, const N: usize> Vector<T, N> {
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self([(); N].map(|_| T::zero()))
    }
}

macro_rules! impl_unit_ctor {
    ($name:ident, $val:expr) => {
        pub fn $name() -> Self
        where
            T: Zero + One,
        {
            Self($val)
        }
    };
}

impl<T> Vector<T, 2> {
    impl_unit_ctor!(unit_x, [T::one(), T::zero()]);
    impl_unit_ctor!(unit_y, [T::zero(), T::one()]);
}

impl<T> Vector<T, 3> {
    impl_unit_ctor!(unit_x, [T::one(), T::zero(), T::zero()]);
    impl_unit_ctor!(unit_y, [T::zero(), T::one(), T::zero()]);
    impl_unit_ctor!(unit_z, [T::zero(), T::zero(), T::one()]);
}

impl<T> Vector<T, 4> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        Self([x, y, z, w])
    }

    impl_unit_ctor!(unit_x, [T::one(), T::zero(), T::zero(), T::zero()]);
    impl_unit_ctor!(unit_y, [T::zero(), T::one(), T::zero(), T::zero()]);
    impl_unit_ctor!(unit_z, [T::zero(), T::zero(), T::one(), T::zero()]);
    impl_unit_ctor!(unit_w, [T::zero(), T::zero(), T::zero(), T::one()]);
}

shared!(Vector);

/// Shorthand for `Vec2::new(...)`.
pub fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vec2::new(x, y)
}

/// Shorthand for `Vec3::new(...)`.
pub fn vec3<T>(x: T, y: T) -> Vec2<T> {
    Vec2::new(x, y)
}

/// Shorthand for `Vec4::new(...)`.
pub fn vec4<T>(x: T, y: T) -> Vec2<T> {
    Vec2::new(x, y)
}


// ===============================================================================================
// ===== Point
// ===============================================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Point<T, const N: usize>([T; N]);

pub type Point2<T> = Point<T, 2>;
pub type Point3<T> = Point<T, 3>;

pub type Point2f = Point2<f32>;
pub type Point3f = Point3<f32>;

/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Zeroable, const N: usize> Zeroable for Point<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Pod, const N: usize> Pod for Point<T, N> {}

impl<T, const N: usize> Point<T, N> {
    pub fn origin() -> Self
    where
        T: Zero,
    {
        Self([(); N].map(|_| T::zero()))
    }
}

impl<T> Point<T, 2> {
}

impl<T> Point<T, 3> {
}


shared!(Point);

/// Shorthand for `Point2::new(...)`.
pub fn point2<T>(x: T, y: T) -> Point2<T> {
    Point2::new(x, y)
}

/// Shorthand for `Point3::new(...)`.
pub fn point3<T>(x: T, y: T) -> Point2<T> {
    Point2::new(x, y)
}


// ===============================================================================================
// ===== Vector "views" for x, y and z field accessors
// ===============================================================================================

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
