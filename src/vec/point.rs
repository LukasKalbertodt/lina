use bytemuck::{Pod, Zeroable};

use crate::num::Zero;


/// A point in `N`-dimensional space with scalar type `T`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Point<T, const N: usize>([T; N]);

/// A point in 2-dimensional space.
pub type Point2<T> = Point<T, 2>;
/// A point in 3-dimensional space.
pub type Point3<T> = Point<T, 3>;

/// A point in 2-dimensional space with scalar type `f32`.
pub type Point2f = Point2<f32>;
/// A point in 3-dimensional space with scalar type `f32`.
pub type Point3f = Point3<f32>;

/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Zeroable, const N: usize> Zeroable for Point<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Pod, const N: usize> Pod for Point<T, N> {}

impl<T, const N: usize> Point<T, N> {
    /// Returns a point with all coordinates being zero (representing the origin).
    pub fn origin() -> Self
    where
        T: Zero,
    {
        Self([(); N].map(|_| T::zero()))
    }

    shared_methods!(Point, "point");
}

impl<T> Point<T, 2> {
    shared_methods2!(Point, "point");
}

impl<T> Point<T, 3> {
    shared_methods3!(Point, "point");
}


shared_impls!(Point, "point", "Point");

/// Shorthand for `Point2::new(...)`.
pub fn point2<T>(x: T, y: T) -> Point2<T> {
    Point2::new(x, y)
}

/// Shorthand for `Point3::new(...)`.
pub fn point3<T>(x: T, y: T, z: T) -> Point3<T> {
    Point3::new(x, y, z)
}
