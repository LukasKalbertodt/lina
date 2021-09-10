use bytemuck::{Pod, Zeroable};

use crate::{Vector, num::Zero, util::zip_map};


/// A point in `N`-dimensional space with scalar type `T`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Point<T, const N: usize>(pub(crate) [T; N]);

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

    /// Converts this point into a vector without changing the component values.
    /// Semantically equivalent to `self - Point::origin()`. Please think twice
    /// before using this method as it blindly changes the semantics of your
    /// value.
    pub fn to_vec(self) -> Vector<T, N> {
        Vector(self.0)
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

impl<T: ops::Add<U>, U, const N: usize> ops::Add<Vector<U, N>> for Point<T, N> {
    type Output = Point<T::Output, N>;
    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        Point(zip_map(self.0, rhs.0, |l, r| l + r))
    }
}

impl<T: ops::AddAssign<U>, U, const N: usize> ops::AddAssign<Vector<U, N>> for Point<T, N> {
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T: ops::Sub<U>, U, const N: usize> ops::Sub<Vector<U, N>> for Point<T, N> {
    type Output = Point<T::Output, N>;
    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        Point(zip_map(self.0, rhs.0, |l, r| l - r))
    }
}

impl<T: ops::SubAssign<U>, U, const N: usize> ops::SubAssign<Vector<U, N>> for Point<T, N> {
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T: ops::Sub<U>, U, const N: usize> ops::Sub<Point<U, N>> for Point<T, N> {
    type Output = Vector<T::Output, N>;
    fn sub(self, rhs: Point<U, N>) -> Self::Output {
        Vector(zip_map(self.0, rhs.0, |l, r| l - r))
    }
}
