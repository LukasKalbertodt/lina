use bytemuck::{Pod, Zeroable};

use crate::{Vector, Scalar, util::zip_map};


/// A point in `N`-dimensional space with scalar type `T`. It represents
/// a *location* in space.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Point<T: Scalar, const N: usize>(pub(crate) [T; N]);

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
unsafe impl<T: Scalar + Zeroable, const N: usize> Zeroable for Point<T, N> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Scalar + Pod, const N: usize> Pod for Point<T, N> {}

impl<T: Scalar, const N: usize> Point<T, N> {
    /// Returns a point with all coordinates being zero (representing the origin).
    pub fn origin() -> Self {
        Self([(); N].map(|_| T::zero()))
    }

    /// Converts this point into a vector without changing the component values.
    /// Semantically equivalent to `self - Point::origin()`. Please think twice
    /// before using this method as it blindly changes the semantics of your
    /// value.
    pub fn to_vec(self) -> Vector<T, N> {
        Vector(self.0)
    }

    /// Returns the centroid ("average") of all given points or `None` if the
    /// given iterator is empty.
    ///
    /// ```
    /// use lina::{Point, point2};
    ///
    /// let centroid = Point::centroid([point2(0.0, 8.0), point2(1.0, 6.0)]);
    /// assert_eq!(centroid, Some(point2(0.5, 7.0)));
    /// ```
    pub fn centroid(points: impl IntoIterator<Item = Self>) -> Option<Self> {
        let mut it = points.into_iter();
        let mut total_displacement = it.next()?.to_vec();
        let mut count = T::one();
        for p in it {
            total_displacement = total_displacement + p.to_vec();
            count = count + T::one();
        }

        Some((total_displacement / count).to_point())
    }

    shared_methods!(Point, "point", "point3");
}

impl<T: Scalar> Point<T, 2> {
    shared_methods2!(Point, "point");
}

impl<T: Scalar> Point<T, 3> {
    shared_methods3!(Point, "point");
}


shared_impls!(Point, "point", "Point");

/// Shorthand for `Point2::new(...)`.
pub fn point2<T: Scalar>(x: T, y: T) -> Point2<T> {
    Point2::new(x, y)
}

/// Shorthand for `Point3::new(...)`.
pub fn point3<T: Scalar>(x: T, y: T, z: T) -> Point3<T> {
    Point3::new(x, y, z)
}

impl<T: Scalar, const N: usize> ops::Add<Vector<T, N>> for Point<T, N> {
    type Output = Point<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        Point(zip_map(self.0, rhs.0, |l, r| l + r))
    }
}

impl<T: Scalar, const N: usize> ops::AddAssign<Vector<T, N>> for Point<T, N> {
    fn add_assign(&mut self, rhs: Vector<T, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T: Scalar, const N: usize> ops::Sub<Vector<T, N>> for Point<T, N> {
    type Output = Point<T, N>;
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        Point(zip_map(self.0, rhs.0, |l, r| l - r))
    }
}

impl<T: Scalar, const N: usize> ops::SubAssign<Vector<T, N>> for Point<T, N> {
    fn sub_assign(&mut self, rhs: Vector<T, N>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T: Scalar, const N: usize> ops::Sub<Point<T, N>> for Point<T, N> {
    type Output = Vector<T, N>;
    fn sub(self, rhs: Point<T, N>) -> Self::Output {
        Vector(zip_map(self.0, rhs.0, |l, r| l - r))
    }
}
