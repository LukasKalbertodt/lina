use std::{array, marker::PhantomData};
use bytemuck::{Pod, Zeroable};

use crate::{Vector, Scalar, Float, WorldSpace, Space, HcPoint};


/// A point in `N`-dimensional space with scalar type `T`. It represents
/// a *location* in space.
#[repr(transparent)]
pub struct Point<T: Scalar, const N: usize, S: Space = WorldSpace>(
    pub(crate) [T; N],
    PhantomData<S>,
);

/// A point in 2-dimensional space.
pub type Point2<T, S = WorldSpace> = Point<T, 2, S>;
/// A point in 3-dimensional space.
pub type Point3<T, S = WorldSpace> = Point<T, 3, S>;

/// A point in 2-dimensional space with scalar type `f32`.
pub type Point2f<S = WorldSpace> = Point2<f32, S>;
/// A point in 3-dimensional space with scalar type `f32`.
pub type Point3f<S = WorldSpace> = Point3<f32, S>;

/// A point in 2-dimensional space with scalar type `f64`.
pub type Point2d<S = WorldSpace> = Point2<f64, S>;
/// A point in 3-dimensional space with scalar type `f64`.
pub type Point3d<S = WorldSpace> = Point3<f64, S>;


impl<T: Scalar, const N: usize, S: Space> Point<T, N, S> {
    /// Returns a point with all coordinates being zero (representing the origin).
    pub fn origin() -> Self {
        std::array::from_fn(|_| T::zero()).into()
    }

    /// Returns the *squared* distance between `self` and `other`, i.e.
    /// `|self - other|²`. If you only need to compare two distances, this can
    /// be used as a faster alternative to [`distance_from`][Self::distance_from],
    /// since the sqrt function is continious.
    ///
    /// ```
    /// use lina::point2;
    ///
    /// let d = point2(1.0, 5.5).distance2_from(point2(4.0, 1.5));
    /// assert_eq!(d, 25.0);
    /// ```
    pub fn distance2_from(self, other: Self) -> T {
        (self - other).length2()
    }

    /// Returns the distance between `self` and `other`, i.e. `|self - other|`.
    ///
    /// If you only need to compare two distances, you may use the faster
    /// [`distance2_from`][Self::distance2_from].
    ///
    /// ```
    /// use lina::point2;
    ///
    /// let d = point2(1.0, 5.5).distance_from(point2(4.0, 1.5));
    /// assert_eq!(d, 5.0);
    /// ```
    pub fn distance_from(self, other: Self) -> T
    where
        T: Float,
    {
        (self - other).length()
    }

    /// Returns this point represented in homogeneous coordinates.
    ///
    /// Numerically, this simply extends this point with a 1 as weight value.
    pub fn to_hc_point(self) -> HcPoint<T, N, S> {
        self.into()
    }

    /// Converts this point into a vector without changing the component values.
    /// Semantically equivalent to `self - Point::origin()`. Please think twice
    /// before using this method as it blindly changes the semantics of your
    /// value.
    pub fn to_vec(self) -> Vector<T, N, S> {
        self.0.into()
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
            total_displacement += p.to_vec();
            count += T::one();
        }

        Some((total_displacement / count).to_point())
    }

    shared_methods!(Point, "point", "point2", "point3");
}

impl<T: Scalar, S: Space> Point<T, 2, S> {
    shared_methods2!(Point, "point");
}

impl<T: Scalar, S: Space> Point<T, 3, S> {
    shared_methods3!(Point, "point");
}


shared_impls!(Point, "point", "Point");

/// Shorthand for `Point2::new(...)`, but with fixed `S = Generic`.
pub fn point2<T: Scalar>(x: T, y: T) -> Point2<T> {
    Point2::new(x, y)
}

/// Shorthand for `Point3::new(...)`, but with fixed `S = Generic`.
pub fn point3<T: Scalar>(x: T, y: T, z: T) -> Point3<T> {
    Point3::new(x, y, z)
}


// =============================================================================================
// ===== Trait impls
// =============================================================================================

impl<T: Scalar + std::hash::Hash, const N: usize, S: Space> std::hash::Hash for Point<T, N, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
impl<T: Scalar, const N: usize, S: Space> PartialEq for Point<T, N, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<T: Scalar + Eq, const N: usize, S: Space> Eq for Point<T, N, S> {}

impl<T: Scalar, const N: usize, S: Space> Clone for Point<T, N, S> {
    fn clone(&self) -> Self {
        Self(self.0, self.1)
    }
}

impl<T: Scalar, const N: usize, S: Space> Copy for Point<T, N, S> {}

/// `[T; N] where T: Zeroable` implements `Zeroable` and this is just a newtype
/// wrapper around an array with `repr(transparent)`.
unsafe impl<T: Scalar + Zeroable, const N: usize, S: Space> Zeroable for Point<T, N, S> {}

/// The struct is marked as `repr(transparent)` so is guaranteed to have the
/// same layout as `[T; N]`. And `bytemuck` itself has an impl for arrays where
/// `T: Pod`.
unsafe impl<T: Scalar + Pod, const N: usize, S: Space> Pod for Point<T, N, S> {}


// =============================================================================================
// ===== Operator impls
// =============================================================================================

impl<T: Scalar, const N: usize, S: Space> ops::Add<Vector<T, N, S>> for Point<T, N, S> {
    type Output = Self;
    fn add(self, rhs: Vector<T, N, S>) -> Self::Output {
        array::from_fn(|i| self[i] + rhs[i]).into()
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::AddAssign<Vector<T, N, S>> for Point<T, N, S> {
    fn add_assign(&mut self, rhs: Vector<T, N, S>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::Sub<Vector<T, N, S>> for Point<T, N, S> {
    type Output = Self;
    fn sub(self, rhs: Vector<T, N, S>) -> Self::Output {
        array::from_fn(|i| self[i] - rhs[i]).into()
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::SubAssign<Vector<T, N, S>> for Point<T, N, S> {
    fn sub_assign(&mut self, rhs: Vector<T, N, S>) {
        for (lhs, rhs) in IntoIterator::into_iter(&mut self.0).zip(rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T: Scalar, const N: usize, S: Space> ops::Sub<Self> for Point<T, N, S> {
    type Output = Vector<T, N, S>;
    fn sub(self, rhs: Self) -> Self::Output {
        array::from_fn(|i| self[i] - rhs[i]).into()
    }
}
