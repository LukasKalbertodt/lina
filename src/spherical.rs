use std::{fmt, hash::Hash, ops, marker::PhantomData};

use crate::{Vec3, Point3, Float, Radians, Space, WorldSpace};


/// A 3D point described in spherical coordinates (theta θ, phi φ, radius r).
pub struct SphericalPos<T: Float, S: Space = WorldSpace> {
    /// Vertical angle θ: 0° points up (north pole), 180° points down (south pole),
    /// 90° is on the equator. Typical range: `0..=π`.
    ///
    /// When created from a point/vector, the angle is in `0..=π`.
    pub theta: Radians<T>,

    /// Horizontal angle φ: 0° points +x, 90° points +y, 180° points -y, 270°
    /// points -y. Typical range: `0..=2π`.
    ///
    /// When created from a point/vector, the angle is in `-π..=π`.
    pub phi: Radians<T>,

    /// The distance from the origin.
    pub r: T,

    _dummy: PhantomData<S>,
}

impl<T: Float, S: Space> SphericalPos<T, S> {
    pub fn new(theta: impl Into<Radians<T>>, phi: impl Into<Radians<T>>, r: T) -> Self {
        Self {
            theta: theta.into(),
            phi: phi.into(),
            r,
            _dummy: PhantomData,
        }
    }

    /// Convenience function for `Vec3::from(self)`.
    pub fn to_vec(self) -> Vec3<T, S> {
        self.into()
    }

    /// Convenience function for `Point3::from(self)`.
    pub fn to_point(self) -> Point3<T, S> {
        self.into()
    }

    pub fn without_radius(self) -> NormedSphericalPos<T, S> {
        self.into()
    }
}

impl<T: Float, S: Space> From<Point3<T, S>> for SphericalPos<T, S> {
    fn from(p: Point3<T, S>) -> Self {
        Self::from(p.to_vec())
    }
}

impl<T: Float, S: Space> From<Vec3<T, S>> for SphericalPos<T, S> {
    fn from(v: Vec3<T, S>) -> Self {
        let nv = v.normalized();
        Self {
            theta: Radians::acos(nv.z),
            phi: crate::atan2(nv.y, nv.x),
            r: v.length(),
            _dummy: PhantomData,
        }
    }
}

impl<T: Float, S: Space> From<SphericalPos<T, S>> for Point3<T, S> {
    fn from(src: SphericalPos<T, S>) -> Self {
        Vec3::from(src).to_point()
    }
}

impl<T: Float, S: Space> From<SphericalPos<T, S>> for Vec3<T, S> {
    fn from(src: SphericalPos<T, S>) -> Self {
        Vec3::new(
            src.r * src.phi.cos() * src.theta.sin(),
            src.r * src.phi.sin() * src.theta.sin(),
            src.r * src.theta.cos(),
        )
    }
}

impl<T: Float, S: Space> fmt::Debug for SphericalPos<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(θ: {:?}, φ: {:?}, r: {:?})", self.theta, self.phi, self.r)
    }
}

impl<T: Float, S: Space> Clone for SphericalPos<T, S> {
    fn clone(&self) -> Self {
        Self::new(self.theta, self.phi, self.r)
    }
}
impl<T: Float, S: Space> Copy for SphericalPos<T, S> {}
impl<T: Float, S: Space> PartialEq for SphericalPos<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.theta == other.theta && self.phi == other.phi && self.r == other.r
    }
}
impl<T: Float + Hash, S: Space> Hash for SphericalPos<T, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.theta.hash(state);
        self.phi.hash(state);
        self.r.hash(state);
    }
}



/// A point on the boundary of a unit sphere described in spherical coordinates
/// (without radius).
pub struct NormedSphericalPos<T: Float, S: Space = WorldSpace> {
    /// Vertical angle θ: 0° points up (north pole), 180° points down (south pole),
    /// 90° is on the equator. Typical range: `0..=π`.
    ///
    /// When created from a point/vector, the angle is in `0..=π`.
    pub theta: Radians<T>,

    /// Horizontal angle φ: 0° points +x, 90° points +y, 180° points -y, 270°
    /// points -y. Typical range: `0..=2π`.
    ///
    /// When created from a point/vector, the angle is in `-π..=π`.
    pub phi: Radians<T>,

    _dummy: PhantomData<S>,
}

impl<T: Float, S: Space> NormedSphericalPos<T, S> {
    pub fn new(theta: impl Into<Radians<T>>, phi: impl Into<Radians<T>>) -> Self {
        Self {
            theta: theta.into(),
            phi: phi.into(),
            _dummy: PhantomData,
        }
    }

    /// Returns the spherical coordinates for `(1, 0, 0)`, namely θ = π/2 and φ = 0.
    pub fn unit_x() -> Self {
        Self::new(Radians::quarter_turn(), Radians::zero())
    }

    /// Returns the spherical coordinates for `(0, 1, 0)`, namely θ = π/2 and φ = π/2.
    pub fn unit_y() -> Self {
        Self::new(Radians::quarter_turn(), Radians::quarter_turn())
    }

    /// Returns the spherical coordinates for `(0, 0, 1)`, namely θ = 0 and φ = 0.
    pub fn unit_z() -> Self {
        Self::new(Radians::zero(), Radians::zero())
    }

    pub fn with_radius(self, r: T) -> SphericalPos<T, S> {
        SphericalPos::new(self.theta, self.phi, r)
    }

    /// Convenience method for `Vec3::from(self)`.
    pub fn to_unit_vec(self) -> Vec3<T, S> {
        self.into()
    }
}

impl<T: Float, S: Space> From<SphericalPos<T, S>> for NormedSphericalPos<T, S> {
    fn from(p: SphericalPos<T, S>) -> Self {
        Self::new(p.theta, p.phi)
    }
}

impl<T: Float, S: Space> From<Vec3<T, S>> for NormedSphericalPos<T, S> {
    fn from(v: Vec3<T, S>) -> Self {
        let nv = v.normalized();
        Self::new(Radians::acos(nv.z), crate::atan2(nv.y, nv.x))
    }
}

impl<T: Float, S: Space> From<NormedSphericalPos<T, S>> for Vec3<T, S> {
    fn from(src: NormedSphericalPos<T, S>) -> Self {
        Vec3::new(
            src.phi.cos() * src.theta.sin(),
            src.phi.sin() * src.theta.sin(),
            src.theta.cos(),
        )
    }
}

impl<T: Float, S: Space> ops::Neg for NormedSphericalPos<T, S> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            theta: Radians::half_turn() - self.theta,
            phi: (self.phi + Radians::half_turn()).normalized(),
            _dummy: PhantomData,
        }
    }
}

impl<T: Float, S: Space> fmt::Debug for NormedSphericalPos<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(θ: {:?}, φ: {:?})", self.theta, self.phi)
    }
}

impl<T: Float, S: Space> Clone for NormedSphericalPos<T, S> {
    fn clone(&self) -> Self {
        Self::new(self.theta, self.phi)
    }
}
impl<T: Float, S: Space> Copy for NormedSphericalPos<T, S> {}
impl<T: Float, S: Space> PartialEq for NormedSphericalPos<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.theta == other.theta && self.phi == other.phi
    }
}
impl<T: Float + Hash, S: Space> Hash for NormedSphericalPos<T, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.theta.hash(state);
        self.phi.hash(state);
    }
}
