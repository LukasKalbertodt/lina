use std::{fmt, ops};

use crate::{vec3, Vec3, Point3, Float, Radians};


/// A 3D point described in spherical coordinates (theta θ, phi φ, radius r).
#[derive(Clone, Copy, PartialEq, Hash)]
pub struct SphericalPos<T: Float> {
    /// Vertical angle θ: 0° points up (north pole), 180° points down (south pole),
    /// 90° is on the equator. Should always be in the range `0..=π`.
    pub theta: Radians<T>,

    /// Horizontal angle φ: 0° points +x, 90° points +y, 180° points -y, 270°
    /// points -y. Should always be in the range `0..=2π`.
    pub phi: Radians<T>,

    /// The distance from the origin.
    pub r: T,
}

impl<T: Float> SphericalPos<T> {
    /// Convenience function for `Vec3::from(self)`.
    pub fn to_vec(self) -> Vec3<T> {
        self.into()
    }

    /// Convenience function for `Point3::from(self)`.
    pub fn to_point(self) -> Point3<T> {
        self.into()
    }

    pub fn without_radius(self) -> NormedSphericalPos<T> {
        self.into()
    }
}

impl<T: Float> From<Point3<T>> for SphericalPos<T> {
    fn from(p: Point3<T>) -> Self {
        Self::from(p.to_vec())
    }
}

impl<T: Float> From<Vec3<T>> for SphericalPos<T> {
    fn from(v: Vec3<T>) -> Self {
        let nv = v.normalized();
        Self {
            theta: Radians::acos(nv.z),
            phi: crate::atan2(nv.y, nv.x),
            r: v.length(),
        }
    }
}

impl<T: Float> From<SphericalPos<T>> for Point3<T> {
    fn from(src: SphericalPos<T>) -> Self {
        Vec3::from(src).to_point()
    }
}

impl<T: Float> From<SphericalPos<T>> for Vec3<T> {
    fn from(src: SphericalPos<T>) -> Self {
        vec3(
            src.r * src.phi.cos() * src.theta.sin(),
            src.r * src.phi.sin() * src.theta.sin(),
            src.r * src.theta.cos(),
        )
    }
}

impl<T: Float> fmt::Debug for SphericalPos<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(θ: {:?}, φ: {:?}, r: {:?})", self.theta, self.phi, self.r)
    }
}


/// A point on the boundary of a unit sphere described in spherical coordinates
/// (without radius).
#[derive(Clone, Copy, PartialEq, Hash)]
pub struct NormedSphericalPos<T: Float> {
    /// Vertical angle θ: 0° points up (north pole), 180° points down (south pole),
    /// 90° is on the equator. Should always be in the range `0..=π`.
    pub theta: Radians<T>,

    /// Horizontal angle φ: 0° points +x, 90° points +y, 180° points -y, 270°
    /// points -y. Should always be in the range `0..=2π`.
    pub phi: Radians<T>,
}

impl<T: Float> NormedSphericalPos<T> {
    /// Returns the spherical coordinates for `(1, 0, 0)`, namely θ = π/2 and φ = 0.
    pub fn unit_x() -> Self {
        Self {
            theta: Radians::quarter_turn(),
            phi: Radians(T::zero()),
        }
    }

    /// Returns the spherical coordinates for `(0, 1, 0)`, namely θ = π/2 and φ = π/2.
    pub fn unit_y() -> Self {
        Self {
            theta: Radians::quarter_turn(),
            phi: Radians::quarter_turn(),
        }
    }

    /// Returns the spherical coordinates for `(0, 0, 1)`, namely θ = 0 and φ = 0.
    pub fn unit_z() -> Self {
        Self {
            theta: Radians(T::zero()),
            phi: Radians(T::zero()),
        }
    }

    pub fn with_radius(self, r: T) -> SphericalPos<T> {
        SphericalPos {
            theta: self.theta,
            phi: self.phi,
            r,
        }
    }

    /// Convenience method for `Vec3::from(self)`.
    pub fn to_unit_vec(self) -> Vec3<T> {
        self.into()
    }
}

impl<T: Float> From<SphericalPos<T>> for NormedSphericalPos<T> {
    fn from(p: SphericalPos<T>) -> Self {
        Self {
            theta: p.theta,
            phi: p.phi,
        }
    }
}

impl<T: Float> From<Vec3<T>> for NormedSphericalPos<T> {
    fn from(v: Vec3<T>) -> Self {
        let nv = v.normalized();
        Self {
            theta: Radians::acos(nv.z),
            phi: crate::atan2(nv.y, nv.x),
        }
    }
}

impl<T: Float> From<NormedSphericalPos<T>> for Vec3<T> {
    fn from(src: NormedSphericalPos<T>) -> Self {
        vec3(
            src.phi.cos() * src.theta.sin(),
            src.phi.sin() * src.theta.sin(),
            src.theta.cos(),
        )
    }
}

impl<T: Float> ops::Neg for NormedSphericalPos<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            theta: Radians::half_turn() - self.theta,
            phi: (self.phi + Radians::half_turn()).normalized(),
        }
    }
}

impl<T: Float> fmt::Debug for NormedSphericalPos<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(θ: {:?}, φ: {:?})", self.theta, self.phi)
    }
}
