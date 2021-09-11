use std::{fmt::Debug, ops};
use num_traits::{Float as _, Zero};

use crate::Float;


/// Values representing an angle. Implemented by [`Radians`] and [`Degrees`].
pub trait Angle: Copy + Debug
    + From<Radians<Self::Unitless>> + From<Degrees<Self::Unitless>>
    + ops::Add<Self, Output = Self> + ops::AddAssign
    + ops::Sub<Self, Output = Self> + ops::SubAssign
    + ops::Mul<Self::Unitless, Output = Self> + ops::MulAssign<Self::Unitless>
    + ops::Div<Self::Unitless, Output = Self> + ops::DivAssign<Self::Unitless>
    + ops::Neg
{
    type Unitless: Float;

    /// A full rotation: 2π radians or 360°.
    fn full_turn() -> Self;

    fn to_radians(self) -> Radians<Self::Unitless>;
    fn to_degrees(self) -> Degrees<Self::Unitless>;
    fn unitless(self) -> Self::Unitless;
    fn from_unitless(v: Self::Unitless) -> Self;

    fn sin(self) -> Self::Unitless {
        self.to_radians().unitless().sin()
    }
    fn cos(self) -> Self::Unitless {
        self.to_radians().unitless().cos()
    }
    fn tan(self) -> Self::Unitless {
        self.to_radians().unitless().tan()
    }

    fn asin(v: Self::Unitless) -> Self {
        Radians(v.asin()).into()
    }
    fn acos(v: Self::Unitless) -> Self {
        Radians(v.acos()).into()
    }
    fn atan(v: Self::Unitless) -> Self {
        Radians(v.atan()).into()
    }

    /// Returns the angle normalized into the range `0..Self::full_turn()`.
    fn normalized(self) -> Self {
        let rem = self.unitless() % Self::full_turn().unitless();
        if rem < Self::Unitless::zero() {
            Self::from_unitless(rem + Self::full_turn().unitless())
        } else {
            Self::from_unitless(rem)
        }
    }

    /// Normalizes this angle *in-place* into the range `0..Self::full_turn()`.
    fn normalize(&mut self) {
        *self = (*self).normalized();
    }
}

/// An angle in radians. A full rotation is 2π radians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Radians<T: Float>(pub T);

/// An angle in degrees. A full rotation is 360°.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Degrees<T: Float>(pub T);

impl<T: Float> Angle for Radians<T> {
    type Unitless = T;

    fn full_turn() -> Self {
        Self(T::PI() + T::PI())
    }
    fn to_radians(self) -> Radians<Self::Unitless> {
        self
    }
    fn to_degrees(self) -> Degrees<T> {
        Degrees(self.0.to_degrees())
    }
    fn unitless(self) -> Self::Unitless {
        self.0
    }
    fn from_unitless(v: Self::Unitless) -> Self {
        Self(v)
    }
}

impl<T: Float> Angle for Degrees<T> {
    type Unitless = T;

    fn full_turn() -> Self {
        Self(num_traits::cast(360.0).unwrap())
    }
    fn to_radians(self) -> Radians<T> {
        Radians(self.0.to_radians())
    }
    fn to_degrees(self) -> Degrees<T> {
        self
    }
    fn unitless(self) -> Self::Unitless {
        self.0
    }
    fn from_unitless(v: Self::Unitless) -> Self {
        Self(v)
    }
}

impl<T: Float> From<Degrees<T>> for Radians<T> {
    fn from(src: Degrees<T>) -> Self {
        src.to_radians()
    }
}

impl<T: Float> From<Radians<T>> for Degrees<T> {
    fn from(src: Radians<T>) -> Self {
        src.to_degrees()
    }
}

macro_rules! impl_ops {
    ($ty:ident) => {
        impl<T: Float> ops::Mul<T> for $ty<T> {
            type Output = Self;
            fn mul(self, rhs: T) -> Self {
                Self(self.0 * rhs)
            }
        }

        impl<T: Float> ops::MulAssign<T> for $ty<T> {
            fn mul_assign(&mut self, rhs: T) {
                self.0 *= rhs;
            }
        }


        impl<T: Float> ops::Div<T> for $ty<T> {
            type Output = Self;
            fn div(self, rhs: T) -> Self {
                Self(self.0 / rhs)
            }
        }

        impl<T: Float> ops::DivAssign<T> for $ty<T> {
            fn div_assign(&mut self, rhs: T) {
                self.0 /= rhs;
            }
        }

        impl<T: Float> ops::Neg for $ty<T> {
            type Output = Self;
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl<T: Float> ops::Add for $ty<T> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<T: Float> ops::AddAssign for $ty<T> {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl<T: Float> ops::Sub for $ty<T> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl<T: Float> ops::SubAssign for $ty<T> {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }
    };
}

impl_ops!(Radians);
impl_ops!(Degrees);
