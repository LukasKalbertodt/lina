use std::{fmt::Debug, ops};

use crate::Float;


macro_rules! shared_methods {
    () => {
        pub fn sin(self) -> T {
            Radians::from(self).0.sin()
        }
        pub fn cos(self) -> T {
            Radians::from(self).0.cos()
        }
        pub fn tan(self) -> T {
            Radians::from(self).0.tan()
        }

        pub fn asin(v: T) -> Self {
            Radians(v.asin()).into()
        }
        pub fn acos(v: T) -> Self {
            Radians(v.acos()).into()
        }
        pub fn atan(v: T) -> Self {
            Radians(v.atan()).into()
        }

        /// Returns the angle normalized into the range `0..Self::full_turn()`.
        #[must_use = "to normalize in-place, use `normalize`, not `normalized`"]
        pub fn normalized(self) -> Self {
            let rem = self.0 % Self::full_turn().0;
            if rem < T::zero() {
                Self(rem + Self::full_turn().0)
            } else {
                Self(rem)
            }
        }

        /// Normalizes this angle *in-place* into the range `0..Self::full_turn()`.
        pub fn normalize(&mut self) {
            *self = (*self).normalized();
        }
    };
}


/// An angle in radians. A full rotation is 2π rad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Radians<T: Float>(pub T);

/// An angle in degrees. A full rotation is 360°.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Degrees<T: Float>(pub T);

impl<T: Float> Radians<T> {
    pub fn full_turn() -> Self {
        Self(T::PI() + T::PI())
    }

    pub fn to_degrees(self) -> Degrees<T> {
        Degrees(self.0.to_degrees())
    }

    shared_methods!();
}

impl<T: Float> Degrees<T> {
    pub fn full_turn() -> Self {
        Self(num_traits::cast(360.0).unwrap())
    }
    pub fn to_radians(self) -> Radians<T> {
        Radians(self.0.to_radians())
    }

    shared_methods!();
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
