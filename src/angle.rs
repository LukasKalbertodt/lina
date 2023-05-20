use std::{fmt, ops};

use crate::Float;


macro_rules! shared_methods {
    () => {
        pub fn half_turn() -> Self {
            Self::full_turn() / T::two()
        }

        pub fn quarter_turn() -> Self {
            Self::full_turn() / T::four()
        }

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
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Radians<T: Float>(pub T);

/// An angle in degrees. A full rotation is 360°.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Degrees<T: Float>(pub T);

impl<T: Float> Radians<T> {
    /// Returns 2π rad.
    pub fn full_turn() -> Self {
        Self(T::PI() + T::PI())
    }

    /// Converts this angle to degrees (`rad * 180/π`).
    pub fn to_degrees(self) -> Degrees<T> {
        Degrees(self.0.to_degrees())
    }

    shared_methods!();
}

impl<T: Float> Degrees<T> {
    /// Returns 360°.
    pub fn full_turn() -> Self {
        Self(num_traits::cast(360.0).unwrap())
    }

    /// Converts this angle to radians (`degrees * π/180`).
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

macro_rules! impl_debug_display {
    ($ty:ident, $unit:literal) => {
        impl<T: Float> fmt::Debug for $ty<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)?;
                f.write_str($unit)
            }
        }

        impl<T: Float + fmt::Display> fmt::Display for $ty<T> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)?;
                f.write_str($unit)
            }
        }
    };
}

impl_debug_display!(Radians, " rad");
impl_debug_display!(Degrees, "°");

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

// Scalar multiplication: `scalar * angle`. Unfortunately, due to Rust's orphan
// rules, this cannot be implemented generically. So we just implement it for
// core primitive types.
macro_rules! impl_scalar_mul {
    ($angle_ty:ident; $($ty:ident),*) => {
        $(
            impl ops::Mul<$angle_ty<$ty>> for $ty {
                type Output = $angle_ty<$ty>;
                fn mul(self, rhs: $angle_ty<$ty>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}

impl_scalar_mul!(Radians; f32, f64);
impl_scalar_mul!(Degrees; f32, f64);
