use std::fmt::Debug;
use num_traits::Float as _;

use crate::Float;


/// Values representing an angle. Implemented by [`Radians`] and [`Degrees`].
pub trait Angle: From<Radians<Self::Unitless>> + From<Degrees<Self::Unitless>> + Copy + Debug {
    type Unitless: Float;

    /// A full rotation: 2π radians or 360°.
    fn full_turn() -> Self;

    fn to_radians(self) -> Radians<Self::Unitless>;
    fn to_degrees(self) -> Degrees<Self::Unitless>;
    fn unitless(self) -> Self::Unitless;

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
}

/// An angle in radians. A full rotation is 2π radians.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Radians<T: Float>(pub T);

/// An angle in degree. A full rotation is 360°.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
