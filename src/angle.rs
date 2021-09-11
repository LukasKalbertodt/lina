use crate::Float;


/// An angle in radians. A full rotation is 2π radians.
pub struct Radians<T: Float>(pub T);

/// An angle in degree. A full rotation is 360°.
pub struct Degrees<T: Float>(pub T);

impl<T: Float> Radians<T> {
    pub fn to_degrees(self) -> Degrees<T> {
        Degrees(self.0.to_degrees())
    }
}

impl<T: Float> Degrees<T> {
    pub fn to_radians(self) -> Radians<T> {
        Radians(self.0.to_radians())
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
