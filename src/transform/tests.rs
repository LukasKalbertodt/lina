use crate::Degrees;

use super::*;


#[test]
#[should_panic]
fn perspective_negative_fov() {
    perspective(Radians(-1.0), 1.0, 0.1..=10.0, 0.0..=1.0);
}

#[test]
#[should_panic]
fn perspective_too_large_fov() {
    perspective(Degrees(190.0), 1.0, 0.1..=10.0, 0.0..=1.0);
}

#[test]
#[should_panic]
fn perspective_negative_aspect_ratio() {
    perspective(Radians(1.0), -0.2, 0.1..=10.0, 0.0..=1.0);
}
