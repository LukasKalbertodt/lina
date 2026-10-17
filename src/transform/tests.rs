use crate::{assert_approx_eq, vec3, Degrees};

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

/// Checks that `m` is a rotation matrix that maps `from` to `to`.
fn check_aligning<T: Float + crate::ApproxEq<Tolerance = T>>(
    from: Dir3<T>,
    to: Dir3<T>,
    tolerance: T,
) {
    let m = rotate3d_aligning(from, to);
    assert_approx_eq!(abs <= tolerance; m.transform(from), to.to_unit_vec());
    assert_approx_eq!(abs <= tolerance; m.transposed::<WorldSpace, WorldSpace>() * m, Mat3::identity());
    assert_approx_eq!(abs <= tolerance; m.determinant(), T::one());
}

#[test]
fn rotate3d_aligning_general() {
    let dirs = [
        <Dir3<f64>>::new(1.0, 0.0, 0.0),
        <Dir3<f64>>::new(0.0, 1.0, 0.0),
        <Dir3<f64>>::new(1.0, 2.0, 3.0),
        <Dir3<f64>>::new(-0.3, 0.1, -7.0),
        <Dir3<f64>>::new(5.0, -5.0, 0.2),
    ];
    for from in dirs {
        for to in dirs {
            check_aligning(from, to, 1e-14);
            check_aligning(from.to_f32(), to.to_f32(), 1e-6);
        }
    }
}

#[test]
fn rotate3d_aligning_almost_opposite() {
    // Pairs of `from` and a direction `w` perpendicular to it.
    let cases = [
        (vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)),
        (vec3(1.0, 2.0, 3.0), vec3(3.0, 0.0, -1.0)),
        (vec3(-0.44, -0.61, 0.66), vec3(0.61, -0.44, 0.0)),
    ];
    for (from, w) in cases {
        let (from, w) = (from.normalized(), w.normalized());
        for delta in [1e-1f64, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] {
            // `to` is `π - delta` away from `from`.
            let to = -from * delta.cos() + w * delta.sin();
            check_aligning(from.to_dir(), to.to_dir(), 1e-14);
            check_aligning(from.to_f32().to_dir(), to.to_f32().to_dir(), 2e-6);
        }
    }
}

#[test]
#[should_panic]
fn rotate3d_aligning_opposite() {
    let d = <Dir3<f32>>::new(1.0, 2.0, 3.0);
    rotate3d_aligning(d, -d);
}
