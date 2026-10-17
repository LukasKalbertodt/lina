use crate::{angle_between, assert_approx_eq, vec2, vec3, Degrees, Radians};


#[test]
fn angle_between_opposite_vectors() {
    // Due to rounding errors, the cosine of the angle between these is
    // slightly below -1, which used to result in NaN.
    let v = vec3(5.8773685f32, -2.1701226, 6.303587);
    assert_eq!(angle_between(v, -v), Radians(std::f32::consts::PI));

    let vectors = [
        vec3(0.1f32, 0.2, 0.3),
        vec3(-3.7, 1.1, 0.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1e-3, -4e2, 7.0),
    ];
    for v in vectors {
        let angle = angle_between(v, -v);
        assert!(!angle.0.is_nan(), "angle between {v:?} and -{v:?} is NaN");
        assert_approx_eq!(abs <= Radians(1e-3); angle, Radians(std::f32::consts::PI));
    }

    let angle = angle_between(vec2(0.3, 0.7), vec2(-0.3, -0.7));
    assert_approx_eq!(abs <= Radians(1e-6); angle, Radians(std::f64::consts::PI));
}


#[test]
fn angle_normalized_tiny_negative() {
    // For these, `angle + full_turn` is rounded to exactly `full_turn`.
    assert_eq!(Radians(-1e-20f64).normalized(), Radians(0.0));
    assert_eq!(Radians(-1e-10f32).normalized(), Radians(0.0));
    assert_eq!(Degrees(-1e-14f64).normalized(), Degrees(0.0));
    assert_eq!(Degrees(-1e-6f32).normalized(), Degrees(0.0));

    assert!(Radians(-1e-6f64).normalized() < Radians::full_turn());
    assert!(Degrees(-1e-3f32).normalized() < Degrees::full_turn());
}
