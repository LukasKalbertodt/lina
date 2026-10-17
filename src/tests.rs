use crate::{
    angle_between, assert_approx_eq, clamp, slerp, vec2, vec3, Degrees, Dir, Dir2f, Dir3f,
    Radians, WorldSpace,
};


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

#[test]
fn clamp_empty_range() {
    assert_eq!(clamp(1.0, 1.0..=1.0), 1.0);
    assert_eq!(clamp(-4.0, 1.0..=1.0), 1.0);
    assert_eq!(clamp(9.0, 1.0..=1.0), 1.0);
    assert_eq!(clamp(5, 3..=3), 3);
}

#[test]
#[should_panic(expected = "'min' is larger than 'max'")]
fn clamp_min_larger_than_max() {
    clamp(1.0, 2.0..=1.0);
}

#[test]
#[should_panic(expected = "in range passed to `clamp`")]
fn clamp_nan_in_range() {
    clamp(1.0, 0.0..=f32::NAN);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "passed as 'val' to `clamp`")]
fn clamp_nan_val_debug() {
    clamp(f32::NAN, 0.0..=1.0);
}

#[test]
#[cfg(not(debug_assertions))]
fn clamp_nan_val_release() {
    assert!(clamp(f32::NAN, 0.0..=1.0).is_nan());
}

#[test]
fn dir_debug() {
    assert_eq!(format!("{:?}", <Dir2f>::unit_y()), "Dir[0.0, 1.0]");
    assert_eq!(format!("{:?}", <Dir3f>::unit_z()), "Dir[0.0, 0.0, 1.0]");
    assert_eq!(format!("{:?}", <Dir<f32, 1, WorldSpace>>::from_vec([-2.0].into())), "Dir[-1.0]");
}

#[test]
fn dir_eq() {
    fn assert_eq_impl<T: Eq>(_: T) {}
    assert_eq_impl(<Dir<i32, 3, WorldSpace>>::unit_x());
    assert_eq_impl(<Dir<i8, 2, WorldSpace>>::new_unchecked(0, -1));
}

#[test]
#[cfg_attr(not(debug_assertions), ignore = "requires debug assertions")]
#[should_panic(expected = "opposite directions")]
fn slerp_opposite() {
    let v = vec3(1.0, 2.0, 3.0);
    slerp(v, -v * 2.0, 0.5);
}

#[test]
#[cfg_attr(not(debug_assertions), ignore = "requires debug assertions")]
#[should_panic(expected = "opposite directions")]
fn slerp_opposite_f32() {
    // `angle_between` used to return NaN for these, so it's worth testing.
    let v = vec3(5.8773685f32, -2.1701226, 6.303587);
    slerp(v, -v, 0.5);
}
