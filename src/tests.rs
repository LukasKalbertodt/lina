use crate::{
    angle_between, assert_approx_eq, clamp, dot, slerp, vec2, vec3, Degrees, Dir, Dir2f, Dir3f,
    Float, Radians, Vector, WorldSpace,
};


#[test]
fn angle_between_opposite_vectors() {
    let vectors = [
        // For this, the cosine of the angle used to be slightly below -1 due
        // to rounding errors, which resulted in NaN.
        vec3(5.8773685f32, -2.1701226, 6.303587),
        vec3(0.1, 0.2, 0.3),
        vec3(-3.7, 1.1, 0.0),
        vec3(1.0, 1.0, 1.0),
        vec3(1e-3, -4e2, 7.0),
    ];
    for v in vectors {
        assert_eq!(angle_between(v, -v), Radians(std::f32::consts::PI), "v = {v:?}");
        let angle = angle_between(v, v * -2.7);
        assert_approx_eq!(abs <= Radians(3e-7); angle, Radians(std::f32::consts::PI));
    }

    let angle = angle_between(vec2(0.3, 0.7), vec2(-0.3, -0.7));
    assert_eq!(angle, Radians(std::f64::consts::PI));
}

#[test]
fn angle_between_special_cases() {
    let v = vec3(1.0f32, -2.0, 3.0);
    assert_eq!(angle_between(v, v), Radians(0.0));
    assert_eq!(angle_between(v, v * 2.5), Radians(0.0));
    assert_eq!(angle_between(v, -v), Radians(std::f32::consts::PI));

    // In 1D, vectors either point in the same or the opposite direction.
    let v1 = |x: f32| <Vector<f32, 1>>::from([x]);
    assert_eq!(angle_between(v1(2.0), v1(3.0)), Radians(0.0));
    assert_eq!(angle_between(v1(2.0), v1(-3.0)), Radians(std::f32::consts::PI));
}

/// Checks `angle_between` for vectors with exactly known angles between them,
/// including angles very close to 0 and π. `u` and `w` have to be orthogonal.
/// `scales` are pairs of lengths for the two vectors.
fn check_angle_between_precision<T: Float, const N: usize>(
    u: [f64; N],
    w: [f64; N],
    scales: &[(f64, f64)],
    tolerance: f64,
) {
    let u = Vector::<f64, N>::from(u).normalized();
    let w = Vector::<f64, N>::from(w).normalized();
    assert!(dot(u, w).abs() < 1e-15);

    let to_t = |v: Vector<f64, N>| v.map(|c| num_traits::cast::<f64, T>(c).unwrap());
    let pi = std::f64::consts::PI;
    let angles = [0.0, 1e-6, 1e-4, 1e-2, 0.5, pi / 2.0, 2.0, pi - 1e-2, pi - 1e-4, pi - 1e-6, pi];
    for &(len_a, len_b) in scales {
        for theta in angles {
            let a = u * len_a;
            let b = (u * theta.cos() + w * theta.sin()) * len_b;
            let angle = angle_between(to_t(a), to_t(b)).0;
            let angle = num_traits::cast::<T, f64>(angle).unwrap();
            assert!(
                (angle - theta).abs() <= tolerance,
                "N = {N}, θ = {theta}, |a| = {len_a}, |b| = {len_b}: got {angle}",
            );
        }
    }
}

#[test]
fn angle_between_precision_f32() {
    // The last two pairs are very short and very long vectors, for which
    // squaring the components would under- or overflow.
    let scales = [(1.0, 1.0), (3.0, 0.2), (1e-15, 2e-15), (1e20, 3e18)];
    let tolerance = 4e-7;
    check_angle_between_precision::<f32, 2>([0.6, 0.8], [-0.8, 0.6], &scales, tolerance);
    check_angle_between_precision::<f32, 3>([1.0, 2.0, 3.0], [3.0, 0.0, -1.0], &scales, tolerance);
    check_angle_between_precision::<f32, 4>(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, -1.0, 4.0, -3.0],
        &scales,
        tolerance,
    );
}

#[test]
fn angle_between_precision_f64() {
    // The last two pairs are very short and very long vectors, for which
    // squaring the components would under- or overflow.
    let scales = [(1.0, 1.0), (3.0, 0.2), (1e-160, 1e-150), (1e160, 1e150)];
    let tolerance = 1e-15;
    check_angle_between_precision::<f64, 2>([0.6, 0.8], [-0.8, 0.6], &scales, tolerance);
    check_angle_between_precision::<f64, 3>([1.0, 2.0, 3.0], [3.0, 0.0, -1.0], &scales, tolerance);
    check_angle_between_precision::<f64, 4>(
        [1.0, 2.0, 3.0, 4.0],
        [2.0, -1.0, 4.0, -3.0],
        &scales,
        tolerance,
    );
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
