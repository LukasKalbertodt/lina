use crate::*;

#[test]
fn add_point_vec() {
    assert_eq!(point2(1, 2) + vec2(3, 5), point2(4, 7));
    assert_eq!(point3(1, 2, 3) + vec3(3, 5, 7), point3(4, 7, 10));

    let mut x = point2(1.0, 2.0);
    x += vec2(3.0, 5.0);
    assert_eq!(x, point2(4.0, 7.0));

    let mut x = point3(1.0, 2.0, 3.0);
    x += vec3(3.0, 5.0, 7.0);
    assert_eq!(x, point3(4.0, 7.0, 10.0));
}

#[test]
fn add_vec_vec() {
    assert_eq!(vec2(1, 2) + vec2(3, 5), vec2(4, 7));
    assert_eq!(vec3(1, 2, 3) + vec3(3, 5, 7), vec3(4, 7, 10));

    let mut x = vec2(1.0, 2.0);
    x += vec2(3.0, 5.0);
    assert_eq!(x, vec2(4.0, 7.0));

    let mut x = vec3(1.0, 2.0, 3.0);
    x += vec3(3.0, 5.0, 7.0);
    assert_eq!(x, vec3(4.0, 7.0, 10.0));
}

#[test]
fn sub_point_vec() {
    assert_eq!(point2(4, 7) - vec2(3, 5), point2(1, 2));
    assert_eq!(point3(4, 7, 10) - vec3(3, 5, 7), point3(1, 2, 3));

    let mut x = point2(4.0, 7.0);
    x -= vec2(3.0, 5.0);
    assert_eq!(x, point2(1.0, 2.0));

    let mut x = point3(4.0, 7.0, 10.0);
    x -= vec3(3.0, 5.0, 7.0);
    assert_eq!(x, point3(1.0, 2.0, 3.0));
}

#[test]
fn sub_vec_vec() {
    assert_eq!(vec2(4, 7) - vec2(3, 5), vec2(1, 2));
    assert_eq!(vec3(4, 7, 10) - vec3(3, 5, 7), vec3(1, 2, 3));

    let mut x = vec2(4.0, 7.0);
    x -= vec2(3.0, 5.0);
    assert_eq!(x, vec2(1.0, 2.0));

    let mut x = vec3(4.0, 7.0, 10.0);
    x -= vec3(3.0, 5.0, 7.0);
    assert_eq!(x, vec3(1.0, 2.0, 3.0));
}

#[test]
fn mul() {
    assert_eq!(vec2(1, 2) * 3, vec2(3, 6));
    assert_eq!(vec3(1, 2, 3) * 3, vec3(3, 6, 9));
    assert_eq!(vec4(1, 2, 3, 4) * 3, vec4(3, 6, 9, 12));
    assert_eq!(point2(1, 2) * 3, point2(3, 6));
    assert_eq!(point3(1, 2, 3) * 3, point3(3, 6, 9));

    let mut x = vec3(0.1, 0.2, 0.3);
    x *= 10.0;
    assert_eq!(x, vec3(1.0, 2.0, 3.0));

    let mut x = point3(0.1, 0.2, 0.3);
    x *= 10.0;
    assert_eq!(x, point3(1.0, 2.0, 3.0));
}

#[test]
fn div() {
    assert_eq!(vec2(3, 6) / 3, vec2(1, 2));
    assert_eq!(vec3(3, 6, 9) / 3, vec3(1, 2, 3));
    assert_eq!(vec4(3, 6, 9, 12) / 3, vec4(1, 2, 3, 4));
    assert_eq!(point2(3, 6) / 3, point2(1, 2));
    assert_eq!(point3(3, 6, 9) / 3, point3(1, 2, 3));

    let mut x = vec3(1.0, 2.0, 3.0);
    x /= 10.0;
    assert_eq!(x, vec3(0.1, 0.2, 0.3));

    let mut x = point3(1.0, 2.0, 3.0);
    x /= 10.0;
    assert_eq!(x, point3(0.1, 0.2, 0.3));
}

#[test]
fn neg() {
    assert_eq!(-vec2(9.0, -4.3), vec2(-9.0, 4.3));
    assert_eq!(-point2(9.0, -4.3), point2(-9.0, 4.3));
}
