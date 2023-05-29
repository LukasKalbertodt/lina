use crate::{Point, Point3f, Vec2, Vec3, Vec3f, point2, point3, vec2, vec3};

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
fn sub_point_point() {
    assert_eq!(point2(4, 7) - point2(3, 5), vec2(1, 2));
    assert_eq!(point3(4, 7, 10) - point3(3, 5, 7), vec3(1, 2, 3));
}

#[test]
fn mul() {
    assert_eq!(vec2(1, 2) * 3, vec2(3, 6));
    assert_eq!(vec3(1, 2, 3) * 3, vec3(3, 6, 9));

    assert_eq!(3.0 * vec2(1.0, 2.0), vec2(3.0, 6.0));
    assert_eq!(3 * vec2(1, 2), vec2(3, 6));
    assert_eq!(3 * vec3(1, 2, 3), vec3(3, 6, 9));

    let mut x = vec3(0.1, 0.2, 0.3);
    x *= 10.0;
    assert_eq!(x, vec3(1.0, 2.0, 3.0));
}

#[test]
fn div() {
    assert_eq!(vec2(3, 6) / 3, vec2(1, 2));
    assert_eq!(vec3(3, 6, 9) / 3, vec3(1, 2, 3));

    let mut x = vec3(1.0, 2.0, 3.0);
    x /= 10.0;
    assert_eq!(x, vec3(0.1, 0.2, 0.3));
}

#[test]
fn neg() {
    assert_eq!(-vec2(9.0, -4.3), vec2(-9.0, 4.3));
}

#[test]
fn centroid() {
    assert_eq!(Point3f::centroid([]), None::<Point3f>);
    assert_eq!(Point::centroid([point2(3.5, 1.0)]), Some(point2(3.5, 1.0)));
    assert_eq!(Point::centroid([point2(0.0, 0.0), point2(4.0, 8.0)]), Some(point2(2.0, 4.0)));
    assert_eq!(
        Point::centroid([point2(-1.0, 9.0), point2(2.0, 0.0), point2(8.0, 3.0)]),
        Some(point2(3.0, 4.0)),
    );
}

#[test]
fn vector_iter_sum() {
    assert_eq!(<Vec<Vec3f>>::new().into_iter().sum::<Vec3f>(), vec3(0.0, 0.0, 0.0));
    assert_eq!(vec![vec3(3, 4, 5)].into_iter().sum::<Vec3<i32>>(), vec3(3, 4, 5));
    assert_eq!(vec![vec2(1, 3), vec2(5, 2)].into_iter().sum::<Vec2<i32>>(), vec2(6, 5));
}
