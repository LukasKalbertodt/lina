use crate::{HcPoint, Point, Point3f, Vec2, Vec3, Vec3f, point2, point3, vec2, vec3, Vector};

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
    assert_eq!(3usize * vec2(1, 2), vec2(3, 6));
    assert_eq!(-2isize * vec3(1, 2, 3), vec3(-2, -4, -6));
    assert_eq!(3usize * <crate::Dir2<usize>>::unit_y(), vec2(0, 3));
    assert_eq!(-2isize * <crate::Dir2<isize>>::unit_x(), vec2(-2, 0));

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

#[test]
fn truncate() {
    assert_eq!(<Point<_, 2>>::from([1, 2]).truncate(), Point::from([1]));
    assert_eq!(<Point<_, 3>>::from([1, 2, 3]).truncate(), Point::from([1, 2]));
    assert_eq!(<Point<_, 4>>::from([1, 2, 3, 4]).truncate(), Point::from([1, 2, 3]));

    assert_eq!(<Vector<_, 2>>::from([1, 2]).truncate(), Vector::from([1]));
    assert_eq!(<Vector<_, 3>>::from([1, 2, 3]).truncate(), Vector::from([1, 2]));
    assert_eq!(<Vector<_, 4>>::from([1, 2, 3, 4]).truncate(), Vector::from([1, 2, 3]));
}

#[test]
fn extend() {
    assert_eq!(<Point<_, 1>>::from([1]).extend(2), Point::from([1, 2]));
    assert_eq!(<Point<_, 2>>::from([1, 2]).extend(3), Point::from([1, 2, 3]));
    assert_eq!(<Point<_, 3>>::from([1, 2, 3]).extend(4), Point::from([1, 2, 3, 4]));

    assert_eq!(<Vector<_, 1>>::from([1]).extend(2), Vector::from([1, 2]));
    assert_eq!(<Vector<_, 2>>::from([1, 2]).extend(3), Vector::from([1, 2, 3]));
    assert_eq!(<Vector<_, 3>>::from([1, 2, 3]).extend(4), Vector::from([1, 2, 3, 4]));
}

#[test]
fn hc_point_array_conversions() {
    let p1 = <HcPoint<_, 1>>::new([1], 2);
    let p2 = <HcPoint<_, 2>>::new([1, 2], 3);
    let p3 = <HcPoint<_, 3>>::new([1, 2, 3], 4);

    assert_eq!(p1.to_array(), [1, 2]);
    assert_eq!(p2.to_array(), [1, 2, 3]);
    assert_eq!(p3.to_array(), [1, 2, 3, 4]);

    assert_eq!(<[_; 2]>::from(p1), [1, 2]);
    assert_eq!(<[_; 3]>::from(p2), [1, 2, 3]);
    assert_eq!(<[_; 4]>::from(p3), [1, 2, 3, 4]);

    assert_eq!(<HcPoint<_, 1>>::from([1, 2]), p1);
    assert_eq!(<HcPoint<_, 2>>::from([1, 2, 3]), p2);
    assert_eq!(<HcPoint<_, 3>>::from([1, 2, 3, 4]), p3);

    assert_eq!(AsRef::<[_; 2]>::as_ref(&p1), &[1, 2]);
    assert_eq!(AsRef::<[_; 3]>::as_ref(&p2), &[1, 2, 3]);
    assert_eq!(AsRef::<[_; 4]>::as_ref(&p3), &[1, 2, 3, 4]);

    let mut p3 = p3;
    AsMut::<[_; 4]>::as_mut(&mut p3)[3] = 7;
    assert_eq!(p3.weight, 7);
}

#[test]
fn point4_named_components() {
    use crate::named_scalar::{HasW, HasX};

    let mut p = <Point<_, 4>>::from([1, 2, 3, 4]);
    assert_eq!((p.x, p.y, p.z, p.w), (1, 2, 3, 4));
    p.w = 7;
    assert_eq!(p, Point::from([1, 2, 3, 7]));
    assert_eq!((*p.x(), *p.w()), (1, 7));
}
