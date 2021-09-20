use crate::*;


#[test]
fn add() {
    let a = Matrix::from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let b = Matrix::from_rows([[60, 50, 40], [30, 20, 10]]);
    let c = Matrix::from_rows([[61, 52, 43], [34, 25, 16]]);
    assert_eq!(a + b, c);

    let mut b = b;
    b += a;
    assert_eq!(b, c);
}

#[test]
fn sub() {
    let a = Matrix::from_rows([[60, 50, 40], [30, 20, 10]]);
    let b = Matrix::from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let c = Matrix::from_rows([[59, 48, 37], [26, 15,  4]]);
    assert_eq!(a - b, c);

    let mut a = a;
    a -= b;
    assert_eq!(a, c);
}

#[test]
fn mul_scalar() {
    let a = Matrix::from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let c = Matrix::from_rows([[ 3,  6,  9], [12, 15, 18]]);
    assert_eq!(a * 3, c);

    let mut a = a;
    a *= 3;
    assert_eq!(a, c);
}

#[test]
fn div_scalar() {
    let a = Matrix::from_rows([[ 3,  6,  9], [12, 15, 18]]);
    let c = Matrix::from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    assert_eq!(a / 3, c);

    let mut a = a;
    a /= 3;
    assert_eq!(a, c);
}

#[test]
fn neg() {
    let a = Matrix::from_rows([[-1.0,  2.0,  3.0], [ 4.0, -5.0,  6.0]]);
    let c = Matrix::from_rows([[ 1.0, -2.0, -3.0], [-4.0,  5.0, -6.0]]);
    assert_eq!(-a, c);
}

#[test]
fn mul_matrix() {
    // 2x2 mul 2x2 -> 2x2
    assert_eq!(
        Mat2::from_rows([[8, 9], [5, -1]]) * Mat2::from_rows([[-2, 3], [4, 0]]),
        Mat2::from_rows([[20, 24], [-14, 15]]),
    );

    // 3x2 mul 2x3 -> 2x2
    assert_eq!(
        Matrix::from_rows([[0, -1, 2], [4, 11, 2]]) * Matrix::from_rows([[3, -1], [1, 2], [6, 1]]),
        Matrix::from_rows([[11, 0], [35, 20]]),
    );

    // 2x3 mul 3x2 -> 3x3
    assert_eq!(
        Matrix::from_rows([[3, -1], [1, 2], [6, 1]]) * Matrix::from_rows([[0, -1, 2], [4, 11, 2]]),
        Matrix::from_rows([[-4, -14, 4], [8, 21, 6], [4, 5, 14]]),
    );

    let a = Matrix::from_rows([
        [ 0,  2,  4,  6],
        [ 1,  3,  5,  7],
        [17, 19, 21, 23],
        [25, 27, 29, 31],
    ]);
    let b = Matrix::from_rows([
        [ 8, 10, 12, 14],
        [ 9, 11, 13, 15],
        [16, 18, 20, 22],
        [24, 26, 28, 30],
    ]);
    assert_eq!(a * b, Matrix::from_rows([
        [ 226,  250,  274,  298],
        [ 283,  315,  347,  379],
        [1195, 1355, 1515, 1675],
        [1651, 1875, 2099, 2323],
    ]));
    assert_eq!(a * Matrix::identity(), a);
    assert_eq!(Matrix::identity() * a, a);
    assert_eq!(b * Matrix::identity(), b);
    assert_eq!(Matrix::identity() * b, b);
}

#[test]
#[should_panic]
fn perspective_negative_fov() {
    Mat4f::perspective(Radians(-1.0), 1.0, 0.1..=10.0, 0.0..=1.0);
}

#[test]
#[should_panic]
fn perspective_too_large_fov() {
    Mat4f::perspective(Degrees(190.0), 1.0, 0.1..=10.0, 0.0..=1.0);
}

#[test]
#[should_panic]
fn perspective_negative_aspect_ratio() {
    Mat4f::perspective(Radians(1.0), -0.2, 0.1..=10.0, 0.0..=1.0);
}
