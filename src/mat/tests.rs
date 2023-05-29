use crate::{Mat2, Matrix, approx::assert_approx_eq, GenericSpace, Scalar, Mat3};


fn from_rows<T: Scalar, const C: usize, const R: usize>(
    rows: [[T; C]; R],
) -> Matrix<T, C, R, GenericSpace, GenericSpace> {
    Matrix::from_rows(rows)
}

#[test]
fn add() {
    let a = from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let b = from_rows([[60, 50, 40], [30, 20, 10]]);
    let c = from_rows([[61, 52, 43], [34, 25, 16]]);
    assert_eq!(a + b, c);

    let mut b = b;
    b += a;
    assert_eq!(b, c);
}

#[test]
fn sub() {
    let a = from_rows([[60, 50, 40], [30, 20, 10]]);
    let b = from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let c = from_rows([[59, 48, 37], [26, 15,  4]]);
    assert_eq!(a - b, c);

    let mut a = a;
    a -= b;
    assert_eq!(a, c);
}

#[test]
fn mul_scalar() {
    let a = from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    let c = from_rows([[ 3,  6,  9], [12, 15, 18]]);
    assert_eq!(a * 3, c);
    assert_eq!(3 * a, c);

    let mut a = a;
    a *= 3;
    assert_eq!(a, c);
}

#[test]
fn div_scalar() {
    let a = from_rows([[ 3,  6,  9], [12, 15, 18]]);
    let c = from_rows([[ 1,  2,  3], [ 4,  5,  6]]);
    assert_eq!(a / 3, c);

    let mut a = a;
    a /= 3;
    assert_eq!(a, c);
}

#[test]
fn neg() {
    let a = from_rows([[-1.0,  2.0,  3.0], [ 4.0, -5.0,  6.0]]);
    let c = from_rows([[ 1.0, -2.0, -3.0], [-4.0,  5.0, -6.0]]);
    assert_eq!(-a, c);
}

#[test]
fn mul_matrix() {
    // 2x2 mul 2x2 -> 2x2
    assert_eq!(
        from_rows([[8, 9], [5, -1]]) * from_rows([[-2, 3], [4, 0]]),
        Mat2::from_rows([[20, 24], [-14, 15]]),
    );

    // 3x2 mul 2x3 -> 2x2
    assert_eq!(
        from_rows([[0, -1, 2], [4, 11, 2]]) * from_rows([[3, -1], [1, 2], [6, 1]]),
        from_rows([[11, 0], [35, 20]]),
    );

    // 2x3 mul 3x2 -> 3x3
    assert_eq!(
        from_rows([[3, -1], [1, 2], [6, 1]]) * from_rows([[0, -1, 2], [4, 11, 2]]),
        from_rows([[-4, -14, 4], [8, 21, 6], [4, 5, 14]]),
    );

    let a = from_rows([
        [ 0,  2,  4,  6],
        [ 1,  3,  5,  7],
        [17, 19, 21, 23],
        [25, 27, 29, 31],
    ]);
    let b = from_rows([
        [ 8, 10, 12, 14],
        [ 9, 11, 13, 15],
        [16, 18, 20, 22],
        [24, 26, 28, 30],
    ]);
    assert_eq!(a * b, from_rows([
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
fn iter_sum() {
    assert_eq!(
        <Vec<Matrix<f32, 4, 4>>>::new().into_iter().sum::<Matrix<f32, 4, 4>>(),
        <Matrix<f32, 4, 4>>::zero(),
    );

    let a = from_rows([[1, 2], [3, 4]]);
    let b = from_rows([[5, 6], [7, 8]]);
    assert_eq!(vec![a].into_iter().sum::<Mat2<i32>>(), a);
    assert_eq!(
        vec![a, b].into_iter().sum::<Mat2<i32>>(),
        from_rows([[6, 8], [10, 12]])
    );
}

#[test]
fn inv2() {
    assert_eq!(crate::Mat2f::identity().inverted(), Some(<Mat2<_>>::identity()));

    assert_eq!(
        <Mat2<_>>::from_diagonal([0.187, 6.5]).inverted(),
        Some(Matrix::from_diagonal([1.0 / 0.187, 1.0 / 6.5])),
    );

    assert_eq!(
        from_rows([
            [1.0, -1.0],
            [0.0,  2.0],
        ]).inverted(),
        Some(Matrix::from_rows([
            [1.0, 0.5],
            [0.0, 0.5],
        ])),
    );

    assert_eq!(
        from_rows([
            [ 1.0, 2.0],
            [-2.0, 1.0],
        ]).inverted(),
        Some(from_rows([
            [0.2, -0.4],
            [0.4,  0.2],
        ])),
    );
}

#[test]
fn non_invertible2() {
    assert_eq!(<Mat2<_>>::from_diagonal([0.0, 2.0]).inverted(), None);
    assert_eq!(
        from_rows([
            [1.0, 2.0],
            [2.0, 4.0],
        ]).inverted(),
        None,
    );
}

#[test]
fn inv3() {
    assert_eq!(crate::Mat3f::identity().inverted(), Some(<Mat3<_>>::identity()));

    assert_eq!(
        <Mat3<_>>::from_diagonal([0.187, 1.0, 6.5]).inverted(),
        Some(Matrix::from_diagonal([1.0 / 0.187, 1.0, 1.0 / 6.5])),
    );

    assert_eq!(
        from_rows([
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0],
        ]).inverted(),
        Some(from_rows([
            [-24.0,  18.0,  5.0],
            [ 20.0, -15.0, -4.0],
            [ -5.0,   4.0,  1.0],
        ])),
    );

    assert_eq!(
        from_rows([
            [3.0, 0.0,  2.0],
            [2.0, 0.0, -2.0],
            [0.0, 1.0,  1.0],
        ]).inverted(),
        Some(from_rows([
            [ 0.2,  0.2, 0.0],
            [-0.2,  0.3, 1.0],
            [ 0.2, -0.3, 0.0],
        ])),
    );
}

#[test]
fn non_invertible3() {
    assert_eq!(<Mat3<_>>::from_diagonal([1.5, 0.0, 2.0]).inverted(), None);
    assert_eq!(
        from_rows([
            [-1.0, 2.0,  0.0],
            [ 1.0, 2.0, -4.0],
            [ 1.0, 2.0, -4.0],
        ]).inverted(),
        None,
    );
}

#[test]
fn inv4() {
    assert_eq!(<Matrix<f32, 4, 4>>::identity().inverted(), Some(Matrix::identity()));

    assert_eq!(
        <Matrix<_, 4, 4>>::from_diagonal([0.187, 1.0, 4.0, 6.5]).inverted(),
        Some(Matrix::from_diagonal([1.0 / 0.187, 1.0, 1.0 / 4.0, 1.0 / 6.5])),
    );

    assert_approx_eq!(
        from_rows([
            [ 1.0,  4.0,  5.0, -1.0],
            [-2.0,  3.0, -1.0,  0.0],
            [ 2.0,  1.0,  1.0,  0.0],
            [ 3.0, -1.0,  2.0,  1.0],
        ]).inverted().unwrap(),
        from_rows([
            [-0.1, -0.1 ,  0.6 , -0.1],
            [ 0.0,  0.25,  0.25,  0.0],
            [ 0.2, -0.05, -0.45,  0.2],
            [-0.1,  0.65, -0.65,  0.9],
        ]);
        ulps <= 1
    );
}

#[test]
fn non_invertible4() {
    assert_eq!(<Matrix<_, 4, 4>>::from_diagonal([1.5, 0.0, 9.4, 2.0]).inverted(), None);
}
