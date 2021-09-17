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
