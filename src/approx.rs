//! Checking if two floating point numbers are approximately equal.
//!


use crate::{
    Vector, Scalar, Point, Radians, Float, Degrees, Matrix, SphericalPos, NormedSphericalPos,
};

/// Types that can be checked for approximate equality. Useful for floats.
///
/// The impls for vectors and other composite types simply check if all
/// components/elements are approximately equal. In particular, the distance
/// between two points is *not* used to check their approximate equality.
///
/// # Background
/// Float equality is an extremely tricky topic and lots has been written about
/// it. Please refer to [this][1] or [this][2] for detailed information. Very briefly:
///
/// - Exact comparisons rarely make sense as all calculations can introduce
///   rounding errors.
/// - There are three methods for approximate equality checks:
///    - *Absolute tolerance*: `(a - b).abs() < tolerance`.
///    - *Relative tolerance*: scale the tolerance by some number, usually the
///      maximum of the two inputs.
///    - *ULPs*: treat as equal if there are no more than a given number of floats
///      between the two inputs.
/// - Unfortunately, which method to use depends on the performed calculations
///   and the inputs.
/// - For unit tests it's probably fine to just pick any method and use the
///   lowest tolerance that makes the test pass.
///
/// Again: what we want is to "ignore the rounding errors". Most of the time,
/// the magnitude of the rounding error is proportional to the magnitude of the
/// number itself. In these cases, using ULPs or a relative tolerance works
/// well. However, sometimes the error is not propertional to the number'S
/// magnitude, which is the tricky part. In those cases, you have to estimate
/// the error manually from the inputs and intermediate results.
///
/// So to decide what method to use, you have to estimate the error of your
/// number(s). Imagine that each float is carrying a second hidden value with
/// it: the expected error. Let's look at some operations:
///
/// - **Real to float**: converting any (exact) decimal number into a float
///     requires rounding most of the time. The closest representable float is
///     picked, and since the gaps between representable floats are proportional
///     to the floats magnitude, the error is too.
/// - **Multiplication/division**: generally, this scales the error nicely, such
///     that the error of the result is also proportional to the magnitude of
///     the result.
/// - **Addition/subtraction**: this is tricky as it can destroy the error size
///     proportionality. Imagine two huge floats with errors proportional to
///     their magnitude and having almost the same value (e.g. being just a few
///     representable floats apart). Subtracting one from the other gives a
///     number close to zero, as the numbers very almost the same. However, the
///     error is still propertional to the inputs, and not to the result! For
///     example, the result of `123456789.0f32 - 123456788.0f32` is `8.0`! This
///     is also what's known as [catastrophic cancellation][3].
///
/// Of course there are many more operations that one needs to think about.
///
///
/// [1]: https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
/// [2]: https://jtempest.github.io/float_eq-rs/book/introduction.html
/// [3]: https://en.wikipedia.org/wiki/Catastrophic_cancellation
pub trait ApproxEq {
    /// Type to specify the tolerance.
    type Tolerance;

    /// Checks if `self` is approximately equal to `other` with the given
    /// absolute tolerance. In other words: this method returns `true` if
    /// `self` and `other` are no more than `abs_tolerance` apart.
    ///
    /// Do *not* just pass `f32::EPSILON` and call it a day. 40% of all
    /// floats are smaller than `f32::EPSILON`! Said constant is only
    /// appropriate when scaled.
    fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool;

    /// Checks if `self` is approximately equal to `other` by allowing up to
    /// `max_steps_apart` many representable float values between the numbers.
    /// Passing 0 makes this equivalent to `==`. Numbers with different signs
    /// are always different (except for 0).
    fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool;

    /// Checks if `self` is approximately equal to `other` with the given
    /// relative tolerance, which is scaled by the maximum of the inputs'
    /// magnitudes.
    ///
    /// Often, you would pass a small integer multiple of `f32::EPSILON` as
    /// tolerance.
    fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool;
}

macro_rules! impl_for_float {
    ($ty:ident) => {
        impl ApproxEq for $ty {
            type Tolerance = Self;

            fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
                self == other || (self - other).abs() <= abs_tolerance
            }

            fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
                // Deal with infinities and zeroes.
                if self.is_sign_positive() != other.is_sign_positive() {
                    return self == other;
                }

                let max = Self::max(self.abs(), other.abs());
                Self::approx_eq_abs(self, other, rel_tolerance * max)
            }

            fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
                if self.is_nan() || other.is_nan() {
                    return false;
                }

                if self.is_sign_positive() != other.is_sign_positive() {
                    return self == other;
                }

                self.to_bits().abs_diff(other.to_bits()) <= max_steps_apart.into()
            }
        }
    };
}

impl_for_float!(f32);
impl_for_float!(f64);


macro_rules! impl_for_vec_point {
    ($ty:ident) => {
        impl<T: ApproxEq<Tolerance = T> + Scalar, const N: usize> ApproxEq for $ty<T, N> {
            type Tolerance = T;

            fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
                (0..N).all(|i| T::approx_eq_abs(self[i], other[i], abs_tolerance))
            }

            fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
                (0..N).all(|i| T::approx_eq_rel(self[i], other[i], rel_tolerance))
            }

            fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
                (0..N).all(|i| T::approx_eq_ulps(self[i], other[i], max_steps_apart))
            }
        }
    };
}

impl_for_vec_point!(Vector);
impl_for_vec_point!(Point);

macro_rules! impl_for_angles {
    ($ty:ident) => {
        impl<T: ApproxEq<Tolerance = T> + Float> ApproxEq for $ty<T> {
            type Tolerance = Self;

            fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
                T::approx_eq_abs(self.0, other.0, abs_tolerance.0)
            }

            fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
                T::approx_eq_rel(self.0, other.0, rel_tolerance.0)
            }

            fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
                T::approx_eq_ulps(self.0, other.0, max_steps_apart)
            }
        }
    };
}

impl_for_angles!(Radians);
impl_for_angles!(Degrees);

impl<T, const C: usize, const R: usize> ApproxEq for Matrix<T, C, R>
where
    T: ApproxEq<Tolerance = T> + Scalar,
{
    type Tolerance = T;

    fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| T::approx_eq_abs(a, b, abs_tolerance))
    }

    fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| T::approx_eq_rel(a, b, rel_tolerance))
    }

    fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| T::approx_eq_ulps(a, b, max_steps_apart))
    }
}

impl<T: ApproxEq<Tolerance = T> + Float> ApproxEq for SphericalPos<T> {
    type Tolerance = T;

    fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
        let c = |a, b| T::approx_eq_abs(a, b, abs_tolerance);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0) && c(self.r, other.r)
    }

    fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
        let c = |a, b| T::approx_eq_rel(a, b, rel_tolerance);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0) && c(self.r, other.r)
    }

    fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
        let c = |a, b| T::approx_eq_ulps(a, b, max_steps_apart);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0) && c(self.r, other.r)
    }
}

impl<T: ApproxEq<Tolerance = T> + Float> ApproxEq for NormedSphericalPos<T> {
    type Tolerance = T;

    fn approx_eq_abs(self, other: Self, abs_tolerance: Self::Tolerance) -> bool {
        let c = |a, b| T::approx_eq_abs(a, b, abs_tolerance);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0)
    }

    fn approx_eq_rel(self, other: Self, rel_tolerance: Self::Tolerance) -> bool {
        let c = |a, b| T::approx_eq_rel(a, b, rel_tolerance);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0)
    }

    fn approx_eq_ulps(self, other: Self, max_steps_apart: u32) -> bool {
        let c = |a, b| T::approx_eq_ulps(a, b, max_steps_apart);
        c(self.theta.0, other.theta.0) && c(self.phi.0, other.phi.0)
    }
}


/// Helper macro to get nicer error messages in tests.
#[cfg(test)]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr; $mode:ident <= $v:expr) => {{
        let a = $a;
        let b = $b;
        if !$crate::approx::assert_approx_eq!(@imp a, b; $mode <= $v) {
            panic!(
                "assert_approx_eq failed!\n\
                    left:  {:#?}\n\
                    right: {:#?}\n",
                a,
                b,
            );
        }
    }};
    (@imp $a:ident, $b:ident; abs <= $v:expr) => {
        $crate::approx::ApproxEq::approx_eq_abs($a, $b, $v)
    };
    (@imp $a:ident, $b:ident; rel <= $v:expr) => {
        $crate::approx::ApproxEq::approx_eq_rel($a, $b, $v)
    };
    (@imp $a:ident, $b:ident; ulps <= $v:expr) => {
        $crate::approx::ApproxEq::approx_eq_ulps($a, $b, $v)
    };
}

#[cfg(test)]
pub(crate) use assert_approx_eq;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_values() {
        assert!(ApproxEq::approx_eq_abs(0.0, -0.0, 0.0));
        assert!(ApproxEq::approx_eq_rel(0.0, -0.0, 0.0));
        assert!(ApproxEq::approx_eq_ulps(0.0, -0.0, 0));

        assert!(!ApproxEq::approx_eq_abs(f32::NEG_INFINITY, f32::INFINITY, 100.0));
        assert!(!ApproxEq::approx_eq_rel(f32::NEG_INFINITY, f32::INFINITY, 1.0));
        assert!(!ApproxEq::approx_eq_ulps(f32::NEG_INFINITY, f32::INFINITY, 10));

        assert!(!ApproxEq::approx_eq_abs(3.14, f32::NAN, 100.0));
        assert!(!ApproxEq::approx_eq_rel(3.14, f32::NAN, 1.0));
        assert!(!ApproxEq::approx_eq_ulps(3.14, f32::NAN, 10));
        assert!(!ApproxEq::approx_eq_abs(f32::NAN, 3.14, 100.0));
        assert!(!ApproxEq::approx_eq_rel(f32::NAN, 3.14, 1.0));
        assert!(!ApproxEq::approx_eq_ulps(f32::NAN, 3.14, 10));
    }

    #[test]
    fn floats() {
        assert!(!ApproxEq::approx_eq_abs(0.6000000000000001, 0.6, 0.0));
        assert!( ApproxEq::approx_eq_abs(0.6000000000000001, 0.6, 0.0000000000000002));
        assert!(!ApproxEq::approx_eq_rel(0.6000000000000001, 0.6, 0.0));
        assert!( ApproxEq::approx_eq_rel(0.6000000000000001, 0.6, f64::EPSILON));
        assert!(!ApproxEq::approx_eq_ulps(0.6000000000000001, 0.6, 0));
        assert!( ApproxEq::approx_eq_ulps(0.6000000000000001, 0.6, 1));
    }
}
