

pub trait Zero {
    fn zero() -> Self;
}

pub trait One {
    fn one() -> Self;
}

macro_rules! impl_num {
    ($ty:ident, $zero:expr, $one:expr) => {
        impl Zero for $ty {
            fn zero() -> Self {
                $zero
            }
        }

        impl One for $ty {
            fn one() -> Self {
                $one
            }
        }
    };
}

impl_num!(f32, 0.0, 1.0);
impl_num!(f64, 0.0, 1.0);
impl_num!(u8, 0, 1);
impl_num!(u16, 0, 1);
impl_num!(u32, 0, 1);
impl_num!(u64, 0, 1);
impl_num!(u128, 0, 1);
impl_num!(i8, 0, 1);
impl_num!(i16, 0, 1);
impl_num!(i32, 0, 1);
impl_num!(i64, 0, 1);
impl_num!(i128, 0, 1);
