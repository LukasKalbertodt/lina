
/// Zips two arrays, maps each pair and returns a new array with the result.
///
/// It's somewhat unfortunate that we have to implement this ourselves here.
/// Hopefully there will be more functionality surounding arrays in `std` so
/// that this helper can be removed. Luckily, this function seems to optimize
/// fine for small `N`s.
pub(crate) fn zip_map<A, B, F, R, const N: usize>(a: [A; N], b: [B; N], mut f: F) -> [R; N]
where
    F: FnMut(A, B) -> R,
{
    use std::{ptr, mem::{self, MaybeUninit}};

    /// We use this to drop already initialized elements in case `f` panics.
    struct Guard<R, const N: usize> {
        arr: [MaybeUninit<R>; N],
        initialized: usize,
    }

    impl<R, const N: usize> Drop for Guard<R, N> {
        fn drop(&mut self) {
            // We only drop all elements that have been initialized.
            unsafe {
                ptr::drop_in_place(
                    &mut self.arr[..self.initialized] as *mut [MaybeUninit<R>] as *mut [R]
                );
            }
        }
    }

    // This is a trick to get the uninitialied array below. I learned it from
    // the crate `inline-const`. Can be removed one `MaybeUninit::uninit_array`
    // is stabilized.
    struct InitDummy<R>(*const R);
    impl<R> InitDummy<R> {
        const INIT: MaybeUninit<R> = MaybeUninit::uninit();
    }

    let mut guard: Guard<R, N> = Guard {
        arr: [<InitDummy<R>>::INIT; N],
        initialized: 0,
    };

    // Actually do the zipping and mapping.
    for (a, b) in IntoIterator::into_iter(a).zip(b) {
        guard.arr[guard.initialized].write(f(a, b));
        guard.initialized += 1;
    }

    // At this point, the `guard.arr` is fully initialized. Unfortunately,
    // `MaybeUninit::array_assume_init` is still unstable and `mem::transmute`
    // does not work with dependently sized arrays yet. So we just use raw
    // pointer casts. This is all safe because `MaybeUninit<R>` has the same
    // memory layout as `R`.
    let out = unsafe { ptr::read(&guard.arr as *const [MaybeUninit<R>; N] as *const [R; N]) };

    // We have to forget the guard here: otherwise, it would drop all elements.
    mem::forget(guard);

    out
}

/// Creates an array from the given function that maps the index to an element.
pub(crate) fn array_from_index<T, const N: usize>(mut f: impl FnMut(usize) -> T) -> [T; N] {
    // We use the fact that `array::map` visits each element in order. That way,
    // we can advance an index.
    let mut i = 0;
    [(); N].map(|_| {
        let out = f(i);
        i += 1;
        out
    })
}


#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
    use super::*;

    struct DropTest {
        value: f32,
        count: Arc<AtomicU64>,
        dropped: bool,
    }

    impl DropTest {
        fn new(value: f32, count: Arc<AtomicU64>) -> Self {
            count.fetch_add(1, Ordering::SeqCst);
            Self {
                value,
                count,
                dropped: false,
            }
        }
    }

    impl Drop for DropTest {
        fn drop(&mut self) {
            println!("Dropping value {} now...", self.value);
            if self.dropped {
                panic!("attempt to drop value twice");
            }
            self.dropped = true;

            let prev = self.count.fetch_sub(1, Ordering::SeqCst);
            if prev == 0 {
                panic!("dropped more values than were initialized");
            }
        }
    }

    fn check_drop(f: impl FnOnce(&dyn Fn(f32) -> DropTest)) {
        let count = Arc::new(AtomicU64::new(0));
        f(&|value| DropTest::new(value, count.clone()));
        if count.load(Ordering::SeqCst) != 0 {
            panic!("Dropped too few elements");
        }
    }


    #[test]
    fn zip_map_drop() {
        check_drop(|new_value| {
            zip_map([], [], |a: f32, b: f32| new_value(a + b));
        });
        check_drop(|new_value| {
            zip_map([1.0], [7.0], |a: f32, b: f32| new_value(a + b));
        });
        check_drop(|new_value| {
            zip_map(
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                |a: i32, b: i32| new_value(a as f32 + b as f32),
            );
        });

    }

    /// Checks if panicking in the closure still drops the correct number of
    /// elements.
    #[test]
    fn zip_map_drop_panic() {
        let count = Arc::new(AtomicU64::new(0));

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            zip_map([1, 2, 3], [7, 8, 9], |a, b| {
                if a == 2 && b == 8 {
                    panic!("intended panic");
                } else {
                    DropTest::new(a as f32 + b as f32, count.clone())
                }
            });
        })).unwrap_err();

        let remaining = count.load(Ordering::SeqCst);
        if remaining != 0 {
            panic!("Dropped too few elements: {} remaining", remaining);
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            zip_map([1, 2, 3], [7, 8, 9], |a, b| {
                if a == 1 && b == 7 {
                    panic!("intended panic");
                } else {
                    DropTest::new(a as f32 + b as f32, count.clone())
                }
            });
        })).unwrap_err();

        let remaining = count.load(Ordering::SeqCst);
        if remaining != 0 {
            panic!("Dropped too few elements: {} remaining", remaining);
        }


        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            zip_map([1, 2, 3], [7, 8, 9], |a, b| {
                if a == 3 && b == 9 {
                    panic!("intended panic");
                } else {
                    DropTest::new(a as f32 + b as f32, count.clone())
                }
            });
        })).unwrap_err();

        let remaining = count.load(Ordering::SeqCst);
        if remaining != 0 {
            panic!("Dropped too few elements: {} remaining", remaining);
        }
    }
}
