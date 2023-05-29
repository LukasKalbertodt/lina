use std::fmt;

use crate::Scalar;

mod inv4;
pub(crate) mod linear;
pub(crate) mod hc;


/// Helper to implement `fmt::Debug` for matrices.
fn debug_matrix_impl<T: Scalar>(
    f: &mut fmt::Formatter,
    cols: usize,
    rows: usize,
    elem: impl Fn(usize, usize) -> T,
) -> fmt::Result {
    /// Helper type to format a matrix row.
    struct DebugRow<F> {
        row_index: usize,
        cols: usize,
        elem: F,
    }

    impl<F: Fn(usize) -> T, T: fmt::Debug> fmt::Debug for DebugRow<F> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if f.alternate() {
                write!(f, "[")?;
            } else {
                write!(f, "row{} [", self.row_index)?;
            }

            for c in 0..self.cols {
                if c != 0 {
                    write!(f, ", ")?;
                }
                (self.elem)(c).fmt(f)?;
            }
            write!(f, "]")
        }
    }

    let mut list = f.debug_list();
    for r in 0..rows {
        list.entry(&DebugRow {
            row_index: r,
            cols,
            elem: |c| elem(r, c),
        });
    }
    list.finish()
}


#[cfg(test)]
mod tests;
