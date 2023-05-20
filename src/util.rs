use std::fmt;

pub(crate) fn debug_list_one_line<T: fmt::Debug>(
    list: &[T],
    f: &mut fmt::Formatter,
) -> fmt::Result {
    write!(f, "[")?;
    for (i, e) in list.iter().enumerate() {
        if i != 0 {
            write!(f, ", ")?;
        }
        e.fmt(f)?;
    }
    write!(f, "]")
}
