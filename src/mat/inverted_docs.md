Returns the inverse of this matrix, if it exists. If the determinant of
this matrix is 0, then it is not invertible and `None` is returned. The
inverted matrix "undoes" the transformation of `self`, meaning that
`m * m.inverted()` is the identity matrix. The inverted matrix represents the
opposite/inverse transformation.
