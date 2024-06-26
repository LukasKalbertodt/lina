//! Transformation matrices for common transformations.
//!
//! - **Linear** transformations:
//!   - Scale: [`scale`] and [`scale_nonuniform`]
//!
//! - **Affine** transformations:
//!   - [`translate`]
//!   - [`look_into`] (world space to camera/view space; rotation + translation)
//!
//! - **Perspective projection**: [`perspective`]
//!

use std::ops::RangeInclusive;

use crate::{
    cross, dot,
    Float, Scalar, Vector, Matrix, Point3, Radians, Vec3, HcMatrix, HcMat3,
    WorldSpace, ViewSpace, ProjSpace, Space, Mat3, Dir3,
};


/// Linear transformation matrix that scales all `N` axis by `factor`.
///
/// Example for `Mat3` (with `f` being `factor`):
///
/// ```text
/// ⎡ f 0 0 ⎤
/// ⎢ 0 f 0 ⎥
/// ⎣ 0 0 f ⎦
/// ```
///
/// # Example
///
/// ```
/// use lina::{Mat3f, transform, vec3};
///
/// let m = transform::scale(3.5);
///
/// assert_eq!(m, Mat3f::from_rows([
///     [3.5, 0.0, 0.0],
///     [0.0, 3.5, 0.0],
///     [0.0, 0.0, 3.5],
/// ]));
/// assert_eq!(m.transform(vec3(1.0, 2.0, 3.0)), vec3(3.5, 7.0, 10.5));
/// ```
pub fn scale<T: Scalar, const N: usize>(factor: T) -> Matrix<T, N, N, WorldSpace, WorldSpace> {
    Matrix::from_diagonal([factor; N])
}

/// Linear transformation matrix that scales each axis according to `factors`.
///
/// Equivalent to [`Matrix::from_diagonal`]. Example for `Mat3` (with `factors`
/// being `[x, y, z]`):
///
/// ```text
/// ⎡ x 0 0 ⎤
/// ⎢ 0 y 0 ⎥
/// ⎣ 0 0 z ⎦
/// ```
///
/// # Example
///
/// ```
/// use lina::{Mat3f, transform, vec3};
///
/// let m = transform::scale_nonuniform([2.0f32, 3.0, 8.0]);
///
/// assert_eq!(m, Mat3f::from_rows([
///     [2.0, 0.0, 0.0],
///     [0.0, 3.0, 0.0],
///     [0.0, 0.0, 8.0],
/// ]));
/// assert_eq!(m.transform(vec3(10.0, 20.0, 30.0)), vec3(20.0, 60.0, 240.0));
/// ```
pub fn scale_nonuniform<T: Scalar, const N: usize>(
    factors: [T; N],
) -> Matrix<T, N, N, WorldSpace, WorldSpace> {
    Matrix::from_diagonal(factors)
}

/// 3D Rotation around the x-axis by `angle` (using the right-hand rule).
///
/// ```text
/// ⎡ 1       0        0 ⎤
/// ⎢ 0  cos(θ)  -sin(θ) ⎥
/// ⎣ 0  sin(θ)   cos(θ) ⎦
/// ```
pub fn rotate3d_around_x<T: Float>(
    angle: impl Into<Radians<T>>,
) -> Mat3<T, WorldSpace, WorldSpace> {
    let (sin, cos) = angle.into().sin_cos();
    Matrix::from_rows([
        [ T::one(), T::zero(), T::zero()],
        [T::zero(),       cos,      -sin],
        [T::zero(),       sin,       cos],
    ])
}

/// 3D Rotation around the y-axis by `angle` (using the right-hand rule).
///
/// ```text
/// ⎡  cos(θ)   0  sin(θ) ⎤
/// ⎢       0   1       0 ⎥
/// ⎣ -sin(θ)   0  cos(θ) ⎦
/// ```
pub fn rotate3d_around_y<T: Float>(
    angle: impl Into<Radians<T>>,
) -> Mat3<T, WorldSpace, WorldSpace> {
    let (sin, cos) = angle.into().sin_cos();
    Matrix::from_rows([
        [      cos, T::zero(),       sin],
        [T::zero(), T::one(),  T::zero()],
        [     -sin, T::zero(),       cos],
    ])
}

/// 3D Rotation around the z-axis by `angle` (using the right-hand rule).
///
/// ```text
/// ⎡ cos(θ)  -sin(θ)  0 ⎤
/// ⎢ sin(θ)   cos(θ)  0 ⎥
/// ⎣      0        0  1 ⎦
/// ```
pub fn rotate3d_around_z<T: Float>(
    angle: impl Into<Radians<T>>,
) -> Mat3<T, WorldSpace, WorldSpace> {
    let (sin, cos) = angle.into().sin_cos();
    Matrix::from_rows([
        [      cos,      -sin, T::zero()],
        [      sin,       cos, T::zero()],
        [T::zero(), T::zero(),  T::one()],
    ])
}

/// 3D Rotation around the given axis by `angle` (using the right-hand rule).
pub fn rotate3d_around<T: Float, S: Space>(
    axis: Dir3<T, S>,
    angle: impl Into<Radians<T>>,
) -> Mat3<T, S, S> {
    let (sin, cos) = angle.into().sin_cos();
    let omc = T::one() - cos;
    let [x, y, z] = axis.to_array();
    Matrix::from_rows([
        [x * x * omc + cos    , x * y * omc - z * sin, x * z * omc + y * sin],
        [y * x * omc + z * sin, y * y * omc + cos    , y * z * omc - x * sin],
        [z * x * omc - y * sin, z * y * omc + x * sin, z * z * omc + cos    ],
    ])
}

/// 3D rotation to align `from` with `to`, i.e. `M * from = to`.
///
/// Panics if `from == -to`, as in that case, the rotation is ambigious/not well
/// defined.
///
/// ```should_panic
/// use lina::{vec3, transform};
///
/// let d = vec3(1.0, 2.0, 0.0).to_dir();
/// let m = transform::rotate3d_aligning(d, -d);
/// ```
pub fn rotate3d_aligning<T: Float, S: Space>(
    from: Dir3<T, S>,
    to: Dir3<T, S>,
) -> Mat3<T, S, S> {
    // TODO: this is an exact check, and below there is another exact check to
    // catch more cases of this going wrong. But this should really use an
    // approx_eq test.
    assert!(from != -to, "`from == -to` in `rotate3d_aligning`");

    // Compare https://math.stackexchange.com/a/476311/340615
    let v = cross(from, to);
    let c = dot(from, to);
    let skew_mat = Mat3::from_rows([
        [T::zero(), -v.z, v.y],
        [v.z, T::zero(), -v.x],
        [-v.y, v.x, T::zero()],
    ]);

    let denom = T::one() + c;
    assert!(!denom.is_zero(), "`from == -to` in `rotate3d_aligning`");
    Mat3::identity() + skew_mat + (skew_mat * skew_mat) * (T::one() / denom)
}

/// Affine transformation matrix that translates according to `v`.
///
/// Example for `HcMat3` (with `v` being `[x, y, z]`):
///
/// ```text
/// ⎡ 1 0 0 x ⎤
/// ⎢ 0 1 0 y ⎥
/// ⎢ 0 0 1 z ⎥
/// ⎣ 0 0 0 1 ⎦
/// ```
///
/// # Example
///
/// ```
/// use lina::{HcMat3f, transform, vec3, point3};
///
/// let m = transform::translate(vec3(2.0, 3.0, 8.0));
///
/// assert_eq!(m, HcMat3f::from_rows([
///     [1.0, 0.0, 0.0, 2.0],
///     [0.0, 1.0, 0.0, 3.0],
///     [0.0, 0.0, 1.0, 8.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ]));
/// assert_eq!(m.transform(point3(10.0, 20.0, 30.0)), point3(12.0, 23.0, 38.0));
/// ```
pub fn translate<T: Scalar, const N: usize, S: Space>(
    v: Vector<T, N, S>,
) -> HcMatrix<T, N, N, S, S> {
    HcMatrix::from_parts(Matrix::identity(), v, Vector::zero(), T::one())
}

/// Affine transformation from world space into camera/view space, typically
/// called the "view matrix".
///
/// In view space, the camera is at the origin, +x points right, +y points up.
/// This view space is right-handed, and thus, +z points outside of the screen
/// and -z points into the screen. Please see [the part on handedness in the
/// these docs][hand-docs] for more information. The returned matrix only
/// translates and rotates, meaning that sizes and angles are unchanged
/// compared to world space.
///
/// The following camera properties have to be given:
/// - `eye`: the position of the camera.
/// - `direction`: the direction the camera is looking in. **Must not** be the
///   zero vector. Does not have to be normalized.
/// - `up`: a vector defining "up" in camera space. **Must not** be the zero
///   vector and **must not** be linearly dependent to `direction` (i.e. they
///   must not point into the same or exactly opposite directions). Does not
///   have to be normalized.
///
/// To avoid float precision problems, `direction` and `up` should *not* have
/// a tiny length and should *not* point in *almost* the same direction.
///
///
///
/// [hand-docs]: ../docs/viewing_pipeline/index.html#choice-of-view-space--handedness
///
/// ## A note on the `up` vector
///
/// There are two main types of cameras in games that are distinguished by
/// whether or not they can "fly a loop" (think airplane game with
/// out-of-cockpit camera).
///
/// In most games, this looping ability is not necessary: in those games, if you
/// move the mouse/controller all the way up or down, the camera stops turning
/// once you look almost straight down or up. Those are usually games with a
/// clear "down" direction (e.g. gravity). In these cases, you usually just
/// pass `(0, 0, 1)` (or `(0, 1, 0)` if you prefer your +y up) as `up` vector
/// and make sure the player cannot look exactly up or down. The latter you can
/// achieve by just having a min and max vertical angle (e.g. 1° and 179°).
///
/// In games where a looping camera is required, you have to maintain and evolve
/// the `up` vector over time. For example, if the player moves the
/// mouse/controller you don't just adjust the look direction, but also the up
/// vector.
pub fn look_into<T: Float, S: Space>(
    eye: Point3<T, S>,
    direction: Vec3<T, S>,
    up: Vec3<T, S>,
) -> HcMat3<T, S, ViewSpace> {
    // This function is unlikely to be called often, so we improve developer
    // experience by having these checks and normalizations.
    assert!(!direction.is_zero(), "direction vector must not be the zero vector");
    assert!(!up.is_zero(), "up vector must not be the zero vector");
    let dir = direction.normalized();

    let right = cross(dir, up);
    assert!(!right.is_zero(), "direction and up vector must be linearly independent");

    let r = right.normalized();
    let u = cross(r, dir);
    let d = dir;
    let eye = eye.to_vec();

    // This is the inverse of this:
    //
    // ⎡ r.x  u.x  d.x  eye.x ⎤
    // ⎢ r.y  u.y  d.y  eye.y ⎥
    // ⎢ r.z  u.z  d.z  eye.z ⎥
    // ⎣   0    0    0      1 ⎦
    let linear = Matrix::from_rows([
        [ r.x,  r.y,  r.z],
        [ u.x,  u.y,  u.z],
        [-d.x, -d.y, -d.z],
    ]);
    let translation = Vec3::new(-dot(eye, r), -dot(eye, u), dot(eye, d));
    HcMat3::from_parts(linear, translation, Vector::zero(), T::one())
}

/// Homogeneous transformation for *perspective* projection from view space to
/// NDC/projection space.
///
/// View space is assumed to be right-handed, i.e. +y pointing up and -z
/// pointing into the screen (satisfied by [`look_into`]). In NDC, `x/w` and
/// `y/w` are in range -1 to 1  and denote the horizontal and vertical screen
/// position, respectively. The +x axis points to the right, the +y axis points
/// up. `z` represents the depth and `z/w` is in range `depth_range_out`.
///
/// **Function inputs**:
///
/// - `vertical_fov`: the vertical field of view of your projection. Has to be
///   less than half a turn (π radians or 180°)!
///
/// - `aspect_ratio`: `width / height` of your target surface, e.g. screen or
///   application window. Has to be positive!
///
/// - `depth_range_in`: the near and far plane of the projection, in world or
///   view space (equivalent since the view matrix does not change distances).
///   The far plane may be ∞ (e.g. `f32::INFINITY`), which is handled properly
///   by this function.
///
/// - `depth_range_out`: the `z` range after the transformation. For `a..=b`,
///   `a` is what the near plane is mapped to and `b` is what the far plane is
///   mapped to. Usually, only the following values make sense:
///   - `0.0..=1.0` as default for WebGPU, Direct3D, Metal, Vulkan.
///   - `-1.0..=1.0` as default for OpenGL.
///   - `1.0..=0.0` for *reverse z* projection. Using this together with a
///     floating point depth buffer is **strongly recommended** as it vastly
///     improves depth precision.
///
///
/// # Example
///
/// (Don't use the finite near and far plane values in this example as "good
/// defaults" for your application. Use values fitting for your use case.)
///
/// ```
/// use lina::{Degrees, transform, HcMat3f};
///
/// let m = transform::perspective(Degrees(90.0), 2.0, 0.1..=f32::INFINITY, 1.0..=0.0);
/// assert_eq!(m, HcMat3f::from_rows([
///     [0.5, 0.0,  0.0, 0.0],
///     [0.0, 1.0,  0.0, 0.0],
///     [0.0, 0.0,  0.0, 0.1],
///     [0.0, 0.0, -1.0, 0.0],
/// ]));
///
/// let m = transform::perspective(Degrees(90.0), 1.0, 0.1..=100.0, 0.0..=1.0);
/// assert_eq!(m, HcMat3f::from_rows([
///     [1.0, 0.0,       0.0,        0.0],
///     [0.0, 1.0,       0.0,        0.0],
///     [0.0, 0.0, -1.001001, -0.1001001],
///     [0.0, 0.0,      -1.0,        0.0],
/// ]));
/// ```
///
///
/// # Adjustments for different view spaces or NDCs
///
/// Different spaces just require minor adjustments of the input or output
/// points. These adjustments can be represented as matrices, of course. And
/// since you can multiply them with the matrix returned by this function, you
/// end up with one correct matrix.
///
/// ## Left-handed view space (+z pointing into the screen)
///
/// Flip the `z` sign of all your points *before* transforming with this matrix.
///
/// ```
/// use lina::{Degrees, HcMat3f, transform, ViewSpace};
///
/// let flip_z_sign = <HcMat3f<ViewSpace, ViewSpace>>::from_diagonal([1.0, 1.0, -1.0, 1.0]);
/// let rh_proj_matrix = transform::perspective(Degrees(90.0), 2.0, 0.1..=100.0, 1.0..=0.0);
/// let lh_proj_matrix = flip_z_sign.and_then(rh_proj_matrix);
/// ```
///
/// ## +y pointing down in NDC (Vulkan)
///
/// Flip the `y` sign of all your points *after* transforming with this matrix.
///
/// ```
/// use lina::{Degrees, HcMat3f, transform, ProjSpace};
///
/// let flip_y_sign = <HcMat3f<ProjSpace, ProjSpace>>::from_diagonal([1.0, -1.0, 1.0, 1.0]);
/// let y_up_proj_matrix = transform::perspective(Degrees(90.0), 2.0, 0.1..=100.0, 1.0..=0.0);
/// let y_down_proj_matrix = y_up_proj_matrix.and_then(flip_y_sign);
/// ```
pub fn perspective<T: Float>(
    vertical_fov: impl Into<Radians<T>>,
    aspect_ratio: T,
    depth_range_in: RangeInclusive<T>,
    depth_range_out: RangeInclusive<T>,
) -> HcMat3<T, ViewSpace, ProjSpace> {
    let vertical_fov = vertical_fov.into();
    assert!(vertical_fov.0 < T::PI(), "`vertical_fov` has to be < π radians/180°");
    assert!(vertical_fov.0 > T::zero(), "`vertical_fov` has to be > 0");
    assert!(aspect_ratio > T::zero(), "`aspect_ratio` needs to be positive");

    let t = (vertical_fov / T::two()).tan();
    let sy = T::one() / t;
    let sx = sy / aspect_ratio;

    // We calculate the a and b elements of the matrix according to the
    // depth mapping (from `depth_range_in` to `depth_range_out`). This
    // article might help with understanding how these formulas came to be:
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    let (near_in, far_in) = depth_range_in.into_inner();
    let (near_out, far_out) = depth_range_out.into_inner();
    let (a, b);
    if far_in.is_infinite() {
        // Curiously, this does not depend on the sign of `far_in`.
        a = -far_out;
        b = -near_in * (far_out - near_out);
    } else {
        a = (far_in * far_out - near_in * near_out) / (near_in - far_in);
        b = (near_in * far_in * (far_out - near_out)) / (near_in - far_in);
    }

    //  ⎡ sx,   0,   0,   0 ⎤
    //  ⎢  0,  sy,   0,   0 ⎥
    //  ⎢  0,   0,   a,   b ⎥
    //  ⎣  0,   0,  -1,   0 ⎦
    let mut out = HcMat3::from_diagonal_parts([sx, sy, a], T::zero());
    out.set_elem(2, 3, b);
    out.set_elem(3, 2, -T::one());
    out
}

/// Homogeneous transformation for *orthographic* projection from view space to
/// NDC/projection space.
///
/// With orthographic projection, also called parallel projection, things
/// further from the camera don't get smaller. All "view rays" are parallel.
/// It's the limit of pulling the camera back and zooming in at the same time.
/// It's eseentially unsuitable for first-person cameras as we humans are not
/// really used to this kind of projection. Instead, its most common use in
/// games is for shadow mapping. But "top down" (e.g. strategy) games could
/// make use of this projection for the main camera as well.
///
/// View space is assumed to be right-handed, i.e. +y pointing up and -z
/// pointing into the screen (satisfied by [`look_into`]). In NDC, `x` and `y`
/// are in range -1 to 1 and denote the horizontal and vertical screen
/// position, respectively. The +x axis points to the right, the +y axis points
/// up. `z` represents the depth and is in range `depth_range_out`.
///
/// Unlike with [`perspective`], this matrix maps `z` linearly from
/// `depth_range_in` to `depth_range_out`. Consequently, you cannot use
/// infinity for any of those values. On the other hand, you can now pass 0 as
/// near plane. Also note that the "reverse depth + float buffer"-trick does
/// not make sense with this matrix; you likely want an integer depth buffer as
/// this provides constant precision.
///
/// This matrix maps points inside one axis aligned box to another. The
/// following table shows the dimensions of these boxes, which depend on the
/// arguments to this function:
///
/// |     | Input Box        | Output Box        |
/// | --- | ---------------- | ----------------- |
/// | `x` | `left..=right`   | `-1..1`           |
/// | `y` | `bottom..=top`   | `-1..1`           |
/// | `z` | `depth_range_in` | `depth_range_out` |
///
/// Note: if your camera (the input to the view matrix) is already at the center
/// of your projection, `left = -right` and `bottom = -top`. Having the
/// flexibility to set all these bounds independently means you don't have to
/// correctly position your camera, which can be convenient in some
/// situations.
///
/// For using this with a different view or NDC space, see [`perspective`].
///
/// # Example
///
/// ```
/// use lina::{Degrees, transform, HcMat3f};
///
/// let m = transform::ortho(-25.0, 25.0, -12.5, 12.5, 0.0..=100.0, 0.0..=1.0);
/// assert_eq!(m, HcMat3f::from_rows([
///     [0.04,  0.0,   0.0, 0.0],
///     [ 0.0, 0.08,   0.0, 0.0],
///     [ 0.0,  0.0, -0.01, 0.0],
///     [ 0.0,  0.0,   0.0, 1.0],
/// ]));
/// ```
pub fn ortho<T: Float>(
    left: T,
    right: T,
    bottom: T,
    top: T,
    depth_range_in: RangeInclusive<T>,
    depth_range_out: RangeInclusive<T>,
) -> HcMat3<T, ViewSpace, ProjSpace> {
    let (near_in, far_in) = depth_range_in.into_inner();
    let (near_out, far_out) = depth_range_out.into_inner();

    let zero = T::zero();
    let one = T::one();
    let two = T::two();

    let width = right - left;
    let height = top - bottom;
    let z_factor = -(far_out - near_out) / (far_in - near_in);
    let z_constant = near_out - (near_in * (far_out - near_out)) / (far_in - near_in);

    HcMat3::from_rows([
        [two / width,         zero,     zero, -(right + left) / width ],
        [       zero, two / height,     zero, -(top + bottom) / height],
        [       zero,         zero, z_factor,               z_constant],
        [       zero,         zero,     zero,                      one],
    ])
}

#[cfg(test)]
mod tests;
