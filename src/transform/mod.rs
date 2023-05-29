//! Transformation matrices for common transformations.
//!
//! - **Linear** transformations:
//!   - Scale: [`scale_cc`], [`scale_hc`], [`scale_nonuniform_cc`], [`scale_nonuniform_hc`]
//!
//! - **Affine** transformations:
//!   - [`translate`]
//!   - [`look_into`] (world space to camera/view space; rotation + translation)
//!
//! - **Perspective projection**: [`perspective`]
//!
//!
//! # Cartesian and Homogeneous coordinates
//!
//! An N×N matrix can only represent *linear* transformations in N-dimensional
//! space. To represent other transformations with matrices, homogeneous
//! coordinates are used. This requires us to add one dimension to our matrix:
//! an N×N matrix can now represent linear, affine and perspective
//! transformations in N − 1 dimensional space.
//!
//! In this library, a *homogenous transformation matrix* describes a matrix
//! that is intended to transform vectors in homogeneous coordinates. In
//! contrast, a *linear transformation matrix* is one that is intended to
//! transform vectors in standard, cartesian coordinates.
//!
//! All functions for linear transformations have two versions: `_cc` for
//! cartesian coordinates and `_hc` for homogeneous coordinates. Using
//! homogeneous coordinates for linear transformations is not necessary, except
//! if you want to combine it with other non-linear transformations. That's
//! what the `_hc` versions are for.

use std::ops::RangeInclusive;

use crate::{
    Float, Vector, Matrix, Point3, Radians, Scalar, Vec3, cross, dot, HcMatrix, HcMat3, vec3,
};


/// *Linear* transformation matrix that scales all `N` axis by `factor`.
///
/// For the homogeneous coordinate version, see [`scale_hc`]. Example for `Mat3`
/// (with `f` being `factor`):
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
pub fn scale<T: Scalar, const N: usize>(factor: T) -> Matrix<T, N, N> {
    Matrix::from_diagonal([factor; N])
}

/// *Linear* transformation matrix that scales each axis according to `factors`.
///
/// For the homogeneous coordinate version, see [`scale_nonuniform_hc`].
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
) -> Matrix<T, N, N> {
    Matrix::from_diagonal(factors)
}


/// *Homogeneous* transformation matrix that translates according to `v`.
///
/// Example for `Mat4` (with `v` being `[x, y, z]`):
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
pub fn translate<T: Scalar, const N: usize>(v: Vector<T, N>) -> HcMatrix<T, N, N> {
    HcMatrix::from_parts(Matrix::identity(), v, Vector::zero(), T::one())
}

/// *Homogeneous* transformation from world space into camera/view space.
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
pub fn look_into<T: Float>(eye: Point3<T>, direction: Vec3<T>, up: Vec3<T>) -> HcMat3<T> {
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
    // ⎢ r.y  u.x  d.x  eye.x ⎥
    // ⎢ r.z  u.x  d.x  eye.x ⎥
    // ⎣   0    0    0      1 ⎦
    let linear = Matrix::from_rows([
        [ r.x,  r.y,  r.z],
        [ u.x,  u.y,  u.z],
        [-d.x, -d.y, -d.z],
    ]);
    let translation = vec3(-dot(eye, r), -dot(eye, u), dot(eye, d));
    HcMat3::from_parts(linear, translation, Vector::zero(), T::one())
}

/// *Homogeneous* transformation for perspective projection from view space to
/// NDC.
///
/// Note that unlike most other homogeneous transformation matrices, this matrix
/// does not necessarily keep `w = 1` in transformed vectors. So you might need
/// to devide the transformed vector by `w`, also called the "perspective
/// divide". You can use [`Matrix::transform_hc_vec`] or
/// [`Matrix::transform_hc_point`] to perform that divide for you.
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
/// use lina::{Degrees, HcMat3f, transform};
///
/// let flip_z_sign = HcMat3f::from_diagonal_parts([1.0, 1.0, -1.0], 1.0);
/// let rh_proj_matrix = transform::perspective(Degrees(90.0), 2.0, 0.1..=100.0, 1.0..=0.0);
/// let lh_proj_matrix = flip_z_sign.and_then(rh_proj_matrix);
/// ```
///
/// ## +y pointing down in NDC (Vulkan)
///
/// Flip the `y` sign of all your points *after* transforming with this matrix.
///
/// ```
/// use lina::{Degrees, HcMat3f, transform};
///
/// let flip_y_sign = HcMat3f::from_diagonal_parts([1.0, -1.0, 1.0], 1.0);
/// let y_up_proj_matrix = transform::perspective(Degrees(90.0), 2.0, 0.1..=100.0, 1.0..=0.0);
/// let y_down_proj_matrix = y_up_proj_matrix.and_then(flip_y_sign);
/// ```
pub fn perspective<T: Float>(
    vertical_fov: impl Into<Radians<T>>,
    aspect_ratio: T,
    depth_range_in: RangeInclusive<T>,
    depth_range_out: RangeInclusive<T>,
) -> HcMat3<T> {
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

#[cfg(test)]
mod tests;
