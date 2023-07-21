# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

## [0.2.1] - 2023-07-21

- Added `transform::ortho`

## [0.2.0] - 2023-06-11

**Summary**: lots has changed, including lots of breaking changes.
Luckily all breaking changes will be caught by the compiler, there are no behavior changes.
The main two changes two keep in mind: almost all types have a `Space` parameter now;
and `HcPoint` and `HcMatrix` have been added, which replace situations where previously, `Point4` or `Mat4` were used.

- **Breaking**: Add `Space` parameter to `Point`, `Vector`, `Matrix`, `SphericalPos`, `SphericalDir` and many functions
- **Breaking**: change `clamp(v, min, max)` to `clamp(v, min..=max)`
- **Breaking**: Remove `f64` type aliases
- **Breaking**: Remove type aliases for 4D vectors and matrices (you likely want to use `Hc*` types instead)
- **Breaking**: remove `Vec4::new`, `vec4`
- **Breaking**: Remove `nightly` feature. Crate always compiles on stable now. Functions that required `nightly` previously are now simply implemented for a small number of fixed dimensions.
- **Breaking**: Rename `NormedSphericalPos` to `SphericalDir`
- Add `HcPoint` and `HcMatrix`
- Add `Dir` for representing directions with a unit vector
- Add four `transform::rotate3d_*` rotation matrices: around x, y, z, or a given axis.
- Add `to_f32` and `to_f64` to point and vector types for simpler casting
- Add `ApproxEq` trait and `assert_approx_eq!` macro
- **Breaking**: Add `ApproxEq` as supertrait to `Float`
- Add `Row`/`Col` proxy types and `Matrix::{row, col}`
- Add `Matrix::{elem, set_elem}` and remove `Index[Mut]` impls
- Improve `Debug` impl of matrices by printing `rowN` prefix in single line mode
- **Breaking**: Change `Matrix::*diagonal` to use `[T; N]` instead of `Vector<T, N>`
- **Breaking**: Change some matrix methods to take `&self` instead of `self` (`iter`, `map`, `zip_map`, `determinant`, `inverted`, `transposed`)
- **Breaking**: only implement `Eq` for containers (vector, point, ...) if the scalar implements `Eq`
- `SphericalPos::from(Vector::null())` now returns `(0°, 0°, 0)` instead of `NaN`s
- Add `sin_cos` for `Degrees` and `Radians`
- Add `Point::vec_to(other_point)` as alternative to subtraction
- Add `angle / angle -> T` operator overload (for `Degrees` and `Radians`)


## [0.1.5] - 2023-05-23
- Add `inverted` and `determinant` to 1x1, 2x2, 3x3 and 4x4 matrices
- Add some `scalar * Degrees` impls for primitive scalars
- Add `iter` method to `Vector`, `Point` and `Matrix`
- Add `to_array` method to `Vector` and `Point` (this is more convenient to use
  than the `Into` impl in some situations)
- Remove stray `dbg!` from `Matrix::transform_hc_vec`

## [0.1.4] - 2023-04-17
- Guard all features requiring nightly behind `nightly` feature gate.
  This makes it possible to compile `lina` on stable.

## [0.1.3] - 2023-04-04
- Remove stabilized feature flag for `array_from_fn`
- Switch to Rust 2021
- Fix some clippy lints
- Fix link in docs
- Fix CI badge in README

## [0.1.2] - 2021-11-13
### Fixed
- Fix `angle_between` for super small values
- Fix `slerp` for input vectors with very small angle between them

## [0.1.1] - 2021-10-24
### Added
- `slerp` function for spherical linear interpolation
- `float * angle` operator impls for `f32` and `f64` (float on the left hand side)
- `scalar * matrix` operator impls for primitive types (scalar on the left hand side)
- `Vector::average` (like `Point::centroid`)
- `std::iter::Sum<Self>` impl for `Vector` and `Matrix`

### Changed
- Clarify documentation about angle ranges of spherical coordinates (phi can be in `-π..=π`)
- Add `debug_assert`s to `angle_between`


## 0.1.0 - 2021-10-17
### Added
- Everything


[Unreleased]: https://github.com/LukasKalbertodt/lina/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/LukasKalbertodt/lina/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/LukasKalbertodt/lina/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/LukasKalbertodt/lina/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/LukasKalbertodt/lina/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/LukasKalbertodt/lina/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/LukasKalbertodt/lina/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/LukasKalbertodt/lina/compare/v0.1.0...v0.1.1
