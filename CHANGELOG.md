# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

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


[Unreleased]: https://github.com/LukasKalbertodt/lina/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/LukasKalbertodt/lina/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/LukasKalbertodt/lina/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/LukasKalbertodt/lina/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/LukasKalbertodt/lina/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/LukasKalbertodt/lina/compare/v0.1.0...v0.1.1
