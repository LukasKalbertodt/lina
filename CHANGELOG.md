# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

## 0.1.1 - 2021-10-24
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


[Unreleased]: https://github.com/LukasKalbertodt/lina/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/LukasKalbertodt/lina/compare/v0.1.0...v0.1.1
