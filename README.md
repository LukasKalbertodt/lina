# `lina`: linear algebra library for 3D applications

[<img alt="CI status of main" src="https://img.shields.io/github/workflow/status/LukasKalbertodt/lina/CI/main?label=CI&logo=github&logoColor=white&style=for-the-badge" height="23">](https://github.com/LukasKalbertodt/lina/actions?query=workflow%3ACI+branch%3Amaster)
[<img alt="Crates.io Version" src="https://img.shields.io/crates/v/lina?logo=rust&style=for-the-badge" height="23">](https://crates.io/crates/lina)
[<img alt="docs.rs" src="https://img.shields.io/crates/v/lina?color=blue&label=docs&style=for-the-badge" height="23">](https://docs.rs/lina)


`lina` is yet another linear algebra library with a focus on 3D applications
like games, i.e. low-dimensional vectors and matrices.

One special feature is the heavy use of const generics to make vectors, points
and matrices generic over their dimensions, while still allowing scalar access
via `.x`, `.y`, `.z` and `.w`. This has several advantages like a reduced API
surface and easier to understand [docs](https://docs.rs/lina).

**Features**:

- Vectors and points (strongly typed *locations* in space) with generic dimension and scalar type
- Matrices with generic dimensions and element type
- Strongly typed angles: `Degrees` and `Radians`
- Commonly used transformation matrices
- Spherical coordinates: `SphericalPos` and `NormedSphericalPos`
- Several helper functions: `atan2`, `clamp`, `lerp`, ...
- Auxiliary documentation about topics like computer graphics, linear algebra, ...

**Still missing** (but planned):

- Rotors and everything related to rotations
- Matrix inversion & determinants
- SIMD
    - Using SIMD is currently not feasible as there is no way to specify
      alignments for the generic `Point`, `Vector` and `Matrix` types. If Rust
      ever offers more alignment control, I will revisit SIMD.
    - But: this might not matter too much as long as you do not perform lots of
      operations on the CPU. The GPU is better at it anyway!


See [**the documentation**](https://docs.rs/lina) for more information.


## Why yet another of these libraries?!

What about
[`cgmath`](https://crates.io/crates/cgmath),
[`nalgebra`](https://nalgebra.org/),
[`glam`](https://crates.io/crates/glam/),
[`ultraviolet`](https://crates.io/crates/ultraviolet)
[`vec`](https://crates.io/crates/vek), ...?
Those are all fine libraries, but I was not 100% happy with either of them.
So I wrote my own!
`lina` is not *better* than these other libraries, just has a different API that better fits my taste.


## Status of this project

`lina` is still young, but I already use it in a project of mine.
For the foreseeable future, I will shape `lina` exactly to *my* liking for use in said project.
But of course, I'm happy about feedback, suggestions and people finding use in `lina`.
Pull requests are generally welcome, but please talk to me first as I might just decline PRs that don't fit my vision for `lina`.


<br />

---

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
