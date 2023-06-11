# `lina`: linear algebra library for 3D applications

[<img alt="CI status of main" src="https://img.shields.io/github/actions/workflow/status/LukasKalbertodt/lina/ci.yml?branch=main&label=CI&logo=github&logoColor=white&style=for-the-badge" height="23">](https://github.com/LukasKalbertodt/lina/actions/workflows/ci.yml)
[<img alt="Crates.io Version" src="https://img.shields.io/crates/v/lina?logo=rust&style=for-the-badge" height="23">](https://crates.io/crates/lina)
[<img alt="docs.rs" src="https://img.shields.io/crates/v/lina?color=blue&label=docs&style=for-the-badge" height="23">](https://docs.rs/lina)


`lina` is a linear algebra library making heavy use of strong typing.
Its focus is on 3D applications like games, i.e. low-dimensional vectors and matrices.

Notable features setting `lina` apart from other similar libraries:

- Usage of const generics, while still allowing scalar access via `.x`, `.y`, `.z` and `.w`.
- Separate types for homogeneous coordinates (`HcPoint` and `HcMatrix`).
- Most types have a `Space` parameter to represent the *logical space* (e.g. model, world, view, ... space) the vector, point, etc lives in.
- Distinction between locations (`Point`) and displacements (`Vector`).

The last three of these illustrate the philosophy of `lina` regarding strong typing.
For motivation and examples, please read [this document](https://docs.rs/lina/latest/lina/docs/strong_typing/).
In fact, this is all still a bit of an experiment, but so far I am very happy with the results in practice!
However, I'm sure this is not for everyone and many would prefer a different API.

Luckily, there exist many other libraries in the Rust ecosystem.
To be clear: `lina` is not *better* than
[`cgmath`](https://crates.io/crates/cgmath),
[`nalgebra`](https://nalgebra.org/),
[`glam`](https://crates.io/crates/glam/),
[`ultraviolet`](https://crates.io/crates/ultraviolet)
[`vek`](https://crates.io/crates/vek), etc.
It is simply different, with an API that better fits my taste – and maybe yours.


**Additional Features**:

- Vectors, points, matrices
- Commonly used transformation matrices
- Operators overloaded as you would expect
- Strongly typed angles: `Degrees` and `Radians`
- Spherical coordinates: `SphericalPos` and `SphericalDir`
- Several helper functions: `atan2`, `clamp`, `lerp`, `slerp`, ...
- Approximate float equality (including `assert_approx_eq!`)

The only major thing that I'd still like to add in the future is rotors.
`lina` offers the standard translation matrices, but rotors can be better for representing and composing rotations.


See [**the documentation**](https://docs.rs/lina) for more information.


## Status of this project

`lina` is is certainly usable and is quite feature rich.
I actively use it in a game project, which also motivated most API design decisions.
But `lina` is not used by many other projects, and there might still be some non-minor API changes.



<br />

---

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
