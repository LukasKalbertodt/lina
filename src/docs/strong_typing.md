Explains the ways that `lina` uses strong typing: the distinction between
`Point` and `Vector`, the distinction between Cartesian and homogeneous
coordinates, and the `Space` parameter.

---

<br>
<br>

As you know, Rust is known for a strong type system and for nudging programmers
to make use of strong typing. It mainly means using different types for
different semantic values so that the compiler can catch logic errors for you.

`lina` tries to heavily lean into the strong typing philosophy via different
means.


## Location vs displacement: `Point` vs `Vector`

Think of `Point`s as an (absolute) *location* in space and of `Vector`s as a
(relative) *displacement* in space. There is a really nice analogy to two types
in the standard library:[`Instant`][std::time::Instant] (absolute location in
time) and [`Duration`][std::time::Duration] (relative displacement in time).

|   | In time | In space |
| - | ------- | -------- |
| Location | `Instant`, e.g. "2015-05-15 08:20:13", "now" | `Point`, e.g. "location of my tea cup" |
| Displacement | `Duration`, e.g. "3 minutes later", "8 days ago" | `Vector`, e.g. "5cm north" |

Now that we know about this semantic distinction, we can notice that some
operations only make sense between specific "kinds":

- Adding a *displacement* to a *location* makes sense and results in a *location*
    - "2015-05-15 08:20:13" + "3 minutes later" is "2015-05-15 08:23:13"
    - "location of my tea cup" + "5cm north" is a location within my living room
- Adding two *displacements* makes sense and results in a displacement
    - "4 hours later" + "15 minutes ago" is "3.75 hours later"
    - "5cm north" and "2m east" is "5cm north and 2m east"
- You can subtract two locations from one another and get a displacement
    - "2018-12-06 17:17:17" - "2015-05-15 08:20:13" represents the duration between
      those two instants: "3 years, 205 days, 8 hours, 57 minutes, 4 seconds"
    - "location of my tea cup" - "location of my right eye" results in a
       displacement, like "5cm up and 5cm north"
- Adding two *absolute locations* makes **no sense**.
    - The operation "2015-05-15 08:20:13" + "2018-12-06 17:17:17" makes no
      sense. Sure, we can numerically add 2015 and 2018 to get 4033, but that
      year has no semantic meaning.
    - Similarly, the operation "my living room" + "my favorite bakery" makes no sense.


Also see [this StackExchange Q&A](https://math.stackexchange.com/q/645672/340615).

However, while this distinction makes sense in most situations, in some cases it is less
important or gets a bit muddled. For example:

Locations can always be seen as a displacement relative to a specific reference
frame or origin. With the examples above (e.g. "location of tea cup"), if you
would want to express that position numerically, you would have to choose a
frame of reference (my house, planet earth, cosmic microwave background, ...).

Sometimes, a vector-like thing can be considered both, a location or
displacement. For example, let's say you have a 3D model of a fox. All those
vertex positions of the model are points in 3D space, i.e. a location. But now
you want to place that model in your 3D scene. The fox has a 3D position in
your world. But to arrive at the final vertex positions, you have two add two
positions?! Well, of course the model positions can be seen as a displacement
relative to an artificial model origin. Or the fox's position in your scene can
be seen as displacement from the scene origin.

For those reasons, `lina` allows you to easily cast between these to kinds via
[`Point::to_vec`] and [`Vector::to_point`]. Before using those, briefly think
whether what you are doing makes sense, though.


## Cartesian and homogeneous coordinates

As explained in [`HcMatrix`], homogeneous coordinates are used in order to
represent affine and projective transformations as matrices. While numerically
identical, there is a big semantic difference between a `Point<4>` (a location
in 4D space, represented with 4 cartesian coordinates) and `HcPoint<3>`
(a location in 3D space, represented with 4 homogeneous coordinates).
Similarly, a homogeneous transformation matrix is numerical identical to a
linear one of dimension N+1, but they represent vastly different things.

In most libraries, these things are not distinguished. And to be fair, for most
3D applications, the distinction is given by the dimension. If one never deals
with 4D linear transformations, for example, then `Mat4` is implicitly always a
homogeneous transformation matrix. However, it is still a bit unclean. And just
saying "both contain 16 floats, why have different types?" can be extended
to "`Vec<u8>` and `String` have the same memory layout, why have different
types?"

In the end, this is always an API design decision and a balance act between the
advantages and disadvantages of strong typing. `lina` made its choice and
hopefully, that will avoid some logic errors and make writing some code easier
(e.g. not needing to remember dividing by w).


## Semantic `Space` parameter

Almost all types in `lina` contain one or two `S: Space` parameters. For,
[`Point`], [`Vector`], [`HcPoint`], [`SphericalPos`] and
[`SphericalDir`], it indicates in which *semantic space* this point or
vector lives. Similarly, for [`Matrix`] and [`HcMatrix`], the two parameter
indicate from and into which semantic space the transformation transforms.

[The viewing pipeline docs][viewing_pipeline] show typical examples of different
spaces, i.e. model, world, view, and projection space. But apart from those,
there are actually many use cases for using multiple coordinate systems.

Let's assume you are developing a game where the player can walk around on a
planet. The natural coordinate system has the planet's center at `[0, 0, 0]`
with the north pole at `[0, 0, radius]`. But you also want to calculate the
motion of the sun and other objects in the sky. As early astronomers can
attest, calculating the orbits in a planet-centric coordinate system is ouch in
the bum. So for that, you want a different coordinate system with the sun at
the origin. Let's call that the helio-centric space or `HelioSpace`. The
planet-centric space you could call `PlanetSpace`.

But that's not all! Let's say you want a big planet. If it has the same radius
as earth, using `f32` gives you a half-meter precision at the surface. Oof.
Using `f64` is not an option unless you prefer to measure your game's
performance in spf (seconds/frame). So now you want a `DrawSpace`, a space that
is centered at the players spawn point. All rendering is done in that space.
(Let's hope the player never travels too far from the spawn point in one
session...)

In short: in typical 3D applications there are many different spaces that points
and vectors logically live in. Mixing points or vectors of different spaces
usually makes no sense. To avoid doing that, `lina` uses strong typing.

For the example from above, it's useful to add the following items to your
project:

```rust
enum HelioSpace {}
impl lina::Space for HelioSpace {}
enum PlanetSpace {}
impl lina::Space for PlanetSpace {}
enum DrawSpace {}
impl lina::Space for DrawSpace {}

type HelioPoint = lina::Point<f64, 3, HelioSpace>;
type PlanetPoint = lina::Point<f32, 3, PlanetSpace>;
type DrawPoint = lina::Point<f32, 3, DrawSpace>;
// ... more type aliases for vectors, directions, ...
```

These type aliases are useful as you can usually fix all generic parameters: the
dimension, scalar type and space! For example, the `HelioSpace` points never
reach the GPU and might require additional precision, so `f64` is fine.

It's worth noting that not only does this prevent switcheroos between
vectors/points of different spaces, but it can also help code readability and
automatically documents functions! Without spaces, you would have to include in
the function's docs in what space a vector is expected.

All of this unfortunately bloats lina's API quite a bit and makes using `lina` a
bit more annoying at some places. For example, `rustc` might ask for type
annotations. But in typical applications this shouldn't be a problem and
hopefully the added benefits are worth it.



[`Matrix`]: crate::Matrix
[`HcMatrix`]: crate::HcMatrix
[`Point`]: crate::Point
[`Vector`]: crate::Vector
[`HcPoint`]: crate::HcPoint
[`SphericalPos`]: crate::SphericalPos
[`SphericalDir`]: crate::SphericalDir
[`Vector::to_point`]: crate::Vector::to_point
[`Point::to_vec`]: crate::Point::to_vec
