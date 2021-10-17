Explains the distinction between `Point` and `Vector`.

---

<br>
<br>

As you know, Rust is known for a strong type system and for nudging programmers
to make use of strong typing. It mainly means using different types for
different semantic values so that the compiler can catch logic errors for you.

`lina` makes use of strong typing by having two separate vector-like types:
`Vector` and `Point`.


## Location versus displacement

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


## API design: a trade-off, not perfection

I hope you agree that, in most situations, it makes sense to draw the semantic
distinction between locations and displacements. For that reason, `lina` has
two types for these two different use cases. *However*, this is just an API
decision and by no means the only right way to go about things. Consider these
things:

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
relative to an artificial model origin.

On the other hand, you could find a lot more useful semantic distinctions. One
of the most relevant ones for `lina` are homogeneous coordinates. They are
fundamentally different from a point in 4D space, but we happily use `Vec4` to
represent them just because both have 4 scalars. This might lead to us
performing operations (e.g. transforming via matrix) on a `Vec4` that don't
make sense. In fact, `lina` might still introduce separate types for points in
homogeneous space in the future. But of course, you don't want to have a type
for each imaginable semantic meaning, e.g. because it would bloat the API.

In the end, in my experience, the distinction of `Vector` and `Point` has more
advantages than disadvantages. It's not always perfect, but better than only
one vector type, in my opinion.
