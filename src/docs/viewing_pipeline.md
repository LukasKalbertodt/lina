How 3D objects get drawn to the screen through a series of transformations.
Foundation of real-time computer graphics (rasterization).

---

<br>
<br>

In your 3D application, you somehow have to draw your scene to the screen. There
are two main approaches: ray-tracing and rasterization. The former is generally
still too expensive to do in real-time, thus rasterization is the standard way
to render games and other interactive 3D applications.

Graphics APIs (like Vulkan, OpenGL, DirectX, WebGPU) basically require you to
transform every 3D point of your object into the so called *normalized device
coordinate system* (**NDC**). This is usually done in multiple steps, going
through a few different *coordinate systems* or *spaces* in this order:

- **Model space**: one object that has not been placed in your scene/world yet.
    For example, its center is usually around the origin (0, 0, 0) and is
    usually in upright position.

- **World space**: this is the most intuitive space and where most of your game
    logic will happen. Here, all objects have been placed in the world by
    moving (translating), rotating and scaling them.

- **View space**: an intermediate space where, by convention, the camera sits at
    the origin (0, 0, 0) and looks down the z axis (-z or +z, see below). This
    simplifies later calculations and makes some operations easier. The world
    has been translated and rotated, but not scaled or otherwise transformed.
    Angles from world space are preserved.

- **NDC**: this is very close to screen space, with `x` and `y` describing the
    horizontal and vertical position on your screen (or rather, the application
    window), respectively. The `z` axis points into or out of your screen and
    is used for depth testing and such things.

- **screen space**: 2D space with `x` and `y` being horizontal and vertical
    position in pixel units. Converting NDC to screen space is straight forward
    and is done internally by your graphics API.

*Aside*: what about **Clip space?** This coordinate system is essentially
the same as NDC, but coordinates are still homogeneous (i.e. 4D) to simplify
clipping. See [this answer](https://gamedev.stackexchange.com/a/65798/85787)
for more details.

Since the graphics APIs only start to interpret coordinates when you pass
them in NDC, everything before that is mostly up to you. The above spaces
are a good convention to stick to, though.


## Transforming between spaces with matrices inside shaders

In practice, you use matrices for almost all transformations from one space
to another. We do not discuss the *model → world* transformation here, as
this depends a lot on your application. The *world → view* and *view → NDC*
transformations are typically done like this:

You place a virtual camera (consisting of a position and look direction) in your
scene. You also have "global" properties like the field of view (FoV) you want
to render with. From those values, you create two matrices: the view matrix
(world → view, via [`transform::look_into`][look_into]) and the projection
matrix (view → NDC, via [`transform::perspective`][perspective]).

You pass both of those matrices to your shader as uniform value or push
constant. Inside the vertex shader, you extend your 3D vertex position with
a `w` coordinate with value 1, giving you a 4D vector representing
homogeneous coordinates in 3D space. Next, multiply that 4D vector with the
view matrix, then with the projection matrix, resulting in another 4D vector
whose `w` component might not be 1. This is what you "return" from the
shader (e.g. assign to `gl_Position`). Your graphics API will then perform
the "perspective divide" automatically: divide `x`, `y` and `z` by `w`,
which are the final 3D NDC coordinates.

It's also possible to pre-multiply both matrices in your application and
only pass the combined matrix to the shader. You can do that if you don't
perform any calculations in view space.


## NDC differences between APIs

Unfortunately, the exact properties/requirements of NDC depend on the
graphic API you are using. The common properties are:

- `x` and `y` are in range -1 to 1 (inclusive) and describe the horizontal
  and vertical position on the screen, respectively.
- The `+x` axis points from left to right.
- The `z` axis direction is not defined. It is used for depth-tests for
  example, but those can be configured for different comparisons (e.g. `<`
  vs. `>`). So this is up to the programmer.
- All points outside of the valid `x`, `y` or `z` ranges are clipped
  (i.e. basically removed) by default.

Differences concern the direction of the `y` axis and the `z` range.

| API | +y is... | z range |
| --- | -------- | ------- |
| WebGPU, Direct3D & Metal | ... up (`[-1, -1]` is bottom left corner) | 0 to 1 |
| OpenGL | ... up (`[-1, -1]` is bottom left corner) | -1 to 1 |
| Vulkan | ... down (`[-1, -1]` is top left corner) | 0 to 1 |

Since you are responsible for creating the point in NDC, your
view-projection matrix depends on the API you are using.

The above values are the defaults for the respective APIs. Some APIs might
allow you to configure these things. For example, in OpenGL, the widely
supported [`ARB_clip_control`][gl-clip-control] allows you to change the +y
direction and z-range.

[gl-clip-control]: https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_clip_control.txt


## Choice of view space & handedness

By convention, in view space, the camera sits at the origin, +x points to
the right, +y points up or down, +z points inside or outside the monitor.
In the majority of cases, +y points up in view space, so we will assume
that.

That leaves the choice of z-direction. Usually, -z points inside the screen
and +z points out of the screen. This is called a *right-handed coordinate
system* because you can use your right hand's thumb (+x), index finger
(+y) and middle finger (+z) at right angles to one another to represent the
axis of this space. The alternative is the left-handed coordinate system
where +z points inside the screen and -z points out of it.

What's important is that **the choice does not matter** as long as you use
an appropriate projection matrix in order to transform your points
correctly into the NDC of your graphics API. View and projection matrix
need to fit to one another and to your API's NDC. If you perform any
calculations in view space, you might merely need to know about the
handedness of your view space. But other than that, it's arbitrary.

[`transform::look_into`][look_into] returns a view matrix that transforms into
the right-handed view space. Similarly, [`transform::perspective`][perspective]
assumes a right-handed view space.

Left-handed versions of these functions are not offered in this library
because the choice is arbitrary and you can easily get a left-handed
version yourself. For the view matrix, just pass `-direction` as direction
to `look_into`. To get a projection matrix that works with a left-handed
view space, just flip the sign of your view space as a transformation
before the projection matrix. That means your projection matrix would be
`transform::perspective(...) * flip` where `flip` is this matrix (e.g. via
`Mat4::from_diagonal([1, 1, -1, 1])`:

```text
⎡ 1  0  0  0 ⎤
⎢ 0  1  0  0 ⎥
⎢ 0  0 -1  0 ⎥
⎣ 0  0  0  1 ⎦
```



[look_into]: crate::transform::look_into
[perspective]: crate::transform::perspective
