
/// A semantic geometric space, e.g. model space, world space, view space.
///
/// Many types in this library have a generic parameter `S: Space`. This can be
/// used to treat points, for example, from different spaces as different
/// types. This is another way to let Rust help catching logic errors by using
/// strong typing.
///
/// For example, consider a 3D application where you load meshes from files. The
/// loaded points exist in "model space" where the object is usually centered
/// around the origin. To place it in the world/scene, you will usually
/// translate, rotate and scale the object. The transformed object (i.e. all
/// its points) now lives in the "world space". It would **not** make sense to
/// calculate the distance between a point in model space and a point in world
/// space. In fact, there is almost no operation that makes sense to deal with
/// points or vectors from two different spaces.
///
/// This trait is just a marker trait, not containing anything interesting. Taim
/// provides a few implementations that might be useful. But you are also
/// encouraged to create your own spaces if the provided ones don't fit your
/// use case. Adding a new space is super trivial. I would recommend also
/// making the space type uninhabited, i.e. `enum Name {}`.
///
/// Note that this trait does not necessarily represent any mathematical
/// concept. Yes, different `Space`s will usually have different basis-vectors,
/// for example. But understand this trait just as abstraction over "model
/// space", "world space", "view space" and others.
pub trait Space: 'static {}

/// A generic space without any semantics, used as default space.
///
/// This space's main purpose is to make using the `S` parameter of types
/// optional. Users who don't want to use the space strong typing feature, can
/// thus ignore it, just keeping all vectors and points in this generic space.
pub enum GenericSpace {}
impl Space for GenericSpace {}

/// A space that is model/object-local usually with a single object at the center.
///
/// Note that this has no special semantics in `lina` and is just provided for
/// your convenience, as this is a very common space one wants to distinguish.
/// The exact semantics are up to you.
pub enum ModelSpace {}
impl Space for ModelSpace {}

/// A space containing the whole scene/world with an arbitrary origin.
///
/// Note that this has no special semantics in `lina` and is just provided for
/// your convenience, as this is a very common space one wants to distinguish.
/// The exact semantics are up to you.
pub enum WorldSpace {}
impl Space for WorldSpace {}

/// A camera-centric space with the camera at the origin looking down an axis
/// (usually z).
///
/// Note that this has no special semantics in `lina` and is just provided for
/// your convenience, as this is a very common space one wants to distinguish.
/// The exact semantics are up to you.
pub enum ViewSpace {}
impl Space for ViewSpace {}

/// A post-projection space with angles and distances distorted.
///
/// Note that this has no special semantics in `lina` and is just provided for
/// your convenience, as this is a very common space one wants to distinguish.
/// The exact semantics are up to you.
pub enum ProjSpace {}
impl Space for ProjSpace {}
