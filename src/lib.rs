
mod vec;
mod util;
pub mod named_scalar;

pub use self::{
    vec::{
        point::{Point, Point2, Point2f, Point3, Point3f, point2, point3},
        vector::{Vector, Vec2, Vec2f, Vec3, Vec3f, Vec4, Vec4f, vec2, vec3, vec4},
    },
};
