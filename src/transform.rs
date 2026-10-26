use crate::types::*;

pub fn translation(t: &Vec3) -> Mat4 {
    Mat4::new(
        1.0, 0.0, 0.0, t.x,
        0.0, 1.0, 0.0, t.y,
        0.0, 0.0, 1.0, t.z,
        0.0, 0.0, 0.0, 1.0,
    )
}

pub fn scale(s: &Vec3) -> Mat4 {
    Mat4::new(
        s.x, 0.0, 0.0, 0.0,
        0.0, s.y, 0.0, 0.0,
        0.0, 0.0, s.z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}

pub fn rotation_z(angle: f32) -> Mat4 {
    let cos_theta = angle.cos();
    let sin_theta = angle.sin();

    Mat4::new(
        cos_theta, -sin_theta, 0.0, 0.0,
        sin_theta, cos_theta, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
}
