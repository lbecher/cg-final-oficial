pub struct VectorInputData {
    pub x: String,
    pub y: String,
    pub z: String,
}

impl Default for VectorInputData {
    fn default() -> Self {
        Self {
            x: "X: 0".to_string(),
            y: "Y: 0".to_string(),
            z: "Z: 0".to_string(),
        }
    }
}

impl VectorInputData {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: format!("X: {}", x),
            y: format!("Y: {}", y),
            z: format!("Z: {}", z),
        }
    }
}