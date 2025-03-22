pub struct ColorInputData {
    pub r: String,
    pub g: String,
    pub b: String,
}

impl Default for ColorInputData {
    fn default() -> Self {
        Self {
            r: "R: 0".to_string(),
            g: "G: 0".to_string(),
            b: "B: 0".to_string(),
        }
    }
}

impl ColorInputData {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: format!("R: {}", r),
            g: format!("G: {}", g),
            b: format!("B: {}", b),
        }
    }
}