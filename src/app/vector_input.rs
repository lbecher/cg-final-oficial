use eframe::egui::{TextEdit, Ui};
use crate::constants::*;
use crate::app::parse_input::*;

pub struct VectorInputData {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub xs: String,
    pub ys: String,
    pub zs: String,
}

impl Default for VectorInputData {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            xs: "X: 0".to_string(),
            ys: "Y: 0".to_string(),
            zs: "Z: 0".to_string(),
        }
    }
}

impl VectorInputData {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            xs: format!("X: {}", x),
            ys: format!("Y: {}", y),
            zs: format!("Z: {}", z),
        }
    }
}

pub fn vector_input(ui: &mut Ui, label: &str, data: &mut VectorInputData) {
    ui.collapsing(label, |ui| {
        ui.add(TextEdit::singleline(&mut data.xs)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut data.ys)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut data.zs)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));

        if ui.button("Aplicar").clicked() {
            parse_input("X:", &mut data.x, &mut data.xs);
            parse_input("Y:", &mut data.y, &mut data.ys);
            parse_input("Z:", &mut data.z, &mut data.zs);
        }
    });
}