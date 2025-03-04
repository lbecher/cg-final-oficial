use eframe::egui::{TextEdit, Ui};
use crate::constants::*;
use crate::app::parse_input::*;
use crate::types::*;

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

pub fn vector_input(
    ui: &mut Ui,
    label: &str,
    strings: &mut VectorInputData,
    values: &mut Vec3,
    redraw: &mut bool,
) {
    ui.collapsing(label, |ui| {
        ui.add(TextEdit::singleline(&mut strings.x)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut strings.y)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut strings.z)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));

        if ui.button("Aplicar").clicked() {
            parse_input("X:", &mut values[0], &mut strings.x);
            parse_input("Y:", &mut values[1], &mut strings.y);
            parse_input("Z:", &mut values[2], &mut strings.z);
            *redraw = true;
        }
    });
}