use eframe::egui::{TextEdit, Ui};
use crate::constants::*;
use crate::app::parse_input::*;
use crate::types::*;

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

pub fn color_input(
    ui: &mut Ui,
    label: &str,
    strings: &mut ColorInputData,
    values: &mut Vec3,
    redraw: &mut bool,
) {
    ui.collapsing(label, |ui| {
        ui.add(TextEdit::singleline(&mut strings.r)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut strings.g)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut strings.b)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));

        if ui.button("Aplicar").clicked() {
            parse_input("R:", &mut values[0], &mut strings.r);
            parse_input("G:", &mut values[1], &mut strings.g);
            parse_input("B:", &mut values[2], &mut strings.b);
            *redraw = true;
        }
    });
}