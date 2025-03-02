use eframe::egui::{TextEdit, Ui};
use crate::constants::*;
use crate::app::parse_input::*;

pub struct ColorInputData {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub rs: String,
    pub gs: String,
    pub bs: String,
}

impl Default for ColorInputData {
    fn default() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            rs: "R: 0".to_string(),
            gs: "G: 0".to_string(),
            bs: "B: 0".to_string(),
        }
    }
}

impl ColorInputData {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self {
            r,
            g,
            b,
            rs: format!("R: {}", r),
            gs: format!("G: {}", g),
            bs: format!("B: {}", b),
        }
    }
}

pub fn color_input(ui: &mut Ui, label: &str, data: &mut ColorInputData) {
    ui.collapsing(label, |ui| {
        ui.add(TextEdit::singleline(&mut data.rs)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut data.gs)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));
        ui.add(TextEdit::singleline(&mut data.bs)
            .desired_width(GUI_VECTOR_INPUT_WIDTH));

        if ui.button("Aplicar").clicked() {
            parse_input("R:", &mut data.r, &mut data.rs);
            parse_input("G:", &mut data.g, &mut data.gs);
            parse_input("B:", &mut data.b, &mut data.bs);
        }
    });
}