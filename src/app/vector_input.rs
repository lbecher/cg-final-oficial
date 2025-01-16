use eframe::egui::{TextEdit, Ui};
use crate::constants::*;
use crate::app::parse_input::*;

pub struct VectorInputData {
    pub xv: f32,
    pub yv: f32,
    pub zv: f32,
    pub xs: String,
    pub ys: String,
    pub zs: String,
}

impl Default for VectorInputData {
    fn default() -> Self {
        Self {
            xv: 0.0,
            yv: 0.0,
            zv: 0.0,
            xs: "X: 0".to_string(),
            ys: "Y: 0".to_string(),
            zs: "Z: 0".to_string(),
        }
    }
}

impl VectorInputData {
    pub fn new(xv: f32, yv: f32, zv: f32) -> Self {
        Self {
            xv,
            yv,
            zv,
            xs: format!("X: {}", xv),
            ys: format!("Y: {}", yv),
            zs: format!("Z: {}", zv),
        }
    }
}

pub fn vector_input(ui: &mut Ui, label: &str, data: &mut VectorInputData) {
    ui.collapsing(label, |ui| {
        ui.horizontal(|ui| {
            ui.add(TextEdit::singleline(&mut data.xs)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut data.ys)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut data.zs)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));

            if ui.button("Aplicar").clicked() {
                parse_input("X:", &mut data.xv, &mut data.xs);
                parse_input("Y:", &mut data.yv, &mut data.ys);
                parse_input("Z:", &mut data.zv, &mut data.zs);
            }
        });
    });
}