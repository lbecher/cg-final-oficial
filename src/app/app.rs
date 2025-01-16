use eframe::{App as EguiApp, Frame};
use eframe::egui::{CentralPanel, Context, SidePanel, TopBottomPanel, Ui, Vec2, Sense};
use crate::app::vector_input::*;
use crate::constants::*;
use crate::object::Object;
use crate::render::Render;

pub struct App {
    objects: Vec<Object>,
    selected_object: Option<usize>,

    render: Render,

    vrp: VectorInputData,
    p: VectorInputData,
    y: VectorInputData,
}

impl Default for App {
    fn default() -> Self {
        Self {
            objects: vec![Object::new(100, 100, 3, 3, 200, 200)],
            selected_object: Some(0),

            render: Render::new(),

            vrp: VectorInputData::new(0.0, 0.0, 0.0),
            p: VectorInputData::default(),
            y: VectorInputData::new(0.0, 1.0, 0.0),
        }
    }
}

impl EguiApp for App {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        TopBottomPanel::top("menu_bar")
            .show(ctx, |ui| {
                self.menu_bar_content(ui);
            });

        SidePanel::right("side_panel")
            .exact_width(GUI_SIDEBAR_WIDTH)
            .resizable(false)
            .show(ctx,  |ui| {
                self.side_panel_content(ui);
            });

        CentralPanel::default()
            .show(ctx, |ui| {
                self.central_panel_content(ui);
            });
    }
}

impl App {
    pub fn menu_bar_content(&mut self, ui: &mut Ui) {
        ui.label("Menu bar");
    }

    pub fn side_panel_content(&mut self, ui: &mut Ui) {
        ui.label("Side panel");

        ui.collapsing("Câmera", |ui| {
            vector_input(ui, "VRP", &mut self.vrp);
            vector_input(ui, "P", &mut self.p);
            vector_input(ui, "Y", &mut self.y);
        });
    }

    pub fn central_panel_content(&mut self, ui: &mut Ui) {
        let painter_size = Vec2::new(GUI_VIEWPORT_WIDTH, GUI_VIEWPORT_HEIGHT);
        let painter_sense = Sense::hover();
        let (
            response,
            painter,
        ) = ui.allocate_painter(painter_size, painter_sense);
    }
}