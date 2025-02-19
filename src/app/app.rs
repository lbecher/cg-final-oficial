use eframe::{App as EguiApp, Frame};
use eframe::emath;
use eframe::egui::{CentralPanel, Context, SidePanel, TopBottomPanel, Ui, Vec2, Sense,
    pos2, Color32, Grid, Pos2, Rect, Shape, Stroke, Widget, Window};
use eframe::epaint::{self, CubicBezierShape, PathShape, QuadraticBezierShape};
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

    l: VectorInputData,

    control_points: Vec<Pos2>,
    degree: usize,

    theme: Theme,
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
            l: VectorInputData::default(),

            control_points: vec![
                Pos2::new(100.0, 100.0),
                Pos2::new(200.0, 200.0),
                Pos2::new(300.0, 100.0),
            ],
            degree: 3,

            theme: Theme::Dark,
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

        ctx.set_visuals(self.theme.visuals());
    }
}

impl App {
    pub fn menu_bar_content(&mut self, ui: &mut Ui) {
        ui.horizontal( |ui| {
            ui.label("Tema:");
            ui.radio_value(&mut self.theme, Theme::Light, "Claro");
            ui.radio_value(&mut self.theme, Theme::Dark, "Escuro");
        });
    }

    pub fn side_panel_content(&mut self, ui: &mut Ui) {
        ui.label("Side panel");

        ui.collapsing("Câmera", |ui| {
            vector_input(ui, "VRP (posição da câmera)", &mut self.vrp);
            vector_input(ui, "P", &mut self.p);
            vector_input(ui, "Y", &mut self.y);
        });

        ui.collapsing("Iluminação", |ui| {
            vector_input(ui, "L (posição da lâmpada)", &mut self.l);
        });
    }

    pub fn central_panel_content(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 300.0), Sense::hover());

        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, response.rect.size()),
            response.rect,
        );

        let control_point_radius = 2.0;

        let control_point_shapes: Vec<Shape> = self
            .control_points
            .iter_mut()
            .enumerate()
            .take(self.degree)
            .map(|(i, point)| {
                let size = Vec2::splat(2.0 * control_point_radius);

                let point_in_screen = to_screen.transform_pos(*point);
                let point_rect = Rect::from_center_size(point_in_screen, size);
                let point_id = response.id.with(i);
                let point_response = ui.interact(point_rect, point_id, Sense::drag());

                *point += point_response.drag_delta();
                *point = to_screen.from().clamp(*point);

                let point_in_screen = to_screen.transform_pos(*point);
                let stroke = Stroke::new(1.0, Color32::RED);

                Shape::circle_filled(point_in_screen, control_point_radius, Color32::RED)
            })
            .collect();

        let points_in_screen: Vec<Pos2> = self
            .control_points
            .iter()
            .take(self.degree)
            .map(|p| to_screen * *p)
            .collect();

        // Desenhar arestas entre os pontos de controle
        let edge_shapes: Vec<Shape> = points_in_screen
            .windows(2)
            .map(|w| {
                Shape::line_segment([w[0], w[1]], Stroke::new(1.0, Color32::WHITE))
            })
            .collect();

        painter.extend(control_point_shapes);
        painter.extend(edge_shapes);
    }
}

#[derive(PartialEq)]
enum Theme {
    Light,
    Dark,
}

impl Theme {
    fn visuals(&self) -> eframe::egui::Visuals {
        match self {
            Theme::Light => eframe::egui::Visuals::light(),
            Theme::Dark => eframe::egui::Visuals::dark(),
        }
    }
}