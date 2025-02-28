use eframe::{App as EguiApp, Frame};
use eframe::emath;
use eframe::egui::{pos2, CentralPanel, Color32, ColorImage, Context, Pos2, Rect, Sense, Shape, SidePanel,  TextureId, TextureOptions, Ui, Vec2};
use crate::app::vector_input::*;
use crate::constants::*;
use crate::object::Object;
use crate::render::Render;
use crate::types::*;

pub struct App {
    objects: Vec<Object>,
    selected_object: Option<usize>,

    render: Render,
    image: ColorImage,

    primary_color: [u8; 4],
    secondary_color: [u8; 4],

    vrp: VectorInputData,
    p: VectorInputData,
    y: VectorInputData,

    l: VectorInputData,

    _control_points: Vec<Vec3>,

    theme: Theme,
}

impl Default for App {
    fn default() -> Self {
        let render = Render::new();

        let buffer = render.buffer.clone();
        let size = [render.buffer_width, render.buffer_height];
        let image = ColorImage::from_rgba_premultiplied(size, &buffer);

        let mut objects = Vec::new();
        objects.push(Object::new(2, 2, 6, 6, 3));
        objects[0].scale(40.0);
        objects[0].translate(&Vec3::new(300.0, 200.0, 0.0));

        Self {
            objects,
            selected_object: Some(0),

            render,
            image,

            primary_color: [0, 255, 0, 255],
            secondary_color: [255, 0, 0, 255],

            vrp: VectorInputData::new(0.0, 0.0, 0.0),
            p: VectorInputData::default(),
            y: VectorInputData::new(0.0, 1.0, 0.0),
            l: VectorInputData::default(),

            _control_points: vec![
                Vec3::new(100.0, 100.0, 0.0),
                Vec3::new(200.0, 200.0, 0.0),
                Vec3::new(300.0, 100.0, 0.0),
            ],

            theme: Theme::Dark,
        }
    }
}

impl EguiApp for App {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
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
    pub fn redraw(&mut self) {
        //let start = Instant::now();

        self.render.clean_buffers();
        for object in &mut self.objects {
            let primary_edge_color = self.primary_color;
            let secondary_edge_color = self.secondary_color;
            let render = &mut self.render;
            render.render(object, primary_edge_color, secondary_edge_color);
        }

        let buffer = self.render.buffer.clone();
        let size = [self.render.buffer_width, self.render.buffer_height];
        self.image = ColorImage::from_rgba_premultiplied(size, &buffer);

        //self.duration = start.elapsed();
    }

    pub fn side_panel_content(&mut self, ui: &mut Ui) {
        ui.heading("Tema");
        ui.radio_value(&mut self.theme, Theme::Light, "Claro");
        ui.radio_value(&mut self.theme, Theme::Dark, "Escuro");

        ui.heading("Projeto");
        if ui.button("Salvar projeto").clicked() {
            //self.save_objects();
        }
        if ui.button("Carregar projeto").clicked() {
           // self.load_objects();
        }
        ui.separator();

        ui.horizontal( |ui| {
            ui.label("Cor primaria:");
            ui.color_edit_button_srgba_unmultiplied(&mut self.primary_color);
        });

        ui.horizontal( |ui| {
            ui.label("Cor secundaria:");
            ui.color_edit_button_srgba_unmultiplied(&mut self.secondary_color);
        });

        ui.collapsing("Câmera", |ui| {
            vector_input(ui, "VRP (posição da câmera)", &mut self.vrp);
            vector_input(ui, "P", &mut self.p);
            vector_input(ui, "Y", &mut self.y);
        });

        ui.collapsing("Iluminação", |ui| {
            vector_input(ui, "L (posição da lâmpada)", &mut self.l);
        });

        if ui.button("Rodar").clicked() {
            if let Some(idx) = self.selected_object {
                self.objects[idx].rotate_z(0.1);
            }
        }
    }

    pub fn central_panel_content(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(GUI_VIEWPORT_WIDTH, GUI_VIEWPORT_HEIGHT), Sense::hover());
        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_size(Pos2::ZERO, response.rect.size()),
            response.rect,
        );

        let name = "render";
        let options = TextureOptions::default();
        let texture = ui.ctx().load_texture(name, self.image.clone(), options);
        let texture_id = TextureId::from(&texture);
        let uv = Rect {
            min: Pos2::new(0.0, 0.0),
            max: Pos2::new(1.0, 1.0),
        };
        painter.image(texture_id, response.rect, uv, Color32::WHITE);

        if let Some(idx) = self.selected_object {
            let m_sru_srt: Mat4 = self.render.m_sru_srt.clone();
            let m_srt_sru: Mat4 = m_sru_srt.try_inverse().unwrap(); //.unwrap_or_else(Mat4::identity);

            let control_point_radius = 2.0;
            let control_points_srt: Vec<Vec3> = self.render.calc_srt_control_points(&self.objects[idx].control_points);
            let control_points: &mut Vec<Vec3> = &mut self.objects[idx].control_points;

            let mut dragged = false;
            let mut shapes = Vec::new();

            for (i, control_point) in control_points_srt.iter().enumerate() {
                let mut point = pos2(control_point.x, control_point.y);
                let size = Vec2::splat(2.0 * control_point_radius);

                let point_in_screen = to_screen.transform_pos(point);
                let point_rect = Rect::from_center_size(point_in_screen, size);
                let point_id = response.id.with(i);
                let point_response = ui.interact(point_rect, point_id, Sense::drag());

                if point_response.dragged() {
                    let drag_delta: Vec2 = point_response.drag_delta();
                    let drag_delta_sru = m_srt_sru * Mat4x1::new(drag_delta.x, drag_delta.y, 0.0, 1.0);

                    point += drag_delta;
                    control_points[i] += mat4x1_to_vec3(&drag_delta_sru);

                    dragged = true;
                }

                let point_in_screen = to_screen.transform_pos(point);
                shapes.push(Shape::circle_filled(point_in_screen, control_point_radius, Color32::RED));
            }

            painter.extend(shapes);

            if dragged {
                self.objects[idx].calc_mesh();
                self.objects[idx].calc_centroid();
                self.redraw();
            }
        }
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