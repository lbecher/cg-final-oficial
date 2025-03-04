use eframe::{App as EguiApp, Frame};
use eframe::emath;
use eframe::egui::{pos2, CentralPanel, Color32, ColorImage, Context, Pos2, Rect, ScrollArea, Sense, Shape, SidePanel,  TextureId, TextureOptions, Ui, Vec2, TextEdit};
use serde_json;
use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::time::{Duration, Instant};
use crate::app::color_input::*;
use crate::app::vector_input::*;
use crate::app::parse_input::*;
use crate::constants::*;
use crate::object::Object;
use crate::render::*;
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

    ila: ColorInputData,
    l: VectorInputData,
    il: ColorInputData,

    render_duration: Duration,

    theme: Theme,

    ni: String,
    ni_value: usize,
    nj: String,
    nj_value: usize,
    resi: String,
    resi_value: usize,
    resj: String,
    resj_value: usize,
    smoothness: String,
    smoothness_value: u8,

    ka: ColorInputData,
    ka_value: Vec3,
    kd: ColorInputData,
    kd_value: Vec3,
    ks: ColorInputData,
    ks_value: Vec3,
    n: String,
    n_value: f32,

    scale: String,
    scale_value: f32,
    translation: VectorInputData,
    translation_value: Vec3,
    rotation: VectorInputData,
    rotation_value: Vec3,
}

impl Default for App {
    fn default() -> Self {
        let render = Render::default();

        let buffer = render.buffer.clone();
        let size = [render.buffer_width, render.buffer_height];
        let image = ColorImage::from_rgba_premultiplied(size, &buffer);

        let objects = Vec::new();
        let selected_object = None;

        let ka_value: Vec3 = Vec3::new(0.4, 0.4, 0.8);
        let kd_value: Vec3 = Vec3::new(0.7, 0.9, 0.4);
        let ks_value: Vec3 = Vec3::new(0.5, 0.1, 0.9);
        let n_value: f32 = 2.0;
        let ka = ColorInputData::new(ka_value.x, ka_value.y, ka_value.z);
        let kd = ColorInputData::new(kd_value.x, kd_value.y, kd_value.z);
        let ks = ColorInputData::new(ks_value.x, ks_value.y, ks_value.z);
        let n = format!("N: {}", n_value);

        let ni_value = 4;
        let nj_value = 4;
        let resi_value = 8;
        let resj_value = 8;
        let smoothness_value = 2;
        let ni = format!("NI: {}", ni_value);
        let nj = format!("NJ: {}", nj_value);
        let resi = format!("RESI: {}", resi_value);
        let resj = format!("RESJ: {}", resj_value);
        let smoothness = format!("Passos: {}", smoothness_value);

        let mut obj = Self {
            objects,
            selected_object,

            primary_color: [0, 255, 0, 255],
            secondary_color: [255, 0, 0, 255],

            vrp: VectorInputData::new(render.camera.vrp.x, render.camera.vrp.y, render.camera.vrp.z),
            p: VectorInputData::new(render.camera.p.x, render.camera.p.y, render.camera.p.z),
            y: VectorInputData::new(render.camera.y.x, render.camera.y.y, render.camera.y.z),

            l: VectorInputData::new(render.light.l.x, render.light.l.y, render.light.l.z),
            il: ColorInputData::new(render.light.il.x, render.light.il.y, render.light.il.z),
            ila: ColorInputData::new(render.light.ila.x, render.light.ila.y, render.light.ila.z),

            render_duration: Duration::default(),

            theme: Theme::Dark,

            ni,
            ni_value,
            nj,
            nj_value,
            resi,
            resi_value,
            resj,
            resj_value,
            smoothness,
            smoothness_value,

            ka,
            ka_value,
            kd,
            kd_value,
            ks,
            ks_value,
            n,
            n_value,

            scale: "1.0".to_string(),
            scale_value: 1.0,
            translation: VectorInputData::new(0.0, 0.0, 0.0),
            translation_value: Vec3::new(0.0, 0.0, 0.0),
            rotation: VectorInputData::new(0.0, 0.0, 0.0),
            rotation_value: Vec3::new(0.0, 0.0, 0.0),

            render,
            image,
        };

        obj.redraw();

        obj
    }
}

impl EguiApp for App {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        SidePanel::right("side_panel")
            .exact_width(GUI_SIDEBAR_WIDTH)
            .resizable(false)
            .show(ctx,  |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    self.side_panel_content(ui);
                });
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
        let start = Instant::now();

        self.render.clean_buffers();
        for object in &mut self.objects {
            let primary_edge_color = self.primary_color;
            let secondary_edge_color = self.secondary_color;
            self.render.render(object, primary_edge_color, secondary_edge_color);
        }

        let buffer = self.render.buffer.clone();
        let size = [self.render.buffer_width, self.render.buffer_height];
        self.image = ColorImage::from_rgba_premultiplied(size, &buffer);

        self.render_duration = start.elapsed();
    }

    pub fn save_objects(&self, path: &str) -> io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &self.objects)?;
        Ok(())
    }

    pub fn load_objects(&mut self, path: &str) -> io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.objects = serde_json::from_reader(reader)?;
        self.redraw();
        Ok(())
    }

    pub fn side_panel_content(&mut self, ui: &mut Ui) {
        let mut redraw = false;

        ui.heading("Projeto");
        if ui.button("Salvar projeto").clicked() {
            if let Some(path) = rfd::FileDialog::new().save_file() {
                if let Err(e) = self.save_objects(path.to_str().unwrap()) {
                    eprintln!("Failed to save objects: {}", e);
                }
            }
        }
        if ui.button("Carregar projeto").clicked() {
            if let Some(path) = rfd::FileDialog::new().pick_file() {
                if let Err(e) = self.load_objects(path.to_str().unwrap()) {
                    eprintln!("Failed to load objects: {}", e);
                }
            }
        }

        ui.separator();

        ui.heading("Tema");
        ui.radio_value(&mut self.theme, Theme::Light, "Claro");
        ui.radio_value(&mut self.theme, Theme::Dark, "Escuro");

        ui.separator();

        ui.heading("Arestas");
        ui.horizontal( |ui| {
            ui.label("Cor primaria:");
            ui.color_edit_button_srgba_unmultiplied(&mut self.primary_color);
        });
        ui.horizontal( |ui| {
            ui.label("Cor secundaria:");
            ui.color_edit_button_srgba_unmultiplied(&mut self.secondary_color);
        });

        ui.separator();

        ui.heading("Sombreamento");
        let old_shader = self.render.shader_type.clone();
        let old_visibility_filter = self.render.visibility_filter;
        ui.radio_value(&mut self.render.shader_type, ShaderType::Wireframe, "Aramado");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Flat, "Constante");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Gouraud, "Gouraud");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Phong, "Phong");
        ui.label(format!("Tempo de renderização: {:?} ms", self.render_duration.as_millis()));
        ui.checkbox(&mut self.render.visibility_filter, "Filtro de visibilidade");
        if self.render.shader_type != old_shader || self.render.visibility_filter != old_visibility_filter {
            self.redraw();
        }

        ui.separator();

        ui.heading("Objetos");
        ui.collapsing("Pontos de controle", |ui| {
            ui.add(TextEdit::singleline(&mut self.ni)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.nj)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.smoothness)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if let Some(idx) = self.selected_object {
                if ui.button("Modificar objeto selecionado").clicked() {
                    if self.parse_ni_nj_smoothness() {
                        self.objects[idx].set_ni_nj(self.ni_value, self.nj_value, self.smoothness_value);
                        redraw = true;
                    }
                }
            }
        });
        ui.collapsing("Resolução", |ui| {
            ui.add(TextEdit::singleline(&mut self.resi)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.resj)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if let Some(idx) = self.selected_object {
                if ui.button("Modificar objeto selecionado").clicked() {
                    if self.parse_resi_resj() {
                        self.objects[idx].set_resi_resj(self.resi_value, self.resj_value);
                        redraw = true;
                    }
                }
            }
        });
        ui.collapsing("Material", |ui| {
            ui.label("Ka:");
            ui.add(TextEdit::singleline(&mut self.ka.r)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.ka.g)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.ka.b)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.label("Kd:");
            ui.add(TextEdit::singleline(&mut self.kd.r)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.kd.g)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.kd.b)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.label("Ks:");
            ui.add(TextEdit::singleline(&mut self.ks.r)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.ks.g)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.ks.b)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.label("N:");
            ui.add(TextEdit::singleline(&mut self.n)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if let Some(idx) = self.selected_object {
                if ui.button("Modificar objeto selecionado").clicked() {
                    if self.parse_ka_kd_ks() {
                        self.objects[idx].ka = self.ka_value;
                        self.objects[idx].kd = self.kd_value;
                        self.objects[idx].ks = self.ks_value;
                        self.objects[idx].n = self.n_value;
                        redraw = true;
                    }
                }
            }
        });
        if ui.button("Criar novo objeto").clicked() {
            if self.parse_object_props() {
                let mut new_object = Object::new(self.ni_value, self.nj_value, self.resi_value, self.resj_value, self.smoothness_value, self.ka_value, self.kd_value, self.ks_value, self.n_value);
                new_object.scale(100.0);
                self.objects.push(new_object);
                self.selected_object = Some(self.objects.len() - 1);
                redraw = true;
            }
        }
        ui.label("Objetos:");
        if self.objects.is_empty() {
            ui.label("Nenhum objeto criado");
        }
        for i in 0..self.objects.len() {
            if ui.selectable_label(self.selected_object == Some(i), format!("Objeto {}", i)).clicked() {
                self.selected_object = Some(i);
            }
        }
        if ui.button("Remover objeto selecionado").clicked() {
            if let Some(idx) = self.selected_object {
                self.objects.remove(idx);
                self.selected_object = None;
                self.redraw();
            }
        }

        ui.separator();

        ui.heading("Iluminação");
        vector_input(ui, "L (posição da lâmpada)", &mut self.l, &mut self.render.light.l, &mut redraw);
        color_input(ui, "IL (cor da lâmpada)", &mut self.il, &mut self.render.light.il, &mut redraw);
        color_input(ui, "ILA (cor da luz ambiente)", &mut self.ila, &mut self.render.light.ila, &mut redraw);

        ui.separator();

        ui.heading("Câmera");
        ui.collapsing("VRP (posição da câmera)", |ui| {
            ui.add(TextEdit::singleline(&mut self.vrp.x)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.vrp.y)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.vrp.z)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if ui.button("Aplicar").clicked() {
                parse_input("X:", &mut self.render.camera.vrp.x, &mut self.vrp.x);
                parse_input("Y:", &mut self.render.camera.vrp.y, &mut self.vrp.y);
                parse_input("Z:", &mut self.render.camera.vrp.z, &mut self.vrp.z);
                self.render.calc_sru_src_matrix();
                redraw = true;
            }
        });
        ui.collapsing("Y", |ui| {
            ui.add(TextEdit::singleline(&mut self.y.x)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.y.y)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.y.z)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if ui.button("Aplicar").clicked() {
                parse_input("X:", &mut self.render.camera.y.x, &mut self.y.x);
                parse_input("Y:", &mut self.render.camera.y.y, &mut self.y.y);
                parse_input("Z:", &mut self.render.camera.y.z, &mut self.y.z);
                self.render.calc_sru_src_matrix();
                redraw = true;
            }
        });
        ui.collapsing("P", |ui| {
            ui.add(TextEdit::singleline(&mut self.p.x)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.p.y)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            ui.add(TextEdit::singleline(&mut self.p.z)
                .desired_width(GUI_VECTOR_INPUT_WIDTH));
            if ui.button("Aplicar").clicked() {
                parse_input("X:", &mut self.render.camera.p.x, &mut self.p.x);
                parse_input("Y:", &mut self.render.camera.p.y, &mut self.p.y);
                parse_input("Z:", &mut self.render.camera.p.z, &mut self.p.z);
                self.render.calc_sru_src_matrix();
                redraw = true;
            }
        });

        ui.separator();

        ui.heading("Transformações");
        if let Some(idx) = self.selected_object {
            ui.collapsing("Translação Manual", |ui| {
                ui.add(TextEdit::singleline(&mut self.translation.x)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                ui.add(TextEdit::singleline(&mut self.translation.y)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                ui.add(TextEdit::singleline(&mut self.translation.z)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                if ui.button("Aplicar").clicked() {
                    parse_input("X:", &mut self.translation_value.x, &mut self.translation.x);
                    parse_input("Y:", &mut self.translation_value.y, &mut self.translation.y);
                    parse_input("Z:", &mut self.translation_value.z, &mut self.translation.z);
                    if let Some(idx) = self.selected_object {
                        self.objects[idx].translate(&self.translation_value);
                        redraw = true;
                    }
                }
            });
            ui.collapsing("Escala Manual", |ui| {
                ui.add(TextEdit::singleline(&mut self.scale)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                if ui.button("Aplicar").clicked() {
                    parse_input("XYZ:", &mut self.scale_value, &mut self.scale);
                    if let Some(idx) = self.selected_object {
                        self.objects[idx].scale(self.scale_value);
                        redraw = true;
                    }
                }
            });
            ui.collapsing("Rotação Manual", |ui| {
                ui.add(TextEdit::singleline(&mut self.rotation.x)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                ui.add(TextEdit::singleline(&mut self.rotation.y)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                ui.add(TextEdit::singleline(&mut self.rotation.z)
                    .desired_width(GUI_VECTOR_INPUT_WIDTH));
                if ui.button("Aplicar").clicked() {
                    parse_input("X:", &mut self.rotation_value.x, &mut self.rotation.x);
                    parse_input("Y:", &mut self.rotation_value.y, &mut self.rotation.y);
                    parse_input("Z:", &mut self.rotation_value.z, &mut self.rotation.z);
                    if let Some(idx) = self.selected_object {
                        self.objects[idx].rotate(&self.rotation_value);
                        redraw = true;
                    }
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Rodar X+").clicked() {
                    self.objects[idx].rotate_x(0.1);
                    self.redraw();
                }
                if ui.button("Rodar X-").clicked() {
                    self.objects[idx].rotate_x(-0.1);
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Rodar Y+").clicked() {
                    self.objects[idx].rotate_y(0.1);
                    self.redraw();
                }
                if ui.button("Rodar Y-").clicked() {
                    self.objects[idx].rotate_y(-0.1);
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Rodar Z+").clicked() {
                    self.objects[idx].rotate_z(0.1);
                    self.redraw();
                }
                if ui.button("Rodar Z-").clicked() {
                    self.objects[idx].rotate_z(-0.1);
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Transladar +X").clicked() {
                    self.objects[idx].translate(&Vec3::new(10.0, 0.0, 0.0));
                    self.redraw();
                }
                if ui.button("Transladar -X").clicked() {
                    self.objects[idx].translate(&Vec3::new(-10.0, 0.0, 0.0));
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Transladar +Y").clicked() {
                    self.objects[idx].translate(&Vec3::new(0.0, 10.0, 0.0));
                    self.redraw();
                }
                if ui.button("Transladar -Y").clicked() {
                    self.objects[idx].translate(& Vec3::new(0.0, -10.0, 0.0));
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Transladar +Z").clicked() {
                    self.objects[idx].translate(&Vec3::new(0.0, 0.0, 10.0));
                    self.redraw();
                }
                if ui.button("Transladar -Z").clicked() {
                    self.objects[idx].translate(&Vec3::new(0.0, 0.0, -10.0));
                    self.redraw();
                }
            });
            ui.horizontal(|ui| {
                if ui.button("Escalar +").clicked() {
                    self.objects[idx].scale(1.1);
                    self.redraw();
                }
                if ui.button("Escalar -").clicked() {
                    self.objects[idx].scale(0.9);
                    self.redraw();
                }
            });
        }

        if redraw {
            self.redraw();
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
            let m_srt_sru: Mat4 = self.render.m_srt_sru.clone();

            let control_point_radius = 2.0;
            let control_points_srt: Vec<Vec3> = self.render.calc_srt_convertions(&self.objects[idx].control_points);
            let control_points: &mut Vec<Vec3> = &mut self.objects[idx].control_points;

            let mut dragged = false;
            let mut shapes = Vec::new();

            for (i, control_point) in control_points_srt.iter().enumerate() {
                let point = pos2(control_point.x, control_point.y);
                let size = Vec2::splat(2.0 * control_point_radius);

                let point_in_screen = to_screen.transform_pos(point);
                let point_rect = Rect::from_center_size(point_in_screen, size);
                let point_id = response.id.with(i);
                let point_response = ui.interact(point_rect, point_id, Sense::drag());

                if point_response.dragged() {
                    let drag_delta = point_response.drag_delta();
                    let old_srt = self.render.m_sru_srt * vec3_to_mat4x1(&control_points[i]);
                    let mut new_srt = old_srt;
                    new_srt.x += drag_delta.x;
                    new_srt.y += drag_delta.y;

                    let new_sru = self.render.m_srt_sru * new_srt;
                    control_points[i] = mat4x1_to_vec3(&new_sru);

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


impl App {
    fn parse_object_props(&mut self) -> bool {
        let mut success = true;
        success &= self.parse_ni_nj_smoothness();
        success &= self.parse_resi_resj();
        success &= self.parse_ka_kd_ks();
        success
    }

    fn parse_ni_nj_smoothness(&mut self) -> bool {
        let mut success = true;
        success &= parse_input("NI:", &mut self.ni_value, &mut self.ni);
        success &= parse_input("NJ:", &mut self.nj_value, &mut self.nj);
        success &= parse_input("Passos:", &mut self.smoothness_value, &mut self.smoothness);
        if self.ni_value < 4 {
            self.ni = "NI: Inválido!".to_string();
            success = false;
        }
        if self.nj_value < 4 {
            self.nj = "NJ: Inválido!".to_string();
            success = false;
        }
        success
    }

    fn parse_resi_resj(&mut self) -> bool {
        let mut success = true;
        success &= parse_input("RESI:", &mut self.resi_value, &mut self.resi);
        success &= parse_input("RESJ:", &mut self.resj_value, &mut self.resj);
        if self.resi_value < 4 {
            self.resi = "RESI: Inválida!".to_string();
            success = false;
        }
        if self.resj_value < 4 {
            self.resj = "RESJ: Inválida!".to_string();
            success = false;
        }
        success
    }

    fn parse_ka_kd_ks(&mut self) -> bool {
        let mut success = true;
        success &= parse_input("R:", &mut self.ka_value.x, &mut self.ka.r);
        success &= parse_input("G:", &mut self.ka_value.y, &mut self.ka.g);
        success &= parse_input("B:", &mut self.ka_value.z, &mut self.ka.b);
        success &= parse_input("R:", &mut self.kd_value.x, &mut self.kd.r);
        success &= parse_input("G:", &mut self.kd_value.y, &mut self.kd.g);
        success &= parse_input("B:", &mut self.kd_value.z, &mut self.kd.b);
        success &= parse_input("R:", &mut self.ks_value.x, &mut self.ks.r);
        success &= parse_input("G:", &mut self.ks_value.y, &mut self.ks.g);
        success &= parse_input("B:", &mut self.ks_value.z, &mut self.ks.b);
        success &= parse_input("N:", &mut self.n_value, &mut self.n);
        success
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
