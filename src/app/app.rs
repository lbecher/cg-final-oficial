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
    ni_value: u8,
    nj: String,
    nj_value: u8,
    ti: String,
    ti_value: u8,
    tj: String,
    tj_value: u8,
    resi: String,
    resi_value: u8,
    resj: String,
    resj_value: u8,
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

    show_control_points: bool,

    window_xmin: String,
    window_xmax: String,
    window_ymin: String,
    window_ymax: String,

    viewport_umin: String,
    viewport_umax: String,
    viewport_vmin: String,
    viewport_vmax: String,

    closed: bool,
    current_tab: Tab,
}

impl Default for App {
    fn default() -> Self {
        let render = Render::default();

        let buffer = render.buffer.clone();
        let size = [render.buffer_width, render.buffer_height];
        let image = ColorImage::from_rgba_premultiplied(size, &buffer);

        let objects = Vec::new();
        let selected_object = None;

        let ka_value: Vec3 = Vec3::new(0.0, 0.0, 0.0);
        let kd_value: Vec3 = Vec3::new(0.0, 0.0, 0.0);
        let ks_value: Vec3 = Vec3::new(0.0, 0.0, 1.0);
        let n_value: f32 = 6.0;
        let ka = ColorInputData::new(ka_value.x, ka_value.y, ka_value.z);
        let kd = ColorInputData::new(kd_value.x, kd_value.y, kd_value.z);
        let ks = ColorInputData::new(ks_value.x, ks_value.y, ks_value.z);
        let n = format!("N: {}", n_value);

        let ni_value = 4;
        let nj_value = 4;
        let ti_value = 3;
        let tj_value = 3;
        let resi_value = 8;
        let resj_value = 8;
        let smoothness_value = 2;
        let ni = format!("NI: {}", ni_value);
        let nj = format!("NJ: {}", nj_value);
        let ti = format!("TI: {}", ti_value);
        let tj = format!("TJ: {}", tj_value);
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
            n,
            n_value,

            render_duration: Duration::default(),

            theme: Theme::Dark,

            ni,
            ni_value,
            nj,
            nj_value,
            ti,
            ti_value,
            tj,
            tj_value,
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

            scale: "1.0".to_string(),
            scale_value: 1.0,
            translation: VectorInputData::new(0.0, 0.0, 0.0),
            translation_value: Vec3::new(0.0, 0.0, 0.0),
            rotation: VectorInputData::new(0.0, 0.0, 0.0),
            rotation_value: Vec3::new(0.0, 0.0, 0.0),

            window_xmin: format!("Xmin: {}", render.window.xmin),
            window_xmax: format!("Xmax: {}", render.window.xmax),
            window_ymin: format!("Ymin: {}", render.window.ymin),
            window_ymax: format!("Ymax: {}", render.window.ymax),

            viewport_umin: format!("Umin: {}", render.viewport.umin),
            viewport_umax: format!("Umax: {}", render.viewport.umax),
            viewport_vmin: format!("Vmin: {}", render.viewport.vmin),
            viewport_vmax: format!("Vmax: {}", render.viewport.vmax),

            render,
            image,

            show_control_points: true,
            closed: false,
            current_tab: Tab::General,
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

        ui.horizontal(|ui| {
            if ui.selectable_label(self.current_tab == Tab::General, "Geral").clicked() {
                self.current_tab = Tab::General;
            }
            if ui.selectable_label(self.current_tab == Tab::Objects, "Objetos").clicked() {
                self.current_tab = Tab::Objects;
            }
            if ui.selectable_label(self.current_tab == Tab::World, "Mundo").clicked() {
                self.current_tab = Tab::World;
            }
            if ui.selectable_label(self.current_tab == Tab::About, "Sobre").clicked() {
                self.current_tab = Tab::About;
            }
        });

        match self.current_tab {
            Tab::General => {
                ui.heading("Projeto");
                ui.horizontal(|ui| {
                    if ui.button("Abrir").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Arquivo LHM", &["lhm"])
                            .pick_file() {
                                if let Err(_e) = self.load_objects(path.to_str().unwrap()) {
                                    eprintln!("Falha ao abrir o projeto!");
                                }
                            }
                    }
                    if ui.button("Salvar").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_file_name("Novo Projeto")
                            .add_filter("Arquivo LHM", &["lhm"])
                            .save_file() {
                                if let Err(_e) = self.save_objects(path.to_str().unwrap()) {
                                    eprintln!("Falha ao salvar o projeto!");
                                }
                            }
                    }
                });

                ui.separator();
                ui.heading("Tema");
                ui.radio_value(&mut self.theme, Theme::Light, "Claro");
                ui.radio_value(&mut self.theme, Theme::Dark, "Escuro");

                ui.separator();
                ui.heading("Arestas");
                ui.horizontal( |ui| {
                    ui.label("Cor primária:");
                    ui.color_edit_button_srgba_unmultiplied(&mut self.primary_color);
                });
                ui.horizontal( |ui| {
                    ui.label("Cor secundária:");
                    ui.color_edit_button_srgba_unmultiplied(&mut self.secondary_color);
                });
                if ui.button("Aplicar cores").clicked() {
                    self.redraw();
                }

                ui.separator();
                ui.heading("Sombreamento");
                let old_shader = self.render.shader_type.clone();
                let old_visibility_filter = self.render.visibility_filter;
                ui.radio_value(&mut self.render.shader_type, ShaderType::Wireframe, "Aramado");
                ui.radio_value(&mut self.render.shader_type, ShaderType::Flat, "Constante");
                ui.radio_value(&mut self.render.shader_type, ShaderType::Gouraud, "Gouraud");
                ui.radio_value(&mut self.render.shader_type, ShaderType::Phong, "Phong");
                ui.label(format!("Tempo de renderização: {} ms", self.render_duration.as_millis()));
                if self.render.shader_type != old_shader || self.render.visibility_filter != old_visibility_filter {
                    redraw = true;
                }

                ui.separator();
                ui.heading("Outras opções");
                ui.checkbox(&mut self.render.visibility_filter, "Filtro de visibilidade");
                ui.checkbox(&mut self.show_control_points, "Pontos de controle");
            }
            Tab::Objects => {
                ui.heading("Objetos");
                ui.collapsing("Pontos de controle", |ui| {
                    ui.add(TextEdit::singleline(&mut self.ni)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.nj)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.ti)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.tj)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.smoothness)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
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
                        let new_object = Object::new(self.ni_value, self.nj_value, self.ti_value, self.tj_value, self.resi_value, self.resj_value, self.smoothness_value, self.ka_value, self.kd_value, self.ks_value, self.n_value, self.closed);
                        self.objects.push(new_object);
                        self.select_object(self.objects.len() - 1);
                        redraw = true;
                    }
                }
                ui.checkbox(&mut self.closed, "Criar objeto fechado");

                ui.separator();
                ui.heading("Lista de Objetos");
                if self.objects.is_empty() {
                    ui.label("Nenhum objeto criado");
                }
                for i in 0..self.objects.len() {
                    if ui.selectable_label(self.selected_object == Some(i), format!("Objeto {}", i)).clicked() {
                        self.select_object(i);
                    }
                }
                if ui.button("Limpar seleção").clicked() {
                    self.selected_object = None;
                }
                if ui.button("Limpar objetos").clicked() {
                    self.objects.clear();
                    self.selected_object = None;
                    redraw = true;
                }
                if ui.button("Remover objeto selecionado").clicked() {
                    if let Some(idx) = self.selected_object {
                        self.objects.remove(idx);
                        self.selected_object = None;
                        self.redraw();
                    }
                }

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
                        if ui.button("Rodar X-").clicked() {
                            self.objects[idx].rotate_x(-0.1);
                            self.redraw();
                        }
                        if ui.button("Rodar X+").clicked() {
                            self.objects[idx].rotate_x(0.1);
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Rodar Y-").clicked() {
                            self.objects[idx].rotate_y(-0.1);
                            self.redraw();
                        }
                        if ui.button("Rodar Y+").clicked() {
                            self.objects[idx].rotate_y(0.1);
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Rodar Z-").clicked() {
                            self.objects[idx].rotate_z(-0.1);
                            self.redraw();
                        }
                        if ui.button("Rodar Z+").clicked() {
                            self.objects[idx].rotate_z(0.1);
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Transladar -X").clicked() {
                            self.objects[idx].translate(&Vec3::new(-1.0, 0.0, 0.0));
                            self.redraw();
                        }
                        if ui.button("Transladar +X").clicked() {
                            self.objects[idx].translate(&Vec3::new(1.0, 0.0, 0.0));
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Transladar -Y").clicked() {
                            self.objects[idx].translate(& Vec3::new(0.0, -1.0, 0.0));
                            self.redraw();
                        }
                        if ui.button("Transladar +Y").clicked() {
                            self.objects[idx].translate(&Vec3::new(0.0, 1.0, 0.0));
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Transladar -Z").clicked() {
                            self.objects[idx].translate(&Vec3::new(0.0, 0.0, -1.0));
                            self.redraw();
                        }
                        if ui.button("Transladar +Z").clicked() {
                            self.objects[idx].translate(&Vec3::new(0.0, 0.0, 1.0));
                            self.redraw();
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Escalar -").clicked() {
                            self.objects[idx].scale(0.95);
                            self.redraw();
                        }
                        if ui.button("Escalar +").clicked() {
                            self.objects[idx].scale(1.05);
                            self.redraw();
                        }
                    });
                }
            }
            Tab::World => {
                ui.heading("Iluminação");
                ui.collapsing("Ila (iluminação ambiente)", |ui| {
                    ui.add(TextEdit::singleline(&mut self.ila.r)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.ila.g)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.ila.b)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    if ui.button("Aplicar").clicked() {
                        parse_input("R:", &mut self.render.light.ila.x, &mut self.ila.r);
                        parse_input("G:", &mut self.render.light.ila.y, &mut self.ila.g);
                        parse_input("B:", &mut self.render.light.ila.z, &mut self.ila.b);
                        redraw = true;
                    }
                });
                ui.collapsing("L (posição)", |ui| {
                    ui.add(TextEdit::singleline(&mut self.l.x)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.l.y)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.l.z)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    if ui.button("Aplicar").clicked() {
                        parse_input("X:", &mut self.render.light.l.x, &mut self.l.x);
                        parse_input("Y:", &mut self.render.light.l.y, &mut self.l.y);
                        parse_input("Z:", &mut self.render.light.l.z, &mut self.l.z);
                        redraw = true;
                    }
                });
                ui.collapsing("Il (emissão)", |ui| {
                    ui.add(TextEdit::singleline(&mut self.il.r)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.il.g)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.il.b)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    if ui.button("Aplicar").clicked() {
                        parse_input("R:", &mut self.render.light.il.x, &mut self.il.r);
                        parse_input("G:", &mut self.render.light.il.y, &mut self.il.g);
                        parse_input("B:", &mut self.render.light.il.z, &mut self.il.b);
                        redraw = true;
                    }
                });

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
                        self.render.calc_sru_srt_matrix();
                        redraw = true;
                    }
                });
                ui.collapsing("Y (orientação da câmera)", |ui| {
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
                        self.render.calc_sru_srt_matrix();
                        redraw = true;
                    }
                });
                ui.collapsing("P (ponto focal)", |ui| {
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
                        self.render.calc_sru_srt_matrix();
                        redraw = true;
                    }
                });
                ui.collapsing("Janela", |ui| {
                    ui.add(TextEdit::singleline(&mut self.window_xmin)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.window_xmax)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.window_ymin)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.window_ymax)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    if ui.button("Aplicar").clicked() {
                        parse_input("Xmin:", &mut self.render.window.xmin, &mut self.window_xmin);
                        parse_input("Xmax:", &mut self.render.window.xmax, &mut self.window_xmax);
                        parse_input("Ymin:", &mut self.render.window.ymin, &mut self.window_ymin);
                        parse_input("Ymax:", &mut self.render.window.ymax, &mut self.window_ymax);
                        self.render.calc_sru_srt_matrix();
                        redraw = true;
                    }
                });
                ui.collapsing("Porta de visão", |ui| {
                    ui.add(TextEdit::singleline(&mut self.viewport_umin)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.viewport_umax)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.viewport_vmin)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    ui.add(TextEdit::singleline(&mut self.viewport_vmax)
                        .desired_width(GUI_VECTOR_INPUT_WIDTH));
                    if ui.button("Aplicar").clicked() {
                        let umin = self.render.viewport.umin;
                        let umax = self.render.viewport.umax;
                        let vmin = self.render.viewport.vmin;
                        let vmax = self.render.viewport.vmax;
                        parse_input("Umin:", &mut self.render.viewport.umin, &mut self.viewport_umin);
                        parse_input("Umax:", &mut self.render.viewport.umax, &mut self.viewport_umax);
                        parse_input("Vmin:", &mut self.render.viewport.vmin, &mut self.viewport_vmin);
                        parse_input("Vmax:", &mut self.render.viewport.vmax, &mut self.viewport_vmax);
                        if self.render.viewport.umax - self.render.viewport.umin < GUI_VIEWPORT_WIDTH - 1.0 {
                            self.viewport_umax = "Umax: Inválido!".to_string();
                            self.viewport_umin = "Umin: Inválido!".to_string();
                            self.render.viewport.umin = umin;
                            self.render.viewport.umax = umax;
                        }
                        if self.render.viewport.vmax - self.render.viewport.vmin < GUI_VIEWPORT_HEIGHT - 1.0 {
                            self.viewport_vmax = "Vmax: Inválido!".to_string();
                            self.viewport_vmin = "Vmin: Inválido!".to_string();
                            self.render.viewport.vmin = vmin;
                            self.render.viewport.vmax = vmax;
                        }
                        self.render.calc_sru_srt_matrix();
                        redraw = true;
                    }
                });
            }
            Tab::About => {
                ui.heading("Desenvolvedores");
                ui.label("- Luiz Fernando Becher de Araujo");
                ui.label("- Heloisa Aparecida Alves");
                ui.label("- Matheus Lucas Ferreira Jacinto");
                ui.separator();
                ui.heading("Desenvolvimento");
                ui.label("Ano: 2024-2025");
                ui.label("Linguagem: Rust");
                ui.hyperlink("https://github.com/lbecher/cg-final-oficial/");
            }
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

        if self.show_control_points {
            if let Some(idx) = self.selected_object {
                let m_srt_sru = self.render.m_srt_sru.clone();

                let control_point_radius = 2.0;
                let mut dragged = false;
                let mut shapes = Vec::new();

                let control_points_srt: Vec<Vec3> = self.render.calc_srt_convertions(&self.objects[idx].control_points);
                let control_points: &mut Vec<Vec3> = &mut self.objects[idx].control_points;

                for (i, control_point) in control_points_srt.iter().enumerate() {
                    let (y, x) = self.render.srt_to_buffer(control_point.y as i32, control_point.x as i32);
                    let point = pos2(x as f32, y as f32);
                    let size = Vec2::splat(2.0 * control_point_radius);

                    let point_in_screen = to_screen.transform_pos(point);
                    let point_rect = Rect::from_center_size(point_in_screen, size);
                    let point_id = response.id.with(i);
                    let point_response = ui.interact(point_rect, point_id, Sense::drag());

                    if point_response.dragged() {
                        // Deslocamento do mouse
                        let drag_delta: Vec2 = point_response.drag_delta();
                        let drag_delta_sru: Mat4x1 = m_srt_sru * Mat4x1::new(drag_delta.x, drag_delta.y, 0.0, 0.0);
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
}


impl App {
    fn parse_object_props(&mut self) -> bool {
        let mut success = true;
        success &= self.parse_ni_nj_ti_tj_smoothness();
        success &= self.parse_resi_resj();
        success &= self.parse_ka_kd_ks();
        success
    }

    fn parse_ni_nj_ti_tj_smoothness(&mut self) -> bool {
        let mut success = true;
        success &= parse_input("NI:", &mut self.ni_value, &mut self.ni);
        success &= parse_input("NJ:", &mut self.nj_value, &mut self.nj);
        success &= parse_input("TI:", &mut self.ti_value, &mut self.ti);
        success &= parse_input("TJ:", &mut self.tj_value, &mut self.tj);
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

    pub fn select_object(&mut self, index: usize) {
        if let Some(object) = self.objects.get(index) {
            self.ni_value = object.ni;
            self.nj_value = object.nj;
            self.ti_value = object.ti;
            self.tj_value = object.tj;
            self.resi_value = object.resi;
            self.resj_value = object.resj;
            self.smoothness_value = object.smoothing_iterations;
            self.ka_value = object.ka;
            self.kd_value = object.kd;
            self.ks_value = object.ks;
            self.n_value = object.n;
            self.ni = format!("NI: {}", self.ni_value);
            self.nj = format!("NJ: {}", self.nj_value);
            self.ti = format!("TI: {}", self.ti_value);
            self.tj = format!("TJ: {}", self.tj_value);
            self.resi = format!("RESI: {}", self.resi_value);
            self.resj = format!("RESJ: {}", self.resj_value);
            self.smoothness = format!("Passos: {}", self.smoothness_value);
            self.ka = ColorInputData::new(self.ka_value.x, self.ka_value.y, self.ka_value.z);
            self.kd = ColorInputData::new(self.kd_value.x, self.kd_value.y, self.kd_value.z);
            self.ks = ColorInputData::new(self.ks_value.x, self.ks_value.y, self.ks_value.z);
            self.selected_object = Some(index);
        }
    }
}

#[derive(PartialEq)]
enum Theme {
    Light,
    Dark,
}

#[derive(PartialEq)]
enum Tab {
    General,
    Objects,
    World,
    About,
}

impl Theme {
    fn visuals(&self) -> eframe::egui::Visuals {
        match self {
            Theme::Light => eframe::egui::Visuals::light(),
            Theme::Dark => eframe::egui::Visuals::dark(),
        }
    }
}
