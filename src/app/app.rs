use eframe::{App as EguiApp, Frame};
use eframe::egui::{vec2, Align2, CentralPanel, Color32, ColorImage, Context, FontId, Key, Painter, Pos2, Rect, ScrollArea, Panel, Sense, TextureHandle, TextureOptions, Ui, Vec2};
use std::time::{Duration, Instant};
use crate::constants::*;
use crate::game::{Game, GameState};
use crate::render::*;
use crate::types::*;

pub struct App {
    game: Game,

    render: Render,
    texture: Option<TextureHandle>,
    /// Intensidade da luz do Sol a pino (antes do fator de luz do dia).
    light_intensity: Vec3,

    last_frame: Instant,
    frame_duration: Duration,
    render_duration: Duration,

    theme: Theme,
}

impl Default for App {
    fn default() -> Self {
        let render = Render::default();
        let light_intensity = render.light.il;

        Self {
            game: Game::new(),

            render,
            texture: None,
            light_intensity,

            last_frame: Instant::now(),
            frame_duration: Duration::default(),
            render_duration: Duration::default(),

            theme: Theme::Dark,
        }
    }
}

impl EguiApp for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        self.frame_duration = now - self.last_frame;
        self.last_frame = now;
        let dt = self.frame_duration.as_secs_f32().min(MAX_FRAME_TIME);

        ctx.memory_mut(|memory| {
            if let Some(id) = memory.focused() {
                memory.surrender_focus(id);
            }
        });

        if ctx.input(|i| i.key_pressed(Key::Space) || i.key_pressed(Key::ArrowUp)) {
            self.game.action();
        }

        self.game.update(dt);
        self.update_lighting();
        self.redraw(&ctx);

        Panel::right("side_panel")
            .exact_size(GUI_SIDEBAR_WIDTH)
            .resizable(false)
            .show(ui, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    self.side_panel_content(ui);
                });
            });

        CentralPanel::default()
            .show(ui, |ui| {
                self.central_panel_content(ui);
            });

        ctx.set_visuals(self.theme.visuals());
        ctx.request_repaint();
    }
}

impl App {
    /// A fonte de luz acompanha o Sol, e sua intensidade (e a cor do céu)
    /// varia com a altura dele.
    fn update_lighting(&mut self) {
        let daylight = self.game.daylight();

        self.render.light.l = self.game.sun_position();
        self.render.light.il = self.light_intensity * (MIN_DAYLIGHT + (1.0 - MIN_DAYLIGHT) * daylight);

        let night = Vec3::from(SKY_NIGHT);
        let day = Vec3::from(SKY_DAY);
        let sky = night + (day - night) * daylight;
        self.render.background = [sky.x as u8, sky.y as u8, sky.z as u8, 255];
    }

    pub fn redraw(&mut self, ctx: &Context) {
        let start = Instant::now();

        self.render.clean_buffers();
        for mut object in self.game.scene() {
            self.render.render(&mut object, [0, 0, 0, 255], [255, 0, 0, 255]);
        }

        let size = [self.render.buffer_width, self.render.buffer_height];
        let image = ColorImage::from_rgba_premultiplied(size, &self.render.buffer);
        match &mut self.texture {
            Some(texture) => texture.set(image, TextureOptions::default()),
            None => self.texture = Some(ctx.load_texture("render", image, TextureOptions::default())),
        }

        self.render_duration = start.elapsed();
    }

    pub fn side_panel_content(&mut self, ui: &mut Ui) {
        ui.heading("Jogo");
        let state = match self.game.state {
            GameState::Ready => "Aguardando início",
            GameState::Running => "Em andamento",
            GameState::GameOver => "Fim de jogo",
        };
        ui.label(format!("Estado: {}", state));
        ui.label(format!("Pontuação: {}", self.game.score as u32));
        ui.label(format!("Recorde: {}", self.game.best_score));
        ui.label(format!("Esferas saltadas: {}", self.game.jumped_obstacles));
        ui.label(format!("Velocidade: {:.1} u/s", self.game.speed));
        ui.label("Barra de espaços ou clique na tela: iniciar, saltar e reiniciar.");

        ui.separator();
        ui.heading("Sombreamento");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Flat, "Constante");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Gouraud, "Gouraud");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Phong, "Phong");
        ui.radio_value(&mut self.render.shader_type, ShaderType::Wireframe, "Aramado");
        ui.checkbox(&mut self.render.visibility_filter, "Ocultação de faces (normal)");

        ui.separator();
        ui.heading("Iluminação");
        let sun = self.game.sun_position();
        ui.label(format!("Sol: ({:.1}, {:.1}, {:.1})", sun.x, sun.y, sun.z));
        ui.label(format!("Luz do dia: {:.0}%", self.game.daylight() * 100.0));

        ui.separator();
        ui.heading("Desempenho");
        ui.label(format!("Tempo de renderização: {} ms", self.render_duration.as_millis()));
        let fps = 1.0 / self.frame_duration.as_secs_f32().max(f32::EPSILON);
        ui.label(format!("Quadros por segundo: {:.0}", fps));

        ui.separator();
        ui.heading("Tema");
        ui.radio_value(&mut self.theme, Theme::Light, "Claro");
        ui.radio_value(&mut self.theme, Theme::Dark, "Escuro");

        ui.separator();
        ui.heading("Desenvolvedores");
        ui.label("- Luiz Fernando Becher de Araujo");
        ui.label("- Heloisa Aparecida Alves");
        ui.label("- Matheus Lucas Ferreira Jacinto");
        ui.hyperlink("https://github.com/lbecher/cg-final-oficial/");
    }

    pub fn central_panel_content(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(GUI_VIEWPORT_WIDTH, GUI_VIEWPORT_HEIGHT), Sense::click());

        if let Some(texture) = &self.texture {
            let uv = Rect {
                min: Pos2::new(0.0, 0.0),
                max: Pos2::new(1.0, 1.0),
            };
            painter.image(texture.id(), response.rect, uv, Color32::WHITE);
        }

        if response.hovered() && ui.input(|i| i.pointer.primary_pressed()) {
            self.game.action();
        }

        self.draw_hud(&painter, response.rect);
    }

    fn draw_hud(&self, painter: &Painter, rect: Rect) {
        let color = if self.game.daylight() > 0.5 { Color32::from_gray(50) } else { Color32::from_gray(230) };

        painter.text(
            rect.right_top() + vec2(-16.0, 12.0),
            Align2::RIGHT_TOP,
            format!("HI {:05}  {:05}", self.game.best_score, self.game.score as u32),
            FontId::monospace(20.0),
            color,
        );

        let (title, subtitle) = match self.game.state {
            GameState::Ready => ("T-REX 3D", "Pressione ESPAÇO ou clique para começar"),
            GameState::GameOver => ("FIM DE JOGO", "Pressione ESPAÇO ou clique para reiniciar"),
            GameState::Running => return,
        };
        painter.text(rect.center() - vec2(0.0, 40.0), Align2::CENTER_CENTER, title, FontId::monospace(32.0), color);
        painter.text(rect.center(), Align2::CENTER_CENTER, subtitle, FontId::proportional(18.0), color);
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
