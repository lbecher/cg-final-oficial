use crate::types::*;

#[derive(Clone, PartialEq)]
pub enum ShaderType {
    Wireframe,
    Constant,
    Gouraud,
    Phong,
}

pub struct Camera {
    pub vrp: Vec3,
    pub p: Vec3,
    pub y: Vec3,
    pub dp: f32,
}

pub struct Window {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

pub struct Viewport {
    pub umin: f32,
    pub umax: f32,
    pub vmin: f32,
    pub vmax: f32,
}

pub struct Render {
    shader_type: ShaderType,
    camera: Camera,
    window: Window,
    viewport: Viewport,
    m_sru_srt: Mat4,
}

impl Render {
    pub fn new() -> Self {
        Self {
            shader_type: ShaderType::Wireframe,
            camera: Camera {
                vrp: Vec3::new(0.0, 0.0, 0.0),
                p: Vec3::new(0.0, 0.0, 1.0),
                y: Vec3::new(0.0, 1.0, 0.0),
                dp: 1.0,
            },
            window: Window {
                xmin: -1.0,
                xmax: 1.0,
                ymin: -1.0,
                ymax: 1.0,
            },
            viewport: Viewport {
                umin: 0.0,
                umax: 1.0,
                vmin: 0.0,
                vmax: 1.0,
            },
            m_sru_srt: Mat4::identity(),
        }
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRC.
    fn calc_sru_src_matrix(camera: &Camera, nn: &Vec3) -> Mat4 {
        todo!();
    }

    /// Calcula a matriz de projeção axonométrica isométrica.
    fn calc_orth_matrix(&self) -> Mat4 {
        todo!();
    }

    /// Calcula a matriz de transformação janela/porta de visão.
    fn calc_jp_matrix(&self) -> Mat4 {
        todo!();
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRT (matriz concatenada).
    fn calc_sru_srt_orth_matrix(camera: &Camera, window: &Window, viewport: &Viewport) -> Mat4 {
        todo!();
    }
}