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
    fn calc_sru_src_matrix(&self, nn: &Vec3) -> Mat4 {
        let v: Vec3 = self.camera.y - (self.camera.y.dot(nn) * nn);
        let vn: Vec3 = v.normalize();
        let u: Vec3 = vn.cross(nn);
        let un: Vec3 = u.normalize();

        Mat4::new(
            un.x, un.y, un.z, -self.camera.vrp.dot(&un),
            vn.x, vn.y, vn.z, -self.camera.vrp.dot(&vn),
            nn.x, nn.y, nn.z, -self.camera.vrp.dot(nn),
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de projeção axonométrica isométrica.
    fn calc_axon_matrix(&self) -> Mat4 {
        let zvp = -self.camera.dp;
        let zprp = 0.0;

        let alpha = 35.26_f32.to_radians();
        let beta = 45.0_f32.to_radians();

        Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ) * Mat4::new(
            (beta.cos()), 0.0, (beta.sin()), 0.0,
            0.0, 1.0, 0.0, 0.0,
            (beta.sin()), 0.0, (beta.cos()), 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de transformação janela/porta de visão.
    fn calc_jp_matrix(&self) -> Mat4 {
        let sx = (self.viewport.umax - self.viewport.umin) / (self.window.xmax - self.window.xmin);
        let sy = (self.viewport.vmax - self.viewport.vmin) / (self.window.ymax - self.window.ymin);
        let tx = -self.window.xmin * sx + self.viewport.umin;
        let ty = -self.window.ymin * sy + self.viewport.vmin;

        Mat4::new(
            sx, 0.0, 0.0, tx,
            0.0, sy, 0.0, ty,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRT (matriz concatenada).
    fn calc_sru_srt_axon_matrix(&self) -> Mat4 {
        let n: Vec3 = self.camera.vrp - self.camera.p;
        let nn: Vec3 = n.normalize();
        self.calc_jp_matrix() * self.calc_axon_matrix() * self.calc_sru_src_matrix(&nn)
    }
}
