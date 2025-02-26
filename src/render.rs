use ordered_float::OrderedFloat;
use std::collections::BTreeMap;
use std::f32::INFINITY;
use crate::constants::*;
use crate::object::Face;
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
    pub shader_type: ShaderType,
    pub camera: Camera,
    pub window: Window,
    pub viewport: Viewport,
    pub m_sru_srt: Mat4,
    z_buffer: Vec<f32>,
    pub buffer: Vec<u8>,
    pub buffer_width: usize,
    pub buffer_height: usize,
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
            z_buffer: vec![INFINITY; GUI_VIEWPORT_WIDTH as usize * GUI_VIEWPORT_WIDTH as usize],
            buffer: vec![0; GUI_VIEWPORT_WIDTH as usize * GUI_VIEWPORT_WIDTH as usize * 4],
            buffer_width: GUI_VIEWPORT_WIDTH as usize,
            buffer_height: GUI_VIEWPORT_WIDTH as usize,
        }
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRC.
    fn calc_sru_src_matrix(&self) -> Mat4 {
        let n: Vec3 = self.camera.vrp - self.camera.p;
        let nn: Vec3 = n.normalize();
        let v: Vec3 = self.camera.y - (self.camera.y.dot(&nn) * nn);
        let vn: Vec3 = v.normalize();
        let u: Vec3 = vn.cross(&nn);
        let un: Vec3 = u.normalize();

        Mat4::new(
            un.x, un.y, un.z, -self.camera.vrp.dot(&un),
            vn.x, vn.y, vn.z, -self.camera.vrp.dot(&vn),
            nn.x, nn.y, nn.z, -self.camera.vrp.dot(&nn),
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de projeção axonométrica isométrica.
    fn calc_proj_matrix(&self) -> Mat4 {
        Mat4::new(
            (2.0_f32).sqrt() / 2.0, 0.0, (2.0_f32).sqrt() / 2.0, 0.0,
            (6.0_f32).sqrt() / 6.0, (3.0_f32).sqrt() / 3.0, -(6.0_f32).sqrt() / 6.0, 0.0,
            -(3.0_f32).sqrt() / 3.0, (6.0_f32).sqrt() / 6.0, (6.0_f32).sqrt() / 6.0, 0.0,
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
    fn calc_sru_srt_matrix(&self) -> Mat4 {
        self.calc_jp_matrix() * self.calc_proj_matrix() * self.calc_sru_src_matrix()
    }

    /// Limpa os buffers de imagem e profundidade.
    pub fn clean_buffers(&mut self) {
        self.buffer = vec![0; GUI_VIEWPORT_WIDTH as usize * GUI_VIEWPORT_WIDTH as usize * 4];
        self.z_buffer = vec![0.0; GUI_VIEWPORT_WIDTH as usize * GUI_VIEWPORT_WIDTH as usize];
    }

    /// Renderiza as faces de uma malha.
    pub fn render(
        &mut self,
        vertices: &[Vec3],
        vertices_srt: &[Vec3],
        faces: &mut [Face],
        primary_edge_color: [u8; 4],
        secondary_edge_color: [u8; 4],
    ) {
        match self.shader_type {
            ShaderType::Wireframe => {
                self.calc_normal_test(vertices, faces, &self.camera);
                self.fill_wireframe(vertices_srt, faces, primary_edge_color, secondary_edge_color);
            }
            ShaderType::Constant => {
                todo!();
            }
            ShaderType::Gouraud => {
                todo!();
            }
            ShaderType::Phong => {
                todo!();
            }
        }
    }

    /// Preenche as faces da malha com a técnica de wireframe.
    fn fill_wireframe(
        &mut self,
        vertices_srt: &[Vec3],
        faces: &[Face],
        primary_edge_color: [u8; 4],
        secondary_edge_color: [u8; 4],
    ) {
        for face in faces.iter() {
            // Para cada face, calcula as interseções da varredura
            for (i, intersections) in Render::calc_intersections(&vertices_srt, face) {
                // TODO! Resolver essa gambiarra
                if intersections.len() < 2 {
                    // Não há interseções suficientes para formar uma aresta (WHY???)
                    continue;
                }

                let mut counter = 0;

                while counter < (intersections.len() / 2) * 2 {
                    let x_initial: usize = intersections[counter].0.ceil() as usize;
                    let x_final: usize = intersections[counter + 1].0.floor() as usize;
                    let z_initial: f32 = *intersections[counter].1;
                    let z_final: f32 = *intersections[counter + 1].1;

                    counter += 2;

                    let dx = (x_final - x_initial) as f32;
                    let dz = z_final - z_initial;
                    let tz = dz / dx;

                    let mut z: f32 = z_initial;

                    // Obtém a cor da aresta de acordo com o teste da normal da face
                    let edge_color = if face.visible {
                        primary_edge_color
                    } else {
                        secondary_edge_color
                    };

                    // Preenche os pixels intermediários da scanline
                    for j in x_initial..=x_final {
                        let z_index = i * self.buffer_width + j;
                        if z <= self.z_buffer[z_index] {
                            let color = [0, 0, 255, 255];
                            self.paint(i, j, color);
                            self.z_buffer[z_index] = z;
                        }
                        z += tz;
                    }
                }
            }
        }
    }

    fn fill_constant(
        &mut self,
        vertices_srt: &[Mat4x1],
        faces: &[([usize; 4], f32)],
    ) {
        todo!();
    }

    fn fill_gouraud(
        &mut self,
        vertices_srt: &[Mat4x1],
        faces: &[([usize; 4], f32)],
    ) {
        todo!();
    }

    fn fill_phong(
        &mut self,
        vertices_srt: &[Mat4x1],
        faces: &[([usize; 4], f32)],
    ) {
        todo!();
    }

    /// Pinta um pixel no buffer de imagem.
    fn paint(
        &mut self,
        i: usize,
        j: usize,
        color: [u8; 4],
    ) {
        if self.buffer_width <= j {
            return;
        }
        if self.buffer_height <= i {
            return;
        }

        let index = ((i * self.buffer_width) + j) * 4;

        self.buffer[index]     = color[0];
        self.buffer[index + 1] = color[1];
        self.buffer[index + 2] = color[2];
        self.buffer[index + 3] = color[3];
    }

    /// Calcula as interseções das arestas da face com as linhas horizontais.
    pub fn calc_intersections(
        vertices_srt: &[Vec3],
        face: &Face,
    ) -> BTreeMap<usize, Vec<(OrderedFloat<f32>, OrderedFloat<f32>)>> {
        let mut intersections: BTreeMap<usize, Vec<(OrderedFloat<f32>, OrderedFloat<f32>)>> = BTreeMap::new();

        for i in 0..3 {
            let mut x0 = vertices_srt[face.vertices[i]].x;
            let mut y0 = vertices_srt[face.vertices[i]].y.round();
            let mut z0 = vertices_srt[face.vertices[i]].z;
            let mut x1 = vertices_srt[face.vertices[i + 1]].x;
            let mut y1 = vertices_srt[face.vertices[i + 1]].y.round();
            let mut z1 = vertices_srt[face.vertices[i + 1]].z;

            if y0 > y1 {
                let x = x0;
                x0 = x1;
                x1 = x;

                let y = y0;
                y0 = y1;
                y1 = y;

                let z = z0;
                z0 = z1;
                z1 = z;
            }

            let dx = x1 - x0;
            let dy = y1 - y0;
            let dz = z1 - z0;
            let tx = dx / dy;
            let tz = dz / dy;

            let mut x = x0;
            let mut y = y0.round();
            let mut z = z0;

            while y < y1 {
                if y >= 0.0 {
                    let x_intersections = intersections.entry(y as usize)
                        .or_insert(Vec::new());
                    x_intersections.push((x.into(), z.into()));
                }
                x += tx;
                y += 1.0;
                z += tz;
            }
        }

        for (_, intersections) in intersections.iter_mut() {
            intersections.sort();
        }

        intersections
    }

    fn calc_normal_test(
        &self,
        vertices: &[Vec3],
        faces: &mut [Face],
        camera: &Camera,
    ) {
        for face in faces.iter_mut() {
            let a: Vec3 = vertices[face.vertices[0]];
            let b: Vec3 = vertices[face.vertices[1]];
            let c: Vec3 = vertices[face.vertices[2]];

            let bc: Vec3 = Vec3::new(c.x - b.x, c.y - b.y, c.z - b.z);
            let ba: Vec3 = Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z);
            let nn: Vec3 = bc.cross(&ba).normalize();

            let centroid: Vec3 = Vec3::new(
                (a.x + b.x + c.x) / 3.0,
                (a.y + b.y + c.y) / 3.0,
                (a.z + b.z + c.z) / 3.0,
            );
            let no: Vec3 = (camera.vrp - centroid).normalize();

            face.visible = nn.dot(&no) > 0.0;
        }
    }
}
