use std::collections::HashMap;
use std::f32::INFINITY;
use crate::constants::*;
use crate::object::{Edge, Face, Object};
use crate::types::*;


#[derive(Clone, PartialEq)]
pub enum ShaderType {
    Wireframe,
    Flat,
    Gouraud,
    Phong,
}

pub struct Camera {
    pub vrp: Vec3,
    pub p: Vec3,
    pub y: Vec3,
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
    pub m_srt_sru: Mat4,
    zbuffer: Vec<f32>,
    pub buffer: Vec<u8>,
    pub buffer_width: usize,
    pub buffer_height: usize,
}

impl Default for Render {
    fn default() -> Self {
        let shader_type = ShaderType::Wireframe;
        let camera = Camera {
            vrp: Vec3::new(-2.0, 1.0, 40.0),
            p: Vec3::new(0.0, 0.0, 0.0),
            y: Vec3::new(0.0, 1.0, 0.0),
        };
        let window = Window {
            xmin: -GUI_VIEWPORT_WIDTH / 2.0,
            xmax: GUI_VIEWPORT_WIDTH / 2.0,
            ymin: -GUI_VIEWPORT_HEIGHT / 2.0,
            ymax: GUI_VIEWPORT_HEIGHT / 2.0,
        };
        let viewport = Viewport {
            umin: 0.0,
            umax: GUI_VIEWPORT_WIDTH - 1.0,
            vmin: 0.0,
            vmax: GUI_VIEWPORT_HEIGHT - 1.0,
        };

        let m_sru_srt = Mat4::identity();
        let m_srt_sru = Mat4::identity();

        let buffer_width = GUI_VIEWPORT_WIDTH as usize;
        let buffer_height = GUI_VIEWPORT_HEIGHT as usize;

        let buffer = vec![0; buffer_width * buffer_height * 4];
        let zbuffer = vec![-INFINITY; buffer_width * buffer_height];

        let mut obj = Self {
            shader_type,
            camera,
            window,
            viewport,
            m_sru_srt,
            m_srt_sru,
            zbuffer,
            buffer,
            buffer_width,
            buffer_height,
        };

        obj.calc_sru_srt_matrix();

        obj
    }
}

impl Render {
    /// Limpa os buffers de imagem e profundidade.
    pub fn clean_buffers(&mut self) {
        self.buffer = vec![0; self.buffer.len()];
        self.zbuffer = vec![-INFINITY; self.zbuffer.len()];
    }

    /// Calcula o teste de visibilidade das arestas da malha.
    fn calc_normal_test(
        &self,
        object: &mut Object,
    ) {
        for face in object.faces.iter() {
            let a: Vec3 = object.vertices[face.vertices[0]];
            let b: Vec3 = object.vertices[face.vertices[1]];
            let c: Vec3 = object.vertices[face.vertices[2]];

            let centroid: Vec3 = Vec3::new(
                (a.x + b.x + c.x) / 3.0,
                (a.y + b.y + c.y) / 3.0,
                (a.z + b.z + c.z) / 3.0,
            );
            let no: Vec3 = (self.camera.vrp - centroid).normalize();

            let visivel: bool = face.normal.dot(&no) > 0.0;

            object.edges[face.edges[0]].visible = visivel;
            object.edges[face.edges[1]].visible = visivel;
            object.edges[face.edges[2]].visible = visivel;
            object.edges[face.edges[3]].visible = visivel;
        }
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRC.
    fn calc_sru_src_matrix(&self) -> Mat4 {
        let n: Vec3 = self.camera.vrp - self.camera.p;
        let nn: Vec3 = n.normalize();

        let v: Vec3 = self.camera.y - (self.camera.y.dot(&nn) * nn);
        let vn: Vec3 = v.normalize();

        let un: Vec3 = vn.cross(&nn);

        Mat4::new(
            un.x, un.y, un.z, -self.camera.vrp.dot(&un),
            vn.x, vn.y, vn.z, -self.camera.vrp.dot(&vn),
            nn.x, nn.y, nn.z, -self.camera.vrp.dot(&nn),
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de projeção axonométrica isométrica.
    fn calc_proj_matrix(&self) -> Mat4 {
        Mat4::identity()
    }

    /// Calcula a matriz de transformação janela/porta de visão.
    fn calc_jp_matrix(&self) -> Mat4 {
        let sx = (self.viewport.umax - self.viewport.umin) / (self.window.xmax - self.window.xmin);
        let sy = (self.viewport.vmin - self.viewport.vmax) / (self.window.ymax - self.window.ymin);
        let tx = -self.window.xmin * sx + self.viewport.umin;
        let ty = self.window.ymin * ((self.viewport.vmax - self.viewport.vmin) / (self.window.ymax - self.window.ymin)) + self.viewport.vmax;

        Mat4::new(
            sx, 0.0, 0.0, tx,
            0.0, sy, 0.0, ty,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRT (matriz concatenada).
    fn calc_sru_srt_matrix(&mut self) {
        self.m_sru_srt = self.calc_jp_matrix() * self.calc_proj_matrix() * self.calc_sru_src_matrix();
        self.m_srt_sru = (self.calc_proj_matrix() * self.calc_sru_src_matrix()).try_inverse().unwrap_or_else(Mat4::identity);
    }

    /// Converte os pontos para o sistema de referência da tela.
    pub fn calc_srt_convertions(
        &self,
        sru_coords: &[Vec3],
    ) -> Vec<Vec3> {
        let mut srt_coords = Vec::with_capacity(sru_coords.len());
        for i in 0..sru_coords.len() {
            let mut srt: Mat4x1 = self.m_sru_srt * vec3_to_mat4x1(&sru_coords[i]);
            srt.x /= srt.w;
            srt.y /= srt.w;
            srt_coords.push(srt.xyz());
        }
        srt_coords
    }

    /// Calcula as interseções das arestas da face com as linhas horizontais.
    pub fn calc_intersections(
        vertices_srt: &[Vec3],
        face: &Face,
    ) -> HashMap<usize, Vec<(f32, f32)>> {
        let mut intersections = HashMap::new();

        for i in 0..4 {
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
                    x_intersections.push((x, z));
                }
                x += tx;
                y += 1.0;
                z += tz;
            }
        }

        for (_, intersections) in intersections.iter_mut() {
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        intersections
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

    /// Renderiza as faces de uma malha.
    pub fn render(
        &mut self,
        object: &mut Object,
        primary_edge_color: [u8; 4],
        secondary_edge_color: [u8; 4],
    ) {
        match self.shader_type {
            ShaderType::Wireframe => {
                self.render_wireframe(object, primary_edge_color, secondary_edge_color);
            }
            ShaderType::Flat => {
                self.render_constant(object);
            }
            ShaderType::Gouraud => {
                self.render_gouraud(object);
            }
            ShaderType::Phong => {
                self.render_phong(object);
            }
        }
    }

    /// Algoritmo para desenho de linhas 3D.
    fn draw_line(
        &mut self,
        start: &Vec3,
        end: &Vec3,
        color: [u8; 4],
    ) {
        let x0 = start.x as isize;
        let y0 = start.y as isize;
        let z0 = start.z;
        let x1 = end.x as isize;
        let y1 = end.y as isize;
        let z1 = end.z;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();

        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };

        let mut x = x0;
        let mut y = y0;

        let mut err = dx - dy;

        let mut points = Vec::new();

        loop {
            points.push((x as usize, y as usize));

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;

            if e2 > -dy {
                err -= dy;
                x += sx;
            }

            if e2 < dx {
                err += dx;
                y += sy;
            }
        }

        let dz = z1 - z0;
        let tz = dz / points.len() as f32;
        let mut z = z0;

        for k in 0..points.len() {
            let (j, i) = points[k];

            if i >= self.buffer_height {
                continue;
            }
            if j >= self.buffer_width {
                continue;
            }

            let zbuffer_index = i * self.buffer_width + j;
            if z > self.zbuffer[zbuffer_index] {
                self.paint(i, j, color);
                self.zbuffer[zbuffer_index] = z;
            }
            z += tz;
        }
    }

    /// Pinta as arestas da malha com o algoritmo de Bresenham adaptado para 3D.
    fn draw_wireframe_edges(
        &mut self,
        vertices_srt: &[Vec3],
        edges: &[Edge],
        primary_edge_color: [u8; 4],
        secondary_edge_color: [u8; 4],
    ) {
        for edge in edges.iter() {
            let start = &vertices_srt[edge.vertices[0]];
            let end = &vertices_srt[edge.vertices[1]];
            if edge.visible {
                self.draw_line(start, end, primary_edge_color);
            } else {
                self.draw_line(start, end, secondary_edge_color);
            }
        }
    }

    /// Preenche as faces da malha com a técnica de wireframe.
    fn render_wireframe(
        &mut self,
        object: &mut Object,
        primary_edge_color: [u8; 4],
        secondary_edge_color: [u8; 4],
    ) {
        let vertices_srt = self.calc_srt_convertions(&object.vertices);

        object.calc_normals();
        self.calc_normal_test(object);

        let mut faces = object.faces.clone();
        faces.sort_by(|a, b| {
            let a_z = (vertices_srt[a.vertices[0]].z + vertices_srt[a.vertices[1]].z + vertices_srt[a.vertices[2]].z + vertices_srt[a.vertices[3]].z) / 4.0;
            let b_z = (vertices_srt[b.vertices[0]].z + vertices_srt[b.vertices[1]].z + vertices_srt[b.vertices[2]].z + vertices_srt[b.vertices[3]].z) / 4.0;
            a_z.partial_cmp(&b_z).unwrap()
        });
        faces.reverse();

        // Para cada face, calcula as interseções da scanline
        for face in faces.iter() {
            let intersections = Self::calc_intersections(&vertices_srt, face);

            // Para cada scanline i
            for (i, scaline_intersections) in intersections.iter() {
                if scaline_intersections.len() < 2 {
                    continue;
                }

                if *i >= self.buffer_height as usize {
                    continue;
                }

                let mut counter = 0;

                // Para cada par de interseções da scanline
                while counter < scaline_intersections.len() {
                    let x0: usize = scaline_intersections[counter].0.ceil() as usize;
                    let x1: usize = scaline_intersections[counter + 1].0.floor() as usize;
                    let z0: f32 = scaline_intersections[counter].1;
                    let z1: f32 = scaline_intersections[counter + 1].1;

                    counter += 2;

                    if x1 < x0 {
                        continue;
                    }

                    let dx = (x1 - x0) as f32;
                    let dz = z1 - z0;
                    let tz = dz / dx;

                    let mut z: f32 = z0;

                    // Para cada pixel na scanline
                    for j in x0..=x1 {
                        if j >= self.buffer_width {
                            continue;
                        }

                        let z_index = i * self.buffer_width + j;

                        if z > self.zbuffer[z_index] {
                            let color = [0,0,0,0];
                            self.paint(*i, j, color);
                            self.zbuffer[z_index] = z;
                        }

                        z += tz;
                    }
                }
            }
        }

        self.draw_wireframe_edges(&vertices_srt, &object.edges, primary_edge_color, secondary_edge_color);
    }

    fn render_constant(
        &mut self,
        object: &mut Object,
    ) {
        let vertices_srt = self.calc_srt_convertions(&object.vertices);
        todo!();
    }

    fn render_gouraud(
        &mut self,
        object: &mut Object,
    ) {
        let vertices_srt = self.calc_srt_convertions(&object.vertices);
        todo!();
    }

    fn render_phong(
        &mut self,
        object: &mut Object,
    ) {
        let vertices_srt = self.calc_srt_convertions(&object.vertices);
        todo!();
    }
}
