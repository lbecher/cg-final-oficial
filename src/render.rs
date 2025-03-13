use std::collections::HashMap;
use std::f32::INFINITY;
use eframe::emath::OrderedFloat;

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

pub struct Light {
    pub l: Vec3,
    pub il: Vec3,
    pub ila: Vec3,
    pub n: f32,
}

pub struct Render {
    pub shader_type: ShaderType,
    pub camera: Camera,
    pub window: Window,
    pub viewport: Viewport,
    pub light: Light,
    pub m_sru_srt: Mat4,
    pub m_srt_sru: Mat4,
    zbuffer: Vec<f32>,
    pub buffer: Vec<u8>,
    pub buffer_width: usize,
    pub buffer_height: usize,
    pub visibility_filter: bool,
}

impl Default for Render {
    fn default() -> Self {
        let shader_type = ShaderType::Wireframe;
        let camera = Camera {
            vrp: Vec3::new(0.0, 0.0, 400.0),
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
        let light = Light {
            l: Vec3::new(300.0, 0.0, 0.0),
            il: Vec3::new(0.0, 200.0, 0.0),
            ila: Vec3::new(100.0, 0.0, 0.0),
            n: 2.0,
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
            light,
            m_sru_srt,
            m_srt_sru,
            zbuffer,
            buffer,
            buffer_width,
            buffer_height,
            visibility_filter: false,
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

    /// Algoritmo para desenho de linhas.
    fn draw_line(
        &mut self,
        start: &Vec3,
        end: &Vec3,
        color: [u8; 4],
    ) {
        let x0 = start.x as i32;
        let y0 = start.y as i32;
        let x1 = end.x as i32;
        let y1 = end.y as i32;

        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();

        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };

        let mut x = x0;
        let mut y = y0;

        let mut err = dx - dy;

        loop {
            self.paint(y, x, color);

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
    }

    /// Verifica se o índice i é válido.
    fn is_valid_i(
        &self,
        i: i32,
    ) -> bool {
        i >= 0 && i <= self.buffer_height as i32
    }

    /// Verifica se o índice j é válido.
    fn is_valid_j(
        &self,
        j: i32,
    ) -> bool {
        j >= 0 && j <= self.buffer_width as i32
    }

    /// Converte coordenadas SRT para do buffer de imagem.
    fn srt_to_buffer(
        &self,
        i: i32,
        j: i32,
    ) -> (i32, i32) {
        let ni = (i as f32 - self.viewport.vmin) / (self.viewport.vmax - self.viewport.vmin);
        let nj = (j as f32 - self.viewport.umin) / (self.viewport.umax - self.viewport.umin);

        let y = (ni * (self.buffer_height as f32 - 1.0)).round() as i32;
        let x = (nj * (self.buffer_width as f32 - 1.0)).round() as i32;

        (y, x)
    }

    /// Seta um Z no ZBuffer.
    fn set_zbuffer(
        &mut self,
        i: i32,
        j: i32,
        z: f32,
    ) {
        let (y, x) = self.srt_to_buffer(i, j);
        let x = x as usize;
        let y = y as usize;
        let index = y * self.buffer_width + x;
        self.zbuffer[index] = z;
    }

    /// Verifica se um pixel pode ser pintado.
    fn can_paint(
        &self,
        i: i32,
        j: i32,
        z: f32,
    ) -> bool {
        let (y, x) = self.srt_to_buffer(i, j);
        let x = x as usize;
        let y = y as usize;
        let index = y * self.buffer_width + x;
        z > self.zbuffer[index]
    }

    /// Pinta um pixel no buffer de imagem.
    fn paint(
        &mut self,
        i: i32,
        j: i32,
        color: [u8; 4],
    ) {
        let (y, x) = self.srt_to_buffer(i, j);
        let x = x as usize;
        let y = y as usize;

        //if (x >= self.buffer_width) || (y >= self.buffer_height) {
        //    return;
        //}

        let index = (y * self.buffer_width + x) * 4;
        self.buffer[index]     = color[0];
        self.buffer[index + 1] = color[1];
        self.buffer[index + 2] = color[2];
        self.buffer[index + 3] = color[3];
    }

    /// Calcula o teste de visibilidade das arestas da malha.
    fn calc_visibility(
        &self,
        face: &mut Face,
        edges: &mut Vec<Edge>,
    ) {
        let o: Vec3 = face.direction;
        let n: Vec3 = face.normal;
        let visibility = o.dot(&n) > 0.0;

        face.visible = visibility;
        for edge_index in face.edges.iter() {
            edges[*edge_index].visible = visibility;
        }
    }

    /// Calcula a matriz de transformação de coordenadas em SRU para SRC.
    pub fn calc_sru_src_matrix(&self) -> Mat4 {
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
    ) -> HashMap<i32, Vec<(f32, f32)>> {
        let mut intersections = HashMap::new();

        for i in 0..face.vertices.len() - 1 {
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
                    let x_intersections = intersections.entry(y as i32)
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

    /// Calcula as interseções das arestas da face com as linhas horizontais.
    pub fn calc_intersections_for_gouraud(
        vertices_srt: &[Vec3],
        intensities: &[Vec3],
        face: &Face,
    ) -> HashMap<i32, Vec<(f32, f32, f32, f32, f32)>> {
        let mut intersections = HashMap::new();

        for i in 0..face.vertices.len() - 1 {
            let mut x0 = vertices_srt[face.vertices[i]].x;
            let mut y0 = vertices_srt[face.vertices[i]].y.round();
            let mut z0 = vertices_srt[face.vertices[i]].z;
            let mut x1 = vertices_srt[face.vertices[i + 1]].x;
            let mut y1 = vertices_srt[face.vertices[i + 1]].y.round();
            let mut z1 = vertices_srt[face.vertices[i + 1]].z;

            let mut r0 = intensities[face.vertices[i]].x;
            let mut g0 = intensities[face.vertices[i]].y;
            let mut b0 = intensities[face.vertices[i]].z;
            let mut r1 = intensities[face.vertices[i + 1]].x;
            let mut g1 = intensities[face.vertices[i + 1]].y;
            let mut b1 = intensities[face.vertices[i + 1]].z;

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

                let r = r0;
                r0 = r1;
                r1 = r;

                let g = g0;
                g0 = g1;
                g1 = g;

                let b = b0;
                b0 = b1;
                b1 = b;
            }

            let dx = x1 - x0;
            let dy = y1 - y0;
            let dz = z1 - z0;
            let dr = r1 - r0;
            let dg = g1 - g0;
            let db = b1 - b0;

            let tx = dx / dy;
            let tz = dz / dy;
            let tr = dr / dy;
            let tg = dg / dy;
            let tb = db / dy;

            let mut x = x0;
            let mut y = y0.round();
            let mut z = z0;
            let mut r = r0;
            let mut g = g0;
            let mut b = b0;

            while y < y1 {
                if y >= 0.0 {
                    let x_intersections = intersections.entry(y as i32)
                        .or_insert(Vec::new());
                    x_intersections.push((x, z, r, g, b));
                }
                x += tx;
                y += 1.0;
                z += tz;
                r += tr;
                g += tg;
                b += tb;
            }
        }

        for (_, intersections) in intersections.iter_mut() {
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        intersections
    }

    /// Calcula as interseções das arestas da face com as linhas horizontais para Phong.
    pub fn calc_intersections_for_phong(
        vertices_srt: &[Vec3],
        normals: &[Vec3],
        face: &Face,
    ) -> HashMap<i32, Vec<(f32, f32, Vec3)>> {
        let mut intersections = HashMap::new();

        for i in 0..face.vertices.len() - 1 {
            let mut x0 = vertices_srt[face.vertices[i]].x;
            let mut y0 = vertices_srt[face.vertices[i]].y.round();
            let mut z0 = vertices_srt[face.vertices[i]].z;
            let mut x1 = vertices_srt[face.vertices[i + 1]].x;
            let mut y1 = vertices_srt[face.vertices[i + 1]].y.round();
            let mut z1 = vertices_srt[face.vertices[i + 1]].z;

            let mut n0: Vec3 = normals[face.vertices[i]];
            let mut n1: Vec3 = normals[face.vertices[i + 1]];

            if y0 > y1 {
                std::mem::swap(&mut x0, &mut x1);
                std::mem::swap(&mut y0, &mut y1);
                std::mem::swap(&mut z0, &mut z1);
                std::mem::swap(&mut n0, &mut n1);
            }

            let dx = x1 - x0;
            let dy = y1 - y0;
            let dz = z1 - z0;
            let dn = n1 - n0;

            let tx = dx / dy;
            let tz = dz / dy;
            let tn = dn / dy;

            let mut x = x0;
            let mut y = y0.round();
            let mut z = z0;
            let mut n = n0;

            while y < y1 {
                if y >= 0.0 {
                    let x_intersections = intersections.entry(y as i32)
                        .or_insert(Vec::new());
                    x_intersections.push((x, z, n));
                }
                x += tx;
                y += 1.0;
                z += tz;
                n += tn;
            }
        }

        for (_, intersections) in intersections.iter_mut() {
            intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        intersections
    }

    /// Calcula a cor de preenchimento da face.
    fn calc_color(
        &self,
        ka: &Vec3,
        kd: &Vec3,
        ks: &Vec3,

        // O ponto em que a iluminação será calculada
        // Constante -> Centroide da face
        // Gouraud -> Um vértice da face
        // Phong -> Um pixel da face
        position: &Vec3,
        direction: &Vec3,
        normal: &Vec3,
    ) -> [u8; 4] {
        let mut it: Vec3 = Vec3::zeros();

        let ln: Vec3 = (self.light.l - position).normalize();
        let id_dot = normal.dot(&ln);
        let r: Vec3 = 2.0 * id_dot * normal - ln;
        let is_dot = r.dot(direction);

        for i in 0..3 {
            let ia = self.light.ila[i] * ka[i];
            it[i] += ia;

            if id_dot > 0.0 {
                let id = self.light.il[i] * kd[i] * id_dot;
                it[i] += id;

                if is_dot > 0.0 {
                    let is = self.light.il[i] * ks[i] * is_dot.powf(self.light.n);
                    it[i] += is;
                };
            }

            if it[i] < 0.0 {
                it[i] = 0.0;
            } else if it[i] > 255.0 {
                it[i] = 255.0;
            }
        }

        [it[0] as u8, it[1] as u8, it[2] as u8, 255]
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
                self.render_flat(object);
            }
            ShaderType::Gouraud => {
                self.render_gouraud(object);
            }
            ShaderType::Phong => {
                self.render_phong(object);
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

        for face in object.faces.iter_mut() {
            face.calc_normal(&object.vertices);
            face.calc_centroid(&object.vertices);
            face.calc_direction(&self.camera.vrp);
            self.calc_visibility(face, &mut object.edges);
        }

        let mut faces = object.faces.clone();
        faces.sort_by(|a, b| {
            let a_depth = a.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / a.vertices.len() as f32;
            let b_depth = b.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / b.vertices.len() as f32;
            let a_depth = OrderedFloat(a_depth);
            let b_depth = OrderedFloat(b_depth);
            a_depth.cmp(&b_depth)
        });

        // Para cada face, calcula as interseções da scanline
        for face in faces.iter_mut() {
            if self.visibility_filter && !face.visible {
                continue;
            }

            let intersections = Self::calc_intersections(&vertices_srt, face);

            // Para cada scanline i
            for (i, scaline_intersections) in intersections.iter() {
                if scaline_intersections.len() < 2 {
                    continue;
                }

                if !self.is_valid_i(*i) {
                    continue;
                }

                let mut counter = 0;

                // Para cada par de interseções da scanline
                while counter < scaline_intersections.len() {
                    let x0 = scaline_intersections[counter].0.ceil() as i32;
                    let x1 = scaline_intersections[counter + 1].0.floor() as i32;

                    counter += 2;

                    if x1 < x0 {
                        continue;
                    }

                    for j in x0..=x1 {
                        if !self.is_valid_j(j) {
                            continue;
                        }

                        let color = [0, 0, 0, 0];
                        self.paint(*i, j, color);
                    }
                }
            }

            let color = if face.visible { primary_edge_color } else { secondary_edge_color };

            for vertex_index in 1..face.vertices.len() {
                let start: &Vec3 = &vertices_srt[face.vertices[vertex_index - 1]];
                let end: &Vec3 = &vertices_srt[face.vertices[vertex_index]];
                self.draw_line(start, end, color);
            }
        }
    }

    fn render_flat(
        &mut self,
        object: &mut Object,
    ) {
        let ka: Vec3 = object.ka;
        let kd: Vec3 = object.kd;
        let ks: Vec3 = object.ks;

        let vertices_srt = self.calc_srt_convertions(&object.vertices);

        for face in object.faces.iter_mut() {
            face.calc_normal(&object.vertices);
            face.calc_centroid(&object.vertices);
            face.calc_direction(&self.camera.vrp);
            self.calc_visibility(face, &mut object.edges);
        }

        let mut faces = object.faces.clone();
        faces.sort_by(|a, b| {
            let a_depth = a.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / a.vertices.len() as f32;
            let b_depth = b.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / b.vertices.len() as f32;
            let a_depth = OrderedFloat(a_depth);
            let b_depth = OrderedFloat(b_depth);
            a_depth.cmp(&b_depth)
        });

        for face in faces.iter_mut() {
            if self.visibility_filter && !face.visible {
                continue;
            }

            let direction: Vec3 = (self.camera.vrp - face.centroid).normalize();
            let color = self.calc_color(&ka, &kd, &ks, &face.centroid, &direction, &face.normal);

            let intersections = Self::calc_intersections(&vertices_srt, face);

            // Para cada scanline i
            for (i, scaline_intersections) in intersections.iter() {
                if scaline_intersections.len() < 2 {
                    continue;
                }

                if !self.is_valid_i(*i) {
                    continue;
                }

                let mut counter = 0;

                // Para cada par de interseções da scanline
                while counter < scaline_intersections.len() {
                    let x0: i32 = scaline_intersections[counter].0.ceil() as i32;
                    let x1: i32 = scaline_intersections[counter + 1].0.floor() as i32;
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
                        if !self.is_valid_j(j) {
                            continue;
                        }

                        if self.can_paint(*i, j, z) {
                            self.paint(*i, j, color);
                            self.set_zbuffer(*i, j, z);
                        }

                        z += tz;
                    }
                }
            }
        }
    }

    fn render_gouraud(
        &mut self,
        object: &mut Object,
    ) {
        let ka: Vec3 = object.ka;
        let kd: Vec3 = object.kd;
        let ks: Vec3 = object.ks;

        let vertices_srt = self.calc_srt_convertions(&object.vertices);

        for face in object.faces.iter_mut() {
            face.calc_normal(&object.vertices);
            face.calc_centroid(&object.vertices);
            face.calc_direction(&self.camera.vrp);
            self.calc_visibility(face, &mut object.edges);
        }

        let mut faces = object.faces.clone();
        faces.sort_by(|a, b| {
            let a_depth = a.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / a.vertices.len() as f32;
            let b_depth = b.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / b.vertices.len() as f32;
            let a_depth = OrderedFloat(a_depth);
            let b_depth = OrderedFloat(b_depth);
            a_depth.cmp(&b_depth)
        });

        let mut vertex_intensities = vec![Vec3::zeros(); object.vertices.len()];
        for (i, vertex) in object.vertices.iter().enumerate() {
            let mut normal = Vec3::zeros();
            let mut count = 0;

            for face in &object.faces {
                if face.vertices.contains(&i) {
                    normal += face.normal;
                    count += 1;
                }
            }

            if count > 0 {
                normal = normal.normalize();
            }

            let direction = (self.camera.vrp - *vertex).normalize();
            let color = self.calc_color(&ka, &kd, &ks, vertex, &direction, &normal);
            vertex_intensities[i] = Vec3::new(color[0] as f32, color[1] as f32, color[2] as f32);
        }

        for face in &mut object.faces {
            if self.visibility_filter && !face.visible {
                continue;
            }

            let intersections = Self::calc_intersections_for_gouraud(&vertices_srt, &vertex_intensities, face);

            for (i, scaline_intersections) in intersections.iter() {
                if scaline_intersections.len() < 2 {
                    continue;
                }

                if !self.is_valid_i(*i) {
                    continue;
                }

                let mut counter = 0;

                while counter < scaline_intersections.len() {
                    let x0: i32 = scaline_intersections[counter].0.ceil() as i32;
                    let z0: f32 = scaline_intersections[counter].1;
                    let x1: i32 = scaline_intersections[counter + 1].0.floor() as i32;
                    let z1: f32 = scaline_intersections[counter + 1].1;

                    let r0: f32 = scaline_intersections[counter].2;
                    let g0: f32 = scaline_intersections[counter].3;
                    let b0: f32 = scaline_intersections[counter].4;
                    let r1: f32 = scaline_intersections[counter + 1].2;
                    let g1: f32 = scaline_intersections[counter + 1].3;
                    let b1: f32 = scaline_intersections[counter + 1].4;

                    counter += 2;

                    if x1 < x0 {
                        continue;
                    }

                    let dx = (x1 - x0) as f32;
                    let dz = z1 - z0;
                    let dr = r1 - r0;
                    let dg = g1 - g0;
                    let db = b1 - b0;
                    let tz = dz / dx;
                    let tr = dr / dx;
                    let tg = dg / dx;
                    let tb = db / dx;

                    let mut z: f32 = z0;
                    let mut r: f32 = r0;
                    let mut g: f32 = g0;
                    let mut b: f32 = b0;

                    for j in x0..=x1 {
                        if !self.is_valid_j(j) {
                            continue;
                        }

                        if self.can_paint(*i, j, z) {
                            let color = [
                                r as u8,
                                g as u8,
                                b as u8,
                                255,
                            ];
                            self.paint(*i as i32, j as i32, color);
                            self.set_zbuffer(*i, j, z);
                        }

                        z += tz;
                        r += tr;
                        g += tg;
                        b += tb;
                    }
                }
            }
        }
    }

    fn render_phong(
        &mut self,
        object: &mut Object,
    ) {
        let ka: Vec3 = object.ka;
        let kd: Vec3 = object.kd;
        let ks: Vec3 = object.ks;

        let vertices_srt = self.calc_srt_convertions(&object.vertices);

        for face in object.faces.iter_mut() {
            face.calc_normal(&object.vertices);
            face.calc_centroid(&object.vertices);
            face.calc_direction(&self.camera.vrp);
            self.calc_visibility(face, &mut object.edges);
        }

        let mut faces = object.faces.clone();
        faces.sort_by(|a, b| {
            let a_depth = a.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / a.vertices.len() as f32;
            let b_depth = b.vertices.iter().map(|&i| vertices_srt[i].z).sum::<f32>() / b.vertices.len() as f32;
            let a_depth = OrderedFloat(a_depth);
            let b_depth = OrderedFloat(b_depth);
            a_depth.cmp(&b_depth)
        });


        let mut vertex_normals = vec![Vec3::zeros(); object.vertices.len()];
        for (i, vertex) in object.vertices.iter().enumerate() {
            let mut normal = Vec3::zeros();
            let mut count = 0;

            for face in &object.faces {
                if face.vertices.contains(&i) {
                    normal += face.normal;
                    count += 1;
                }
            }

            if count > 0 {
                normal = normal.normalize();
            }

            vertex_normals[i] = normal;
        }

        for face in faces.iter_mut() {
            if self.visibility_filter && !face.visible {
                continue;
            }

            let intersections = Self::calc_intersections_for_phong(&vertices_srt, &vertex_normals, face);

            for (i, scaline_intersections) in intersections.iter() {
                if scaline_intersections.len() < 2 {
                    continue;
                }

                if !self.is_valid_i(*i) {
                    continue;
                }

                let mut counter = 0;

                while counter < scaline_intersections.len() {
                    let x0: i32 = scaline_intersections[counter].0.ceil() as i32;
                    let z0: f32 = scaline_intersections[counter].1;
                    let n0: Vec3 = scaline_intersections[counter].2;
                    let x1: i32 = scaline_intersections[counter + 1].0.floor() as i32;
                    let z1: f32 = scaline_intersections[counter + 1].1;
                    let n1: Vec3 = scaline_intersections[counter + 1].2;

                    counter += 2;

                    if x1 < x0 {
                        continue;
                    }

                    let dx = (x1 - x0) as f32;
                    let dz = z1 - z0;
                    let dn = n1 - n0;
                    let tz = dz / dx;
                    let tn = dn / dx;

                    let mut z: f32 = z0;
                    let mut n: Vec3 = n0;

                    for j in x0..=x1 {
                        if !self.is_valid_j(j) {
                            continue;
                        }

                        if self.can_paint(*i, j, z) {
                            let position = Vec3::new(j as f32, *i as f32, z);
                            let direction = (self.camera.vrp - position).normalize();
                            let normal = n.normalize();

                            let color = self.calc_color(&ka, &kd, &ks, &position, &direction, &normal);
                            self.paint(*i as i32, j as i32, color);
                            self.set_zbuffer(*i, j, z);
                        }

                        z += tz;
                        n += tn;
                    }
                }
            }
        }
    }
}
