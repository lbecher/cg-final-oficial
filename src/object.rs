use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::constants::*;
use crate::types::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    pub vertices: Vec<usize>,
    pub visible: bool,
    pub normal: Vec3,
    pub direction: Vec3,
    pub centroid: Vec3,
}

impl Face {
    /// Calcula a normal da face.
    pub fn calc_normal(&mut self, vertices: &Vec<Vec3>) {
        let a: Vec3 = vertices[self.vertices[0]];
        let b: Vec3 = vertices[self.vertices[1]];
        let c: Vec3 = vertices[self.vertices[2]];

        let bc: Vec3 = c - b;
        let ba: Vec3 = a - b;

        self.normal = bc.cross(&ba).normalize();
    }

    /// Calcula a direção.
    pub fn calc_direction(&mut self, vrp: &Vec3) {
        let direction: Vec3 = *vrp - self.centroid;
        self.direction = direction.normalize();
    }

    /// Calcula o centroide da face.
    pub fn calc_centroid(&mut self, vertices: &Vec<Vec3>) {
        let mut centroid = Vec3::zeros();
        for i in 0..self.vertices.len() {
            centroid = centroid + vertices[self.vertices[i]];
        }
        self.centroid = centroid / self.vertices.len() as f32;
    }
}

/// Estrutura para armazenar uma superfície BSpline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    /// Quantidades de pontos de controle na direção i.
    ni: u8,
    /// Quantidades de pontos de controle na direção j.
    nj: u8,
    /// Grau do polinômio interpolador na direção i.
    ti: u8,
    /// Grau do polinômio interpolador na direção j.
    tj: u8,
    /// Resolução na direção i.
    resi: u8,
    /// Resolução na direção j.
    resj: u8,
    /// Nós (knots) na direção i.
    knots_i: Vec<f32>,
    /// Nós (knots) na direção j.
    knots_j: Vec<f32>,

    /// Pontos de controle da superfície.
    pub control_points: Vec<Vec3>,

    /// Vértices da malha.
    pub vertices: Vec<Vec3>,
    /// Faces da malha.
    pub faces: Vec<Face>,
    /// Centroide da malha.
    pub centroid: Vec3,

    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub n: f32,

    pub closed: bool,
}

impl Object {
    pub fn new(
        ni: u8,
        nj: u8,
        ti: u8,
        tj: u8,
        resi: u8,
        resj: u8,
        smoothing_iterations: u8,
        ka: Vec3,
        kd: Vec3,
        ks: Vec3,
        n: f32,
        closed: bool,
    ) -> Self {
        let control_points: Vec<Vec3> = if !closed {
            Self::gen_control_points(ni, nj, smoothing_iterations)
        } else {
            Self::gen_closed_control_points(ni, nj)
        };

        let knots_i: Vec<f32> = Self::spline_knots(ni as usize, ti as usize);
        let knots_j: Vec<f32> = Self::spline_knots(nj as usize, tj as usize);

        let vertices: Vec<Vec3> = Vec::with_capacity(resi as usize * resj as usize);
        let faces: Vec<Face> = Vec::with_capacity((resi as usize - 1) * (resj as usize - 1));

        let mut obj = Self {
            ni,
            nj,
            ti,
            tj,
            resi,
            resj,
            knots_i,
            knots_j,

            control_points,
            vertices,
            faces,

            centroid: Vec3::zeros(),

            ka,
            kd,
            ks,
            n,

            closed,
        };

        obj.calc_mesh();
        obj.calc_edges_and_faces();
        obj.calc_centroid();

        obj
    }

    pub fn set_ni_nj_ti_tj(&mut self, ni: u8, nj: u8, ti: u8, tj: u8, smoothing_iterations: u8) {
        self.ni = ni;
        self.nj = nj;

        self.knots_i = Self::spline_knots(ni as usize, ti as usize);
        self.knots_j = Self::spline_knots(nj as usize, tj as usize);

        self.control_points = Self::gen_control_points(ni, nj, smoothing_iterations);

        self.calc_mesh();
        self.calc_edges_and_faces();
        self.calc_centroid();
    }

    pub fn set_resi_resj(&mut self, resi: u8, resj: u8) {
        self.resi = resi;
        self.resj = resj;

        self.calc_mesh();
        self.calc_edges_and_faces();
        self.calc_centroid();
    }

    //--------------------------------------------------------------------------------
    // Geração da superfície
    //--------------------------------------------------------------------------------

    fn gen_control_points(ni: u8, nj: u8, smoothing_iterations: u8) -> Vec<Vec3> {
        let mut rng = rand::thread_rng();
        let mut control_points: Vec<Vec3> = Vec::with_capacity((ni as usize + 1) * (nj as usize + 1));
        let ti = 4.0 / ni as f32;
        let tj = 4.0 / nj as f32;
        let mut i = ni as f32 / -2.0;
        let mut j;
        for _ in 0..=ni {
            j = nj as f32 / -2.0;
            for _ in 0..=nj {
                control_points.push(Vec3::new(
                    i,
                    j,
                    rng.gen_range(0.0..2.0),
                ));
                j += tj;
            }
            i += ti;
        }
        Self::smooth_control_points(&mut control_points, smoothing_iterations, ni, nj);
        control_points
    }

    fn gen_closed_control_points(ni: u8, nj: u8) -> Vec<Vec3> {
        let mut control_points: Vec<Vec3> = Vec::with_capacity((ni as usize + 1) * (nj as usize + 1));
        let step_i = 8.0 * std::f32::consts::PI / (ni as f32 + 1.0);
        let step_j = 4.0 / nj as f32;
        for i in 0..=ni {
            for j in 0..=nj {
                let x = i as f32 * step_i;
                let y = j as f32 * step_j;
                control_points.push(Vec3::new(x.cos(), y,  x.sin()));
            }
        }
        control_points
    }

    fn spline_knots(n: usize, t: usize) -> Vec<f32> {
        let mut knots = Vec::with_capacity(n + t + 1);
        for j in 0..=(n + t) {
            if j < t {
                knots.push(0.0);
            } else if j <= n {
                knots.push((j + 1 - t) as f32);
            } else {
                knots.push((n + 2 - t) as f32);
            }
        }
        knots
    }

    fn spline_blend(k: usize, t: usize, u: &[f32], v: f32) -> f32 {
        if t == 1 {
            if u[k] <= v && v < u[k + 1] {
                1.0
            } else {
                0.0
            }
        } else {
            let mut value = 0.0;
            let denom1 = u[k + t - 1] - u[k];
            let denom2 = u[k + t] - u[k + 1];

            if denom1 != 0.0 {
                value += ((v - u[k]) / denom1) * Self::spline_blend(k, t - 1, u, v);
            }
            if denom2 != 0.0 {
                value += ((u[k + t] - v) / denom2) * Self::spline_blend(k + 1, t - 1, u, v);
            }
            value
        }
    }

    fn compute_basis(knots: &[f32], n: usize, res: usize, t: usize) -> Vec<f32> {
        let mut basis = vec![0.0; (n + 1) * res];
        let increment = (n as f32 + 2.0 - t as f32) / ((res - 1) as f32);
        let epsilon = 1e-6;
        for i in 0..res {
            let raw_u = i as f32 * increment;
            let u = if raw_u < knots[knots.len() - 1] {
                raw_u
            } else {
                knots[knots.len() - 1] - epsilon
            };
            for k in 0..=n {
                basis[k * res + i] = Self::spline_blend(k, t, knots, u);
            }
        }
        basis
    }

    /// Gera a malha da superfície.
    pub fn calc_mesh(&mut self) {
        let ni = self.ni as usize;
        let nj = self.nj as usize;
        let ti = self.ti as usize;
        let tj = self.tj as usize;
        let resi = self.resi as usize;
        let resj = self.resj as usize;

        // 1) Pré-computar as funções de base para as duas direções
        let basis_i = Self::compute_basis(&self.knots_i, ni, resi, ti);
        let basis_j = Self::compute_basis(&self.knots_j, nj, resj, tj);

        // 2) Calcular os vértices em paralelo usando Rayon
        let cps = self.control_points.clone();
        let new_vertices: Vec<Vec3> = (0..resi)
            .into_par_iter()
            .flat_map_iter(|i| {
                // Clonar as variáveis para uso no iterador interno
                let basis_i_c = basis_i.clone();
                let basis_j_c = basis_j.clone();
                let cps_c = cps.clone();
                (0..resj).map(move |j| {
                    let mut sum = Vec3::zeros();
                    for ki in 0..=ni {
                        let bi = basis_i_c[ki * resi + i];
                        for kj in 0..=nj {
                            let bj = basis_j_c[kj * resj + j];
                            let blend = bi * bj;
                            let cp_idx = ki * (nj + 1) + kj;
                            sum = sum + (cps_c[cp_idx] * blend);
                        }
                    }
                    sum
                })
            })
            .collect();

        // 3) Atribuir os vértices calculados
        self.vertices = new_vertices;
    }

    /// Gera as faces triangulares da malha.
    fn calc_edges_and_faces_tri(&mut self) {
        let resi = self.resi as usize;
        let resj = self.resj as usize;
        let i_max = if self.closed { resi } else { resi - 1 };
        for i in 0..i_max {
            let next_i = if self.closed { (i + 1) % resi } else { i + 1 };
            for j in 0..resj - 1 {
                let a_index = next_i * resj + j;
                let b_index = next_i * resj + (j + 1);
                let c_index = i * resj + (j + 1);
                let d_index = i * resj + j;

                let abc_face = Face {
                    vertices: vec![a_index, b_index, c_index, a_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    direction: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(abc_face);

                let acd_face = Face {
                    vertices: vec![a_index, c_index, d_index, a_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    direction: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(acd_face);
            }
        }
    }

    fn calc_edges_and_faces_quad(&mut self) {
        let resi = self.resi as usize;
        let resj = self.resj as usize;
        let i_max = if self.closed { resi } else { resi - 1 };
        for i in 0..i_max {
            let next_i = if self.closed { (i + 1) % resi } else { i + 1 };
            for j in 0..resj - 1 {
                let a_index = next_i * resj + j;
                let b_index = next_i * resj + (j + 1);
                let c_index = i * resj + (j + 1);
                let d_index = i * resj + j;

                let face = Face {
                    vertices: vec![a_index, b_index, c_index, d_index, a_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    direction: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(face);
            }
        }
        // Removida abordagem de criação de faces extras; o fechamento é tratado pelo wrap modular.
    }

    // Novo método para gerar arestas e faces com base na flag use_triangular_faces
    pub fn calc_edges_and_faces(&mut self) {
        self.faces.clear();
        if USE_TRIANGULAR_FACES {
            self.calc_edges_and_faces_tri();
        } else {
            self.calc_edges_and_faces_quad();
        }
    }

    /// Calcula o centroide através do box envolvente.
    pub fn calc_centroid(&mut self) {
        let first: Vec3 = self.vertices[0];
        let mut min: Vec3 = first;
        let mut max: Vec3 = first;

        for p in &self.vertices {
            min.x = p.x.min(min.x);
            min.y = p.y.min(min.y);
            min.z = p.z.min(min.z);
            max.x = p.x.max(max.x);
            max.y = p.y.max(max.y);
            max.z = p.z.max(max.z);
        }

        self.centroid = Vec3::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        );
    }

    //--------------------------------------------------------------------------------
    // Métodos de transformação
    //--------------------------------------------------------------------------------

    /// Gera a matriz de translação.
    fn gen_translation_matrix(&self, translation: &Vec3) -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Aplica a transformação de translação nos pontos de controle e vértices
    pub fn translate(&mut self, translation: &Vec3) {
        let translation_matrix: Mat4 = self.gen_translation_matrix(translation);

        for control_point in &mut self.control_points {
            let cp: Mat4x1 = translation_matrix * vec3_to_mat4x1(control_point);
            *control_point = cp.xyz();
        }

        for vertex in &mut self.vertices {
            let vt: Mat4x1 = translation_matrix * vec3_to_mat4x1(vertex);
            *vertex = vt.xyz();
        }

        self.calc_centroid();
    }

    /// Aplica a transformação de escala nos pontos de controle e vértices
    pub fn scale(&mut self, scale: f32) {
        let scale_matrix: Mat4 = Mat4::new(
            scale, 0.0, 0.0, 0.0,
            0.0, scale, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let centroid = &self.centroid;
        let minus_centroid = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        for control_point in &mut self.control_points {
            let mut cp: Mat4x1 = to_centroid * vec3_to_mat4x1(control_point);
            cp = scale_matrix * cp;
            cp = to_origin * cp;
            *control_point = cp.xyz();
        }

        for vertex in &mut self.vertices {
            let mut vt: Mat4x1 = to_centroid * vec3_to_mat4x1(vertex);
            vt = scale_matrix * vt;
            vt = to_origin * vt;
            *vertex = vt.xyz();
        }
    }

    /// Aplica a transformação de rotação em X nos pontos de controle e vértices
    pub fn rotate_x(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos_theta, -sin_theta, 0.0,
            0.0, sin_theta, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let centroid = &self.centroid;
        let minus_centroid = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        for control_point in &mut self.control_points {
            let mut cp: Mat4x1 = to_centroid * vec3_to_mat4x1(control_point);
            cp = rotation_matrix * cp;
            cp = to_origin * cp;
            *control_point = cp.xyz();
        }

        for vertex in &mut self.vertices {
            let mut vt: Mat4x1 = to_centroid * vec3_to_mat4x1(vertex);
            vt = rotation_matrix * vt;
            vt = to_origin * vt;
            *vertex = vt.xyz();
        }

        //self.calc_centroid();
    }

    /// Aplica a transformação de rotação nos pontos de controle e vértices
    pub fn rotate(&mut self, rotation: &Vec3) {
        self.rotate_x(rotation.x.to_radians());
        self.rotate_y(rotation.y.to_radians());
        self.rotate_z(rotation.z.to_radians());
    }

    /// Aplica a transformação de rotação em Y nos pontos de controle e vértices
    pub fn rotate_y(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            cos_theta, 0.0, sin_theta, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin_theta, 0.0, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let centroid = &self.centroid;
        let minus_centroid = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        for control_point in &mut self.control_points {
            let mut cp: Mat4x1 = to_centroid * vec3_to_mat4x1(control_point);
            cp = rotation_matrix * cp;
            cp = to_origin * cp;
            *control_point = cp.xyz();
        }

        for vertex in &mut self.vertices {
            let mut vt: Mat4x1 = to_centroid * vec3_to_mat4x1(vertex);
            vt = rotation_matrix * vt;
            vt = to_origin * vt;
            *vertex = vt.xyz();
        }

        //self.calc_centroid();
    }

    /// Aplica a transformação de rotação em Y nos pontos de controle e vértices
    pub fn rotate_z(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            cos_theta, -sin_theta, 0.0, 0.0,
            sin_theta, cos_theta, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let centroid = &self.centroid;
        let minus_centroid = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        for control_point in &mut self.control_points {
            let mut cp: Mat4x1 = to_centroid * vec3_to_mat4x1(control_point);
            cp = rotation_matrix * cp;
            cp = to_origin * cp;
            *control_point = cp.xyz();
        }

        for vertex in &mut self.vertices {
            let mut vt: Mat4x1 = to_centroid * vec3_to_mat4x1(vertex);
            vt = rotation_matrix * vt;
            vt = to_origin * vt;
            *vertex = vt.xyz();
        }

        //self.calc_centroid();
    }


    //--------------------------------------------------------------------------------
    // Outros métodos
    //--------------------------------------------------------------------------------

    /// Suaviza as coordenadas z dos pontos de controle.
    fn smooth_control_points(
        control_points: &mut Vec<Vec3>,
        smoothing_iterations: u8,
        ni: u8,
        nj: u8,
    ) {
        let ni = ni as usize;
        let nj = nj as usize;
        for _ in 0..smoothing_iterations {
            let mut new_control_points: Vec<Vec3> = control_points.clone();

            for i in 1..(ni - 1) {
                for j in 1..(nj - 1) {
                    let index: usize = i * (nj + 1) + j;
                    let neighbors: [Vec3; 4] = [
                        control_points[(i - 1) * (nj + 1) + j],
                        control_points[(i + 1) * (nj + 1) + j],
                        control_points[i * (nj + 1) + (j - 1)],
                        control_points[i * (nj + 1) + (j + 1)],
                    ];

                    let avg_z = neighbors.iter().map(|v| v.z).sum::<f32>() / neighbors.len() as f32;
                    new_control_points[index].z = avg_z;
                }
            }

            *control_points = new_control_points;
        }
    }

    /// Calcula as normais das faces.
    pub fn calc_normals(&mut self) {
        for face in &mut self.faces {
            face.calc_normal(&self.vertices);
        }
    }
}