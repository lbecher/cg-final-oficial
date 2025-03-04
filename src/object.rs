use rand::Rng;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use crate::constants::*;
use crate::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub vertices: [usize; 2],
    pub visible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    pub vertices: Vec<usize>,
    pub edges: Vec<usize>,
    pub visible: bool,
    pub normal: Vec3,
    pub centroid: Vec3,
}

impl Face {
    /// Calcula a normal da face.
    pub fn calc_normal(&mut self, vertices: &Vec<Vec3>) {
        let a = vertices[self.vertices[0]];
        let b = vertices[self.vertices[1]];
        let c = vertices[self.vertices[2]];

        let bc = c - b;
        let ba = a - c;

        self.normal = bc.cross(&ba).normalize();
    }

    /// Calcula o centroide da face.
    pub fn calc_centroid(&mut self, vertices: &Vec<Vec3>) {
        let mut centroid = Vec3::zeros();
        for i in 0..3 {
            centroid = centroid + vertices[self.vertices[i]];
        }
        self.centroid = centroid / 3.0
    }
}

/// Estrutura para armazenar uma superfície BSpline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    /// Quantidades de pontos de controle na direção i.
    ni: usize,
    /// Quantidades de pontos de controle na direção j.
    nj: usize,
    /// Resolução na direção i.
    resi: usize,
    /// Resolução na direção j.
    resj: usize,
    /// Nós (knots) na direção i.
    knots_i: Vec<f32>,
    /// Nós (knots) na direção j.
    knots_j: Vec<f32>,

    /// Pontos de controle da superfície.
    pub control_points: Vec<Vec3>,

    /// Vértices da malha.
    pub vertices: Vec<Vec3>,
    /// Arestas da malha.
    pub edges: Vec<Edge>,
    /// Faces da malha.
    pub faces: Vec<Face>,
    /// Centroide da malha.
    pub centroid: Vec3,

    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub n: f32,
}

impl Object {
    pub fn new(
        ni: usize,
        nj: usize,
        resi: usize,
        resj: usize,
        smoothing_iterations: u8,
        ka: Vec3,
        kd: Vec3,
        ks: Vec3,
        n: f32,
    ) -> Self {
        let control_points: Vec<Vec3> = Self::gen_control_points(ni, nj, smoothing_iterations);

        let knots_i: Vec<f32> = Self::spline_knots(ni, TI);
        let knots_j: Vec<f32> = Self::spline_knots(nj, TJ);

        let vertices: Vec<Vec3> = Vec::with_capacity(resi * resj);
        let edges: Vec<Edge> = Vec::with_capacity(resj*(resi - 1) + resi*(resj - 1));
        let faces: Vec<Face> = Vec::with_capacity((resi - 1) * (resj - 1));

        let mut obj = Self {
            ni,
            nj,
            resi,
            resj,
            knots_i,
            knots_j,

            control_points,
            vertices,
            edges,
            faces,

            centroid: Vec3::zeros(),

            ka,
            kd,
            ks,
            n,
        };

        obj.calc_mesh();
        obj.calc_edges_and_faces();
        obj.calc_centroid();

        obj
    }

    pub fn set_ni_nj_resi_resj(&mut self, ni: usize, nj: usize, smoothing_iterations: u8, resi: usize, resj: usize) {
        self.ni = ni;
        self.nj = nj;
        self.resi = resi;
        self.resj = resj;

        self.knots_i = Self::spline_knots(ni, TI);
        self.knots_j = Self::spline_knots(nj, TJ);

        self.control_points = Self::gen_control_points(self.ni, self.nj, smoothing_iterations);

        self.calc_mesh();
        self.calc_edges_and_faces();
        self.calc_centroid();
    }

    pub fn get_ni_nj(&self) -> (usize, usize) {
        (self.ni, self.nj)
    }

    pub fn get_resi_resj(&self) -> (usize, usize) {
        (self.resi, self.resj)
    }

    //--------------------------------------------------------------------------------
    // Geração da superfície
    //--------------------------------------------------------------------------------

    fn gen_control_points(ni: usize, nj: usize, smoothing_iterations: u8) -> Vec<Vec3> {
        let mut rng = rand::thread_rng();
        let mut control_points: Vec<Vec3> = Vec::with_capacity((ni + 1) * (nj + 1));
        for i in 0..=ni {
            for j in 0..=nj {
                control_points.push(Vec3::new(
                    i as f32,
                    j as f32,
                    rng.gen_range(0.0..2.0),
                ));
            }
        }
        Self::smooth_control_points(&mut control_points, smoothing_iterations, ni, nj);
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
        let ni = self.ni;
        let nj = self.nj;
        let resi = self.resi;
        let resj = self.resj;

        // 1) Pré-computar as funções de base para as duas direções
        let basis_i = Self::compute_basis(&self.knots_i, ni, resi, TI);
        let basis_j = Self::compute_basis(&self.knots_j, nj, resj, TJ);

        // 2) Alocar o vetor final (vazio) onde colocaremos todos os vértices
        let mut new_vertices = vec![Vec3::zeros(); resi * resj];

        // 3) Calcular quantas threads e dividir linhas em blocos
        let n_threads = crate::utils::num_cpu_threads();
        let chunk_size = resi / n_threads;
        let remainder = resi % n_threads;

        // Clonar para passar às threads
        let basis_i_arc = Arc::new(basis_i);
        let basis_j_arc = Arc::new(basis_j);
        let cps_arc = Arc::new(self.control_points.clone());

        let mut handles = Vec::with_capacity(n_threads);
        let mut ranges = Vec::with_capacity(n_threads);

        // Vamos determinar o intervalo de linhas [start_i..end_i) que cada thread processa
        let mut current_i = 0;
        for thread_id in 0..n_threads {
            // Quantas linhas esta thread vai pegar
            let lines_in_this_chunk = chunk_size + if thread_id < remainder { 1 } else { 0 };
            let start_i = current_i;
            let end_i = start_i + lines_in_this_chunk;
            current_i = end_i;

            // Clonamos as referências (Arc) para usar dentro da thread
            let basis_i_ref = Arc::clone(&basis_i_arc);
            let basis_j_ref = Arc::clone(&basis_j_arc);
            let cps_ref = Arc::clone(&cps_arc);

            // Criamos a thread
            let handle = std::thread::spawn(move || {
                // Aloca vetor parcial para as (lines_in_this_chunk * resj) posições
                let mut partial = vec![Vec3::zeros(); lines_in_this_chunk * resj];

                // Para cada linha local
                for local_i in 0..lines_in_this_chunk {
                    // i = linha global, mas relativo ao start_i deste chunk
                    let i = start_i + local_i;

                    for j in 0..resj {
                        let mut sum = Vec3::zeros();

                        // Recupera funções de base pré-calculadas:
                        //   bi = basis_i_ref[ki * resi + i]
                        //   bj = basis_j_ref[kj * resj + j]
                        // e soma contribuições dos pontos de controle
                        for ki in 0..=ni {
                            let bi = basis_i_ref[ki * resi + i];
                            for kj in 0..=nj {
                                let bj = basis_j_ref[kj * resj + j];
                                let blend = bi * bj;

                                let cp_idx = ki * (nj + 1) + kj;
                                sum = sum + (cps_ref[cp_idx] * blend);
                            }
                        }

                        partial[local_i * resj + j] = sum;
                    }
                }

                partial // Retorna o vetor parcial ao final
            });

            // Guardamos o handle da thread e o intervalo que ela processou
            handles.push(handle);
            ranges.push((start_i, end_i));
        }

        // 4) Juntamos todos os pedaços no vetor final `new_vertices`
        for (handle, (start_i, end_i)) in handles.into_iter().zip(ranges.into_iter()) {
            // Esperamos a thread terminar e recuperar o vetor parcial
            let partial = handle.join().unwrap();

            // Copia `partial` na posição certa de `new_vertices`
            // Cada `local_i` corresponde a i global = start_i + local_i
            // Tamanho de partial = (end_i - start_i) * resj
            let mut offset = start_i * resj;
            for vtx in partial {
                new_vertices[offset] = vtx;
                offset += 1;
            }
        }

        // 5) Finalmente, atribuir ao self.vertices
        self.vertices = new_vertices;
    }

    /// Gera as faces da malha.
    fn calc_edges_and_faces(&mut self) {
        self.faces.clear();
        for i in 0..self.resi - 1 {
            for j in 0..self.resj - 1 {
                // Nada novo sob o sol, apenas mantendo o sentido anti-horário.
                let a_index = (i + 1) * self.resj + j;
                let b_index = (i + 1) * self.resj + (j + 1);
                let c_index = i * self.resj + (j + 1);
                let d_index = i * self.resj + j;

                // IMPORTANTE!
                // Usamos as arestas (edges) apenas para desenhar as arestas do Wireframe.

                // Devemos garantir que as arestas sejam únicas.
                // Para isso, vamos sempre adicionar as arestas
                // da diagonal, da direita e de cima como novas arestas.
                let ac_index = self.edges.len();
                self.edges.push(Edge { vertices: [a_index, c_index], visible: false });

                let bc_index = ac_index + 1;
                self.edges.push(Edge { vertices: [b_index, c_index], visible: false });

                let cd_index = bc_index + 1;
                self.edges.push(Edge { vertices: [c_index, d_index], visible: false });

                // Já as arestas da esquerda e de baixo são adicionadas
                // como novas caso estejam na posição inicial. Caso contrário,
                // usamos as arestas já existentes das faces anteriores.
                let ab_index: usize;
                if i == 0 {
                    ab_index = self.edges.len();
                    self.edges.push(Edge { vertices: [a_index, b_index], visible: false });
                } else {
                    let index = (i - 1) * self.resj + (j * 2) + 1;
                    ab_index = self.faces[index].edges[1];
                };

                let da_index: usize;
                if j == 0 {
                    da_index = self.edges.len();
                    self.edges.push(Edge { vertices: [d_index, a_index], visible: false });
                } else {
                    let index = i * self.resj + ((j - 1) * 2);
                    da_index = self.faces[index].edges[1];
                };

                // Usamos as faces para o peenchimento.
                // Armazenamos os índices das arestas para depois
                // poder modificar o atributo visible com o resultado do teste
                // de visibilidade e atribuir a cor correta da aretas.
                let abc_face = Face {
                    vertices: vec![a_index, b_index, c_index, a_index],
                    edges: vec![ab_index, bc_index, ac_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(abc_face);

                let acd_face = Face {
                    vertices: vec![a_index, c_index, d_index, a_index],
                    edges: vec![ac_index, cd_index, da_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(acd_face);
                /* 
                // Nada novo sob o sol, apenas mantendo o sentido anti-horário.
                let a_index = (i + 1) * self.resj + j;
                let b_index = (i + 1) * self.resj + (j + 1);
                let c_index = i * self.resj + (j + 1);
                let d_index = i * self.resj + j;

                // Usamos as arestas para desenhar as bordas.
                // Devemos garantir que as arestas sejam únicas.
                // Para isso, vamos sempre adicionar as arestas da direita e de cima
                // como novas arestas.
                let cb_index = self.edges.len();
                self.edges.push(Edge { vertices: [c_index, b_index], visible: false });

                let dc_index = cb_index + 1;
                self.edges.push(Edge { vertices: [d_index, c_index], visible: false });

                // Já as arestas da esquerda e de baixo são adicionadas
                // como novas caso estejam na posição inicial. Caso contrário,
                // usamos as arestas já existentes das faces anteriores.
                let ab_index: usize;
                if i == 0 {
                    ab_index = self.edges.len();
                    self.edges.push(Edge { vertices: [a_index, b_index], visible: false });
                } else {
                    ab_index = self.faces[(i - 1) * (self.resj - 1) + j].edges[3];
                };

                let da_index: usize;
                if j == 0 {
                    da_index = self.edges.len();
                    self.edges.push(Edge { vertices: [d_index, a_index], visible: false });
                } else {
                    da_index = self.faces[i * (self.resj - 1) + (j - 1)].edges[2];
                };

                // Usamos as faces para o peenchimento.
                // Armazenamos os índices das arestas para poder
                // associar a face para obtermos o resultado do teste
                // de visibilidade para atribuir a cor correta.
                let face = Face {
                    vertices: [a_index, b_index, c_index, d_index, a_index],
                    edges: [ab_index, cb_index, dc_index, da_index],
                    visible: false,
                    normal: Vec3::zeros(),
                    centroid: Vec3::zeros(),
                };
                self.faces.push(face);
                */
            }
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
        ni: usize,
        nj: usize,
    ) {
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