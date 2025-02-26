use rand::Rng;
use std::sync::{Arc, Mutex};
use std::cmp::min;
use crate::constants::*;
use crate::types::*;

pub struct Edge {
    pub vertices: [usize; 2],
}

pub struct Face {
    pub vertices: [usize; 4],
    pub edges: [usize; 4],
    pub visible: bool,
}

/// Estrutura para armazenar uma superfície BSpline.
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

    pub control_points: Vec<Vec3>,
    pub control_points_srt: Vec<Vec3>,
    pub vertices: Vec<Vec3>,
    pub vertices_srt: Vec<Vec3>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,

    /// Centroide dos pontos de controle.
    pub centroid: Vec3,

    /// Número de iterações de suavização das coordenadas z.
    pub smoothing_iterations: u8,

    // TODO: Adicionar propriedades de cores
}

impl Object {
    pub fn new(
        ni: usize,
        nj: usize,
        resi: usize,
        resj: usize,
        smoothing_iterations: u8,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let mut control_points: Vec<Vec3> = Vec::with_capacity((ni + 1) * (nj + 1));
        for i in 0..=ni {
            for j in 0..=nj {
                control_points.push(Vec3::new(
                    i as f32,
                    j as f32,
                    rng.gen_range(0.0..10.0),
                ));
            }
        }
        Self::smooth_control_points(&mut control_points, smoothing_iterations, ni, nj);
        let control_points_srt: Vec<Vec3> = Vec::with_capacity((ni + 1) * (nj + 1));

        let knots_i: Vec<f32> = Self::spline_knots(ni, 3);
        let knots_j: Vec<f32> = Self::spline_knots(nj, 3);

        let vertices: Vec<Vec3> = Vec::with_capacity(resi * resj);
        let vertices_srt: Vec<Vec3> = Vec::with_capacity(resi * resj);

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
            control_points_srt,
            vertices,
            vertices_srt,
            edges: Vec::with_capacity(resj*(resi - 1) + resi*(resj - 1)),
            faces: Vec::with_capacity((resi - 1) * (resj - 1)),

            centroid: Vec3::zeros(),

            smoothing_iterations,
        };

        obj.calc_mesh();
        obj.calc_edges_and_faces();
        obj.calc_centroid();

        obj
    }

    //--------------------------------------------------------------------------------
    // Geração da superfície
    //--------------------------------------------------------------------------------

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

    /// Gera a malha da superfície.
    pub fn calc_mesh(&mut self) {
        let ni = self.ni;
        let nj = self.nj;
        let resi = self.resi;
        let resj = self.resj;

        // Zera os vértices
        self.vertices.clear();
        for _ in 0..self.resi*self.resj {
            self.vertices.push(Vec3::zeros());
        }

        // Cálculo dos incrementos
        let increment_i = (self.ni as f32 - TI as f32 + 2.0) / resi as f32;
        let increment_j = (self.nj as f32 - TJ as f32 + 2.0) / resj as f32;

        // Vamos obter o número de threads
        let n_threads = crate::utils::num_cpu_threads();

        // Dividimos as linhas (i) em blocos para cada thread processar
        // Pegamos o resto da divisão para distribuir as iterações que sobraram entre as threads
        let chunk_size = resi / n_threads;
        let remainder = resi % n_threads;

        // Para facilitar o acesso concorrente, usamos Arc para ler e escrever de forma segura.
        // - knots_i e knots_j, control_points são apenas lidos (podem ser compartilhados sem Mutex).
        // - vertices precisa de Mutex (semáforo) para escrita simultânea.
        let arc_knots_i = Arc::new(self.knots_i.clone());
        let arc_knots_j = Arc::new(self.knots_j.clone());
        let arc_control_points = Arc::new(self.control_points.clone());

        // Precisamos de Mutex para poder escrever em 'vertices' paralelamente
        let arc_vertices = Arc::new(Mutex::new(vec![Vec3::zeros(); resi * resj]));

        // Vetor de handles de thread
        let mut handles = Vec::with_capacity(n_threads);

        for i in 0..n_threads {
            // Calculamos o intervalo de linhas (i) que essa thread irá processar
            let start_i = i * chunk_size + min(i, remainder);
            let end_i = start_i + chunk_size + if i < remainder { 1 } else { 0 };

            // Clonamos as referências compartilhadas
            let knots_i = Arc::clone(&arc_knots_i);
            let knots_j = Arc::clone(&arc_knots_j);
            let control_points = Arc::clone(&arc_control_points);
            let vertices = Arc::clone(&arc_vertices);

            // Spawn da thread
            let handle = std::thread::spawn(move || {
                // Vetor local para armazenar o resultado parcial
                let mut local_vertices =
                    vec![Vec3::new(0.0, 0.0, 0.0); (end_i - start_i) * resj];

                // Iniciamos o intervalo de i de acordo com start_i
                let mut interval_i = start_i as f32 * increment_i;

                for i in start_i..end_i {
                    let mut interval_j = 0.0;

                    for j in 0..resj {
                        let local_idx = (i - start_i) * resj + j;

                        // Soma as contribuições de cada ponto de controle
                        for ki in 0..=ni {
                            for kj in 0..=nj {
                                let bi = Self::spline_blend(ki, TI, &knots_i, interval_i);
                                let bj = Self::spline_blend(kj, TJ, &knots_j, interval_j);

                                let blend = bi * bj;
                                let cp_idx = ki * (nj + 1) + kj;

                                local_vertices[local_idx] =
                                    local_vertices[local_idx] + control_points[cp_idx] * blend;
                            }
                        }
                        interval_j += increment_j;
                    }
                    interval_i += increment_i;
                }

                // Copiar o resultado parcial para o vetor global
                {
                    let mut global_vertices = vertices.lock().unwrap();
                    for i_local in 0..(end_i - start_i) {
                        for j_local in 0..resj {
                            let local_out_idx = i_local * resj + j_local;
                            let global_out_idx = (start_i + i_local) * resj + j_local;
                            global_vertices[global_out_idx] = local_vertices[local_out_idx];
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Esperar todas as threads terminarem
        for handle in handles {
            handle.join().unwrap();
        }

        // Recuperar os vértices calculados para dentro de self.vertices
        {
            let final_vertices = arc_vertices.lock().unwrap();
            self.vertices.copy_from_slice(&final_vertices);
        }
    }

    /// Gera as faces da malha.
    fn calc_edges_and_faces(&mut self) {
        self.faces.clear();
        for i in 0..self.resi - 1 {
            for j in 0..self.resj - 1 {
                // Nada novo sob o sol, apenas mantendo o sentido anti-horário.
                let a_index = i * self.resj + j;
                let b_index = i * self.resj + (j + 1);
                let c_index = (i + 1) * self.resj + (j + 1);
                let d_index = (i + 1) * self.resj + j;

                // Usamos as arestas para desenhar as bordas.
                // Devemos garantir que as arestas sejam únicas.
                // Para isso, vamos sempre adicionar as arestas da direita e de cima
                // como novas arestas.
                let cb_index = self.edges.len();
                self.edges.push(Edge { vertices: [c_index, b_index] });

                let dc_index = cb_index + 1;
                self.edges.push(Edge { vertices: [d_index, c_index] });

                // Já as arestas da esquerda e de baixo são adicionadas
                // como novas caso estejam na posição inicial. Caso contrário,
                // usamos as arestas já existentes das faces anteriores.
                let ab_index: usize;
                if i == 0 {
                    ab_index = self.edges.len();
                    self.edges.push(Edge { vertices: [a_index, b_index] });
                } else {
                    ab_index = self.faces[(i - 1) * (self.resj - 1) + j].edges[3];
                };

                let da_index: usize;
                if j == 0 {
                    da_index = self.edges.len();
                    self.edges.push(Edge { vertices: [d_index, a_index] });
                } else {
                    da_index = self.faces[i * (self.resj - 1) + (j - 1)].edges[2];
                };

                // Usamos as faces para o peenchimento.
                // Armazenamos os índices das arestas para poder
                // associar a face para obtermos o resultado do teste
                // de visibilidade para atribuir a cor correta.
                self.faces.push(Face {
                    vertices: [a_index, b_index, c_index, d_index],
                    edges: [da_index, ab_index, cb_index, dc_index],
                    visible: true,
                });
            }
        }
    }

    /// Calcula o centroide através do box envolvente dos pontos de controle.
    pub fn calc_centroid(&mut self) {
        if self.control_points.is_empty() {
            self.centroid = Vec3::zeros();
            return;
        }

        let first: Vec3 = self.control_points[0];
        let mut min: Vec3 = first;
        let mut max: Vec3 = first;

        for p in &self.control_points {
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

    /// Gera as conversões dos pontos de controle e vértices para o sistema de referência da tela.
    pub fn calc_srt_convertions(&mut self, m_sru_srt: &Mat4) {
        std::thread::scope(|s| {
            s.spawn(|| {
                // Calcula os pontos de controle no sistema de referência da tela
                self.control_points_srt.clear();
                for i in 0..self.control_points.len() {
                    let mut control_point_srt: Mat4x1 = m_sru_srt * vec3_to_mat4x1(&self.control_points[i]);
                    control_point_srt.x /= control_point_srt.w;
                    control_point_srt.y /= control_point_srt.w;
                    self.control_points_srt.push(control_point_srt.xyz());
                }
            });
            s.spawn(|| {
                // Calcilando os vértices no sistema de referência da tela
                self.vertices_srt.clear();
                for i in 0..self.vertices.len() {
                    let mut vertex_srt: Mat4x1 = m_sru_srt * vec3_to_mat4x1(&self.vertices[i]);
                    vertex_srt.x /= vertex_srt.w;
                    vertex_srt.y /= vertex_srt.w;
                    self.vertices_srt.push(vertex_srt.xyz());
                }
            });
        });
    }

    //--------------------------------------------------------------------------------
    // Métodos de transformação
    //--------------------------------------------------------------------------------

    /// Gera a matriz de translação
    pub fn gen_translation_matrix(&self, translation: &Vec3) -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Aplica a transformação de translação nos pontos de controle e vértices
    pub fn translate(&mut self, translation: &Vec3, m_sru_srt: &Mat4) {
        let sru_translation_matrix: Mat4 = self.gen_translation_matrix(translation);

        // Atualiza pontos de controle
        for cp in &mut self.control_points {
            *cp = (sru_translation_matrix * vec3_to_mat4x1(cp)).xyz();
        }

        // Atualiza vértices
        for vt in &mut self.vertices {
            *vt = (sru_translation_matrix * vec3_to_mat4x1(vt)).xyz();
        }

        self.calc_centroid();
        self.calc_srt_convertions(m_sru_srt);
    }

    /// Aplica a transformação de escala nos pontos de controle e vértices
    pub fn scale(&mut self, scale: f32, m_sru_srt: &Mat4) {
        let scale_matrix: Mat4 = Mat4::new(
            scale, 0.0, 0.0, 0.0,
            0.0, scale, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // Atualiza pontos de controle
        for cp in &mut self.control_points {
            *cp = (scale_matrix * vec3_to_mat4x1(cp)).xyz();
        }

        // Atualiza vértices
        for vt in &mut self.vertices {
            *vt = (scale_matrix * vec3_to_mat4x1(vt)).xyz();
        }

        self.calc_centroid();
        self.calc_srt_convertions(m_sru_srt);
    }

    /// Aplica a transformação de rotação em X nos pontos de controle e vértices
    pub fn rotate_x(&mut self, angle: f32, m_sru_srt: &Mat4) {
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

        self.calc_centroid();
        self.calc_srt_convertions(m_sru_srt);
    }

    /// Aplica a transformação de rotação em Y nos pontos de controle e vértices
    pub fn rotate_y(&mut self, angle: f32, m_sru_srt: &Mat4) {
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

        self.calc_centroid();
        self.calc_srt_convertions(m_sru_srt);
    }

    /// Aplica a transformação de rotação em Y nos pontos de controle e vértices
    pub fn rotate_z(&mut self, angle: f32, m_sru_srt: &Mat4) {
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

        self.calc_centroid();
        self.calc_srt_convertions(m_sru_srt);
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
}