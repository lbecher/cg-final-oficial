use rand::Rng;
use std::f32::consts::SQRT_2;
use std::sync::{Arc, Mutex};
use std::cmp::min;
use crate::types::*;

/// Estrutura para armazenar uma superfície BSpline.
#[derive(Debug)]
pub struct Object {
    /// Quantidades de pontos de controle na direção i.
    ni: usize,
    /// Quantidades de pontos de controle na direção j.
    nj: usize,
    /// Ordem da spline (grau do polinômio interpolador) na direção i.
    ti: usize,
    /// Ordem da spline (grau do polinômio interpolador) na direção j.
    tj: usize,
    /// Resolução na direção i.
    resi: usize,
    /// Resolução na direção j.
    resj: usize,

    /// Nós (knots) na direção i.
    knots_i: Vec<f32>,
    /// Nós (knots) na direção j.
    knots_j: Vec<f32>,

    /// Pontos de controle.
    pub control_points: Vec<Vec3>,
    pub control_points_srt: Vec<Vec3>,
    /// Lista de vertices da malha interpolada.
    pub vertices: Vec<Vec3>,
    pub vertices_srt: Vec<Vec3>,
    /// Lista de faces da malha interpolada.
    faces: Vec<[usize; 4]>,

    /// Centroide dos pontos de controle
    pub centroid: Vec3,

    /// Número de iterações de suavização das coordenadas z.
    pub smoothing_iterations: usize,
}

impl Object {
    pub fn new(
        ni: usize,
        nj: usize,
        ti: usize,
        tj: usize,
        resi: usize,
        resj: usize,
        smoothing_iterations: usize,
        m_sru_srt: &Mat4,
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

        let knots_i: Vec<f32> = Self::spline_knots(ni, ti);
        let knots_j: Vec<f32> = Self::spline_knots(nj, tj);

        let vertices: Vec<Vec3> = vec![Vec3::zeros(); resi * resj];

        let mut obj = Self {
            ni,
            nj,
            ti,
            tj,
            resi,
            resj,

            knots_i,
            knots_j,

            control_points_srt: vec![Vec3::zeros(); control_points.len()],
            control_points,
            vertices_srt: vec![Vec3::zeros(); vertices.len()],
            vertices,
            faces: Vec::with_capacity((resi - 1) * (resj - 1)),

            centroid: Vec3::zeros(),

            smoothing_iterations,
        };

        obj.calc_mesh();
        obj.calc_faces();
        obj.calc_centroid();
        obj.calc_srt_convertions(m_sru_srt);

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
        let ti = self.ti;
        let tj = self.tj;
        let resi = self.resi;
        let resj = self.resj;

        // Suavizar os pontos de controle se o número de iterações for maior que zero
        if self.smoothing_iterations > 0 {
            self.smooth_control_points(self.smoothing_iterations);
        }
        
        // Zera os vértices iniciais
        for ipt in &mut self.vertices {
            *ipt = Vec3::new(0.0, 0.0, 0.0);
        }

        // Cálculo dos incrementos
        let increment_i = (self.ni as f32 - self.ti as f32 + 2.0) / resi as f32;
        let increment_j = (self.nj as f32 - self.tj as f32 + 2.0) / resj as f32;

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
                                let bi = Self::spline_blend(ki, ti, &knots_i, interval_i);
                                let bj = Self::spline_blend(kj, tj, &knots_j, interval_j);

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
    fn calc_faces(&mut self) {
        self.faces.clear();
        for i in 0..self.resi - 1 {
            for j in 0..self.resj - 1 {
                self.faces.push([
                    i * self.resj + j,
                    i * self.resj + (j + 1),
                    (i + 1) * self.resj + (j + 1),
                    (i + 1) * self.resj + j,
                ]);
            }
        }
    }

    /// Gera as conversões dos pontos de controle e vértices para o sistema de referência da tela.
    pub fn calc_srt_convertions(&mut self, m_sru_srt: &Mat4) {
        std::thread::scope(|s| {
            s.spawn(|| {
                // Calcula os pontos de controle no sistema de referência da tela
                for (i, control_point) in self.control_points.iter().enumerate() {
                    let mut control_point_srt: Mat4x1 = m_sru_srt * vec3_to_mat4x1(control_point);
                    control_point_srt.x /= control_point_srt.w;
                    control_point_srt.y /= control_point_srt.w;
                    self.control_points_srt[i] = control_point_srt.xyz();
                }
            });  
            s.spawn(|| {
                // Calcilando os vértices no sistema de referência da tela
                for (i, vertex) in self.vertices.iter().enumerate() {
                    let mut vertex_srt: Mat4x1 = m_sru_srt * vec3_to_mat4x1(vertex);
                    vertex_srt.x /= vertex_srt.w;
                    vertex_srt.y /= vertex_srt.w;
                    self.vertices_srt[i] = vertex_srt.xyz();
                }
            });
        });
    }

    //--------------------------------------------------------------------------------
    // Métodos de transformação
    //--------------------------------------------------------------------------------

    /// Gera a matriz de translação
    pub fn gen_translation_matrix(&self, translation: &Mat4x1) -> Mat4 {
        Mat4::new(
            1.0, 0.0, 0.0, translation.x,
            0.0, 1.0, 0.0, translation.y,
            0.0, 0.0, 1.0, translation.z,
            0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Aplica a transformação de translação nos pontos de controle e vértices
    pub fn translate(&mut self, translation: &Mat4x1, m_sru_srt: &Mat4) {
        let sru_translation_matrix: Mat4 = self.gen_translation_matrix(translation);
        let srt_translation: Mat4x1 = m_sru_srt * translation;
        let srt_translation_matrix: Mat4 = self.gen_translation_matrix(&srt_translation);
        
        std::thread::scope(|s| {
            s.spawn(|| {
                // Transladar os pontos de controle
                for control_point in &mut self.control_points {
                    let cp: Mat4x1 = sru_translation_matrix * vec3_to_mat4x1(control_point);
                    *control_point = cp.xyz();
                }
            });
            s.spawn(|| {
                // Transladar os pontos de controle no sistema de referência da tela
                for control_point_srt in &mut self.control_points_srt {
                    let cp: Mat4x1 = srt_translation_matrix * vec3_to_mat4x1(control_point_srt);
                    *control_point_srt = cp.xyz();
                }
            });
            s.spawn(|| {
                // Transladar os vértices
                for vertex in &mut self.vertices {
                    let vt: Mat4x1 = sru_translation_matrix * vec3_to_mat4x1(vertex);
                    *vertex = vt.xyz();
                }
            });
            s.spawn(|| {
                // Transladar os vértices no sistema de referência da tela
                for vertex_srt in &mut self.vertices_srt {
                    let vt: Mat4x1 = srt_translation_matrix * vec3_to_mat4x1(vertex_srt);
                    *vertex_srt = vt.xyz();
                }
            });
        });

        
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

        std::thread::scope(|s| {
            s.spawn(|| {
                // Escala os pontos de controle
                for control_point in &mut self.control_points {
                    let cp: Mat4x1 = scale_matrix * vec3_to_mat4x1(control_point);
                    *control_point = cp.xyz();
                }
            });
            s.spawn(|| {
                // Escala os pontos de controle no sistema de referência da tela
                for control_point_srt in &mut self.control_points_srt {
                    let cp: Mat4x1 = scale_matrix * vec3_to_mat4x1(control_point_srt);
                    *control_point_srt = cp.xyz();
                }
            });
            s.spawn(|| {
                // Escala os vértices
                for vertex in &mut self.vertices {
                    let vt: Mat4x1 = scale_matrix * vec3_to_mat4x1(vertex);
                    *vertex = vt.xyz();
                }
            });
            s.spawn(|| {
                // Escala os vértices no sistema de referência da tela
                for vertex_srt in &mut self.vertices_srt {
                    let vt: Mat4x1 = scale_matrix * vec3_to_mat4x1(vertex_srt);
                    *vertex_srt = vt.xyz();
                }
            });
        });

        self.calc_centroid();
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

        let centroid: Mat4x1 = vec3_to_mat4x1(&self.centroid);
        let minus_centroid: Mat4x1 = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        let srt_centroid: Mat4x1 = m_sru_srt * centroid;
        let srt_minus_centroid: Mat4x1 = -srt_centroid;
        let srt_to_origin: Mat4 = self.gen_translation_matrix(&srt_centroid);
        let srt_to_centroid: Mat4 = self.gen_translation_matrix(&srt_minus_centroid);

        std::thread::scope(|s| {
            s.spawn(|| {
                // Rotacionar os pontos de controle
                for control_point in &mut self.control_points {
                    let cp: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(control_point)));
                    *control_point = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os pontos de controle no sistema de referência da tela
                for control_point_srt in &mut self.control_points_srt {
                    let cp: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(control_point_srt)));
                    *control_point_srt = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices
                for vertex in &mut self.vertices {
                    let vt: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(vertex)));
                    *vertex = vt.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices no sistema de referência da tela
                for vertex_srt in &mut self.vertices_srt {
                    let vt: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(vertex_srt)));
                    *vertex_srt = vt.xyz();
                }
            });
        });

        self.calc_centroid();
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

        let centroid: Mat4x1 = vec3_to_mat4x1(&self.centroid);
        let minus_centroid: Mat4x1 = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        let srt_centroid: Mat4x1 = m_sru_srt * centroid;
        let srt_minus_centroid: Mat4x1 = -srt_centroid;
        let srt_to_origin: Mat4 = self.gen_translation_matrix(&srt_centroid);
        let srt_to_centroid: Mat4 = self.gen_translation_matrix(&srt_minus_centroid);

        std::thread::scope(|s| {
            s.spawn(|| {
                // Rotacionar os pontos de controle
                for control_point in &mut self.control_points {
                    let cp: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(control_point)));
                    *control_point = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os pontos de controle no sistema de referência da tela
                for control_point_srt in &mut self.control_points_srt {
                    let cp: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(control_point_srt)));
                    *control_point_srt = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices
                for vertex in &mut self.vertices {
                    let vt: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(vertex)));
                    *vertex = vt.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices no sistema de referência da tela
                for vertex_srt in &mut self.vertices_srt {
                    let vt: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(vertex_srt)));
                    *vertex_srt = vt.xyz();
                }
            });
        });

        self.calc_centroid();
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

        let centroid: Mat4x1 = vec3_to_mat4x1(&self.centroid);
        let minus_centroid: Mat4x1 = -centroid;
        let to_origin: Mat4 = self.gen_translation_matrix(&centroid);
        let to_centroid: Mat4 = self.gen_translation_matrix(&minus_centroid);

        let srt_centroid: Mat4x1 = m_sru_srt * centroid;
        let srt_minus_centroid: Mat4x1 = -srt_centroid;
        let srt_to_origin: Mat4 = self.gen_translation_matrix(&srt_centroid);
        let srt_to_centroid: Mat4 = self.gen_translation_matrix(&srt_minus_centroid);

        std::thread::scope(|s| {
            s.spawn(|| {
                // Rotacionar os pontos de controle
                for control_point in &mut self.control_points {
                    let cp: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(control_point)));
                    *control_point = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os pontos de controle no sistema de referência da tela
                for control_point_srt in &mut self.control_points_srt {
                    let cp: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(control_point_srt)));
                    *control_point_srt = cp.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices
                for vertex in &mut self.vertices {
                    let vt: Mat4x1 = to_centroid * (rotation_matrix * (to_origin * vec3_to_mat4x1(vertex)));
                    *vertex = vt.xyz();
                }
            });
            s.spawn(|| {
                // Rotacionar os vértices no sistema de referência da tela
                for vertex_srt in &mut self.vertices_srt {
                    let vt: Mat4x1 = srt_to_centroid * (rotation_matrix * (srt_to_origin * vec3_to_mat4x1(vertex_srt)));
                    *vertex_srt = vt.xyz();
                }
            });
        });

        self.calc_centroid();
    }


    //--------------------------------------------------------------------------------
    // Outros métodos
    //--------------------------------------------------------------------------------

    /// Calcula o centroide dos pontos de controle
    pub fn calc_centroid(&mut self) {
        let mut centroid = Vec3::zeros();
        for p in &self.control_points {
            centroid += *p;
        }
        self.centroid = centroid / self.control_points.len() as f32;
    }

    /// Suaviza as coordenadas z dos pontos de controle
    fn smooth_control_points(&mut self, iterations: usize) {
        for _ in 0..iterations {
            let mut new_control_points = self.control_points.clone();

            for i in 1..(self.ni - 1) {
                for j in 1..(self.nj - 1) {
                    let idx = i * (self.nj + 1) + j;
                    let neighbors = [
                        self.control_points[(i - 1) * (self.nj + 1) + j],
                        self.control_points[(i + 1) * (self.nj + 1) + j],
                        self.control_points[i * (self.nj + 1) + (j - 1)],
                        self.control_points[i * (self.nj + 1) + (j + 1)],
                    ];

                    let avg_z = neighbors.iter().map(|v| v.z).sum::<f32>() / neighbors.len() as f32;
                    new_control_points[idx].z = avg_z;
                }
            }

            self.control_points = new_control_points;
        }
    }
}