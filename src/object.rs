use rand::Rng;
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

    /// Lista de vertices da malha interpolada.
    vertices: Vec<Vec3>,
    /// Lista de faces da malha interpolada.
    faces: Vec<[usize; 4]>,
}

impl Object {
    pub fn new(
        ni: usize,
        nj: usize,
        ti: usize,
        tj: usize,
        resi: usize,
        resj: usize,
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

        let mut obj = Self {
            ni,
            nj,
            ti,
            tj,
            resi,
            resj,
            control_points,

            knots_i,
            knots_j,

            vertices: vec![Vec3::zeros(); resi * resj],
            faces: Vec::with_capacity((resi - 1) * (resj - 1)),
        };

        obj.gen_mesh();

        obj
    }

    /// Gera a malha da superfície.
    pub fn gen_mesh(&mut self) {
        // Zera os vértices iniciais
        for ipt in &mut self.vertices {
            *ipt = Vec3::new(0.0, 0.0, 0.0);
        }

        // Cálculo dos incrementos
        let increment_i = (self.ni as f32 - self.ti as f32 + 2.0) / self.resi as f32;
        let increment_j = (self.nj as f32 - self.tj as f32 + 2.0) / self.resj as f32;

        // Vamos obter o número de threads
        let n_threads = crate::utils::num_cpu_threads();

        // Dividimos as linhas (i) em blocos para cada thread processar
        // Pegamos o resto da divisão para distribuir as iterações que sobraram entre as threads
        let chunk_size = self.resi / n_threads;
        let remainder = self.resi % n_threads;

        // Para facilitar o acesso concorrente, usamos Arc para ler e escrever de forma segura.
        // - knots_i e knots_j, control_points são apenas lidos (podem ser compartilhados sem Mutex).
        // - vertices precisa de Mutex (semáforo) para escrita simultânea.
        let arc_knots_i = Arc::new(self.knots_i.clone());
        let arc_knots_j = Arc::new(self.knots_j.clone());
        let arc_control_points = Arc::new(self.control_points.clone());

        // Precisamos de Mutex para poder escrever em 'vertices' paralelamente
        let arc_vertices = Arc::new(Mutex::new(vec![Vec3::zeros(); self.resi * self.resj]));

        // Clonamos valores necessários (por simplicidade, eles podem ser copiados ou clonados)
        let ni = self.ni;
        let nj = self.nj;
        let ti = self.ti;
        let tj = self.tj;
        let resi = self.resi;
        let resj = self.resj;

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

                // Copiamos o resultado parcial para o vetor global
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

        // Esperamos todas as threads terminarem
        for handle in handles {
            handle.join().unwrap();
        }

        // Agora podemos recuperar os vértices calculados para dentro de self.vertices
        {
            let final_vertices = arc_vertices.lock().unwrap();
            self.vertices.copy_from_slice(&final_vertices);
        }

        // Por fim, geramos as faces (pode ser paralelizado também, mas aqui está em uma thread só)
        self.faces.clear();
        for i in 0..resi - 1 {
            for j in 0..resj - 1 {
                self.faces.push([
                    i * resj + j,
                    i * resj + (j + 1),
                    (i + 1) * resj + (j + 1),
                    (i + 1) * resj + j,
                ]);
            }
        }
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

    /// Retorna slice imutável para vértices da malha
    pub fn get_vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// Retorna slice imutável para as faces interpolados
    pub fn get_faces(&self) -> &[[usize; 4]] {
        &self.faces
    }

    pub fn translate(&mut self, dx: f32, dy: f32, dz: f32) {
        let translation_matrix: Mat4 = Mat4::new(
            1.0, 0.0, 0.0, dx,
            0.0, 1.0, 0.0, dy,
            0.0, 0.0, 1.0, dz,
            0.0, 0.0, 0.0, 1.0,
        );

        for p in &mut self.control_points {
            let v = translation_matrix * p.to_homogeneous();
            *p = v.xyz();
        }

        for v in &mut self.vertices {
            let vt = translation_matrix * v.to_homogeneous();
            *v = vt.xyz();
        }
    }

    pub fn scale(&mut self, scale: f32) {
        let scale_matrix: Mat4 = Mat4::new(
            scale, 0.0, 0.0, 0.0,
            0.0, scale, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        for p in &mut self.control_points {
            let v = scale_matrix * p.to_homogeneous();
            *p = v.xyz();
        }

        for v in &mut self.vertices {
            let vt = scale_matrix * v.to_homogeneous();
            *v = vt.xyz();
        }
    }

    pub fn rotate_x(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, cos_theta, -sin_theta, 0.0,
            0.0, sin_theta, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        for p in &mut self.control_points {
            let v = rotation_matrix * p.to_homogeneous();
            *p = v.xyz();
        }

        for v in &mut self.vertices {
            let vt = rotation_matrix * v.to_homogeneous();
            *v = vt.xyz();
        }
    }

    pub fn rotate_y(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            cos_theta, 0.0, sin_theta, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -sin_theta, 0.0, cos_theta, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        for p in &mut self.control_points {
            let v = rotation_matrix * p.to_homogeneous();
            *p = v.xyz();
        }

        for v in &mut self.vertices {
            let vt = rotation_matrix * v.to_homogeneous();
            *v = vt.xyz();
        }
    }

    pub fn rotate_z(&mut self, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        let rotation_matrix: Mat4 = Mat4::new(
            cos_theta, -sin_theta, 0.0, 0.0,
            sin_theta, cos_theta, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        for p in &mut self.control_points {
            let v = rotation_matrix * p.to_homogeneous();
            *p = v.xyz();
        }

        for v in &mut self.vertices {
            let vt = rotation_matrix * v.to_homogeneous();
            *v = vt.xyz();
        }
    }
}