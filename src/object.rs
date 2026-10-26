use std::f32::consts::PI;
use crate::types::*;

#[derive(Debug, Clone)]
pub struct Face {
    pub vertices: Vec<usize>,
    pub visible: bool,
    pub normal: Vec3,
    pub direction: Vec3,
    pub centroid: Vec3,
}

impl Face {
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.push(vertices[0]);
        Self {
            vertices,
            visible: false,
            normal: Vec3::zeros(),
            direction: Vec3::zeros(),
            centroid: Vec3::zeros(),
        }
    }

    /// Calcula a normal da face.
    pub fn calc_normal(&mut self, vertices: &Vec<Vec3>) {
        let a: Vec3 = vertices[self.vertices[0]];
        let b: Vec3 = vertices[self.vertices[1]];
        let c: Vec3 = vertices[self.vertices[2]];

        let bc: Vec3 = c - b;
        let ba: Vec3 = a - b;

        self.normal = bc.cross(&ba).normalize();
    }

    /// Calcula o centroide da face.
    pub fn calc_centroid(&mut self, vertices: &Vec<Vec3>) {
        // O último índice repete o primeiro, então é ignorado.
        let n = self.vertices.len() - 1;
        let mut centroid = Vec3::zeros();
        for i in 0..n {
            centroid = centroid + vertices[self.vertices[i]];
        }
        self.centroid = centroid / n as f32;
    }
}

/// Coeficientes do material usados no modelo de iluminação.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub ka: Vec3,
    pub kd: Vec3,
    pub ks: Vec3,
    pub n: f32,
}

impl Material {
    pub fn new(ka: [f32; 3], kd: [f32; 3], ks: [f32; 3], n: f32) -> Self {
        Self {
            ka: Vec3::from(ka),
            kd: Vec3::from(kd),
            ks: Vec3::from(ks),
            n,
        }
    }
}

/// Estrutura para armazenar uma malha poligonal.
#[derive(Debug, Clone)]
pub struct Object {
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
}

impl Object {
    pub fn new(
        vertices: Vec<Vec3>,
        faces: Vec<Vec<usize>>,
        material: &Material,
    ) -> Self {
        let faces: Vec<Face> = faces.into_iter().map(Face::new).collect();

        let mut obj = Self {
            vertices,
            faces,
            centroid: Vec3::zeros(),
            ka: material.ka,
            kd: material.kd,
            ks: material.ks,
            n: material.n,
        };

        obj.calc_centroid();

        obj
    }

    //--------------------------------------------------------------------------------
    // Primitivas geométricas
    //--------------------------------------------------------------------------------

    /// Gera um paralelepípedo alinhado aos eixos entre os cantos `min` e `max`.
    /// Cada face tem seus próprios vértices, para que as arestas continuem vivas
    /// mesmo nos sombreamentos que interpolam pelos vértices (Gouraud e Phong).
    pub fn cuboid(min: Vec3, max: Vec3, material: &Material) -> Self {
        let (x0, y0, z0) = (min.x, min.y, min.z);
        let (x1, y1, z1) = (max.x, max.y, max.z);

        let quads: [[Vec3; 4]; 6] = [
            // Frente (+z)
            [Vec3::new(x0, y0, z1), Vec3::new(x1, y0, z1), Vec3::new(x1, y1, z1), Vec3::new(x0, y1, z1)],
            // Trás (-z)
            [Vec3::new(x1, y0, z0), Vec3::new(x0, y0, z0), Vec3::new(x0, y1, z0), Vec3::new(x1, y1, z0)],
            // Direita (+x)
            [Vec3::new(x1, y0, z1), Vec3::new(x1, y0, z0), Vec3::new(x1, y1, z0), Vec3::new(x1, y1, z1)],
            // Esquerda (-x)
            [Vec3::new(x0, y0, z0), Vec3::new(x0, y0, z1), Vec3::new(x0, y1, z1), Vec3::new(x0, y1, z0)],
            // Topo (+y)
            [Vec3::new(x0, y1, z1), Vec3::new(x1, y1, z1), Vec3::new(x1, y1, z0), Vec3::new(x0, y1, z0)],
            // Base (-y)
            [Vec3::new(x0, y0, z0), Vec3::new(x1, y0, z0), Vec3::new(x1, y0, z1), Vec3::new(x0, y0, z1)],
        ];

        let mut vertices: Vec<Vec3> = Vec::with_capacity(24);
        let mut faces: Vec<Vec<usize>> = Vec::with_capacity(6);
        for quad in quads.iter() {
            let base = vertices.len();
            vertices.extend_from_slice(quad);
            faces.push(vec![base, base + 1, base + 2, base + 3]);
        }

        Self::new(vertices, faces, material)
    }

    /// Gera uma esfera centrada na origem por aproximação planar, com `slices`
    /// divisões em longitude e `stacks` divisões em latitude.
    pub fn sphere(radius: f32, slices: usize, stacks: usize, material: &Material) -> Self {
        let mut vertices: Vec<Vec3> = Vec::with_capacity(2 + (stacks - 1) * slices);

        // Polo norte, anéis intermediários e polo sul.
        vertices.push(Vec3::new(0.0, radius, 0.0));
        for i in 1..stacks {
            let theta = PI * i as f32 / stacks as f32;
            for j in 0..slices {
                let phi = 2.0 * PI * j as f32 / slices as f32;
                vertices.push(Vec3::new(
                    radius * theta.sin() * phi.cos(),
                    radius * theta.cos(),
                    radius * theta.sin() * phi.sin(),
                ));
            }
        }
        vertices.push(Vec3::new(0.0, -radius, 0.0));

        let north = 0;
        let south = vertices.len() - 1;
        let ring = |i: usize, j: usize| 1 + (i - 1) * slices + (j % slices);

        let mut faces: Vec<Vec<usize>> = Vec::with_capacity(slices * stacks);
        for j in 0..slices {
            faces.push(vec![north, ring(1, j + 1), ring(1, j)]);
        }
        for i in 1..stacks - 1 {
            for j in 0..slices {
                faces.push(vec![ring(i, j), ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j)]);
            }
        }
        for j in 0..slices {
            faces.push(vec![south, ring(stacks - 1, j), ring(stacks - 1, j + 1)]);
        }

        Self::new(vertices, faces, material)
    }

    //--------------------------------------------------------------------------------
    // Transformações
    //--------------------------------------------------------------------------------

    /// Retorna uma cópia do objeto com todos os vértices multiplicados pela
    /// matriz de transformação (coordenadas homogêneas).
    pub fn transformed(&self, matrix: &Mat4) -> Self {
        let mut obj = self.clone();
        for vertex in &mut obj.vertices {
            let vt: Mat4x1 = matrix * vec3_to_mat4x1(vertex);
            *vertex = vt.xyz();
        }
        obj.calc_centroid();
        obj
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
}
