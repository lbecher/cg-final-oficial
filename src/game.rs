use std::f32::consts::PI;
use rand::RngExt;

use crate::constants::*;
use crate::object::{Material, Object};
use crate::transform::*;
use crate::types::*;

#[derive(Clone, Copy, PartialEq)]
pub enum GameState {
    Ready,
    Running,
    GameOver,
}

struct Obstacle {
    x: f32,
    scale: f32,
    angle: f32,
    bounce_phase: f32,
    bounce_speed: f32,
    bounce_height: f32,
    passed: bool,
}

impl Obstacle {
    fn radius(&self) -> f32 {
        SPHERE_RADIUS * self.scale
    }

    fn center(&self) -> Vec3 {
        let bounce = self.bounce_height * self.bounce_phase.sin().abs();
        Vec3::new(self.x, GROUND_Y + self.radius() + bounce, 0.0)
    }
}

pub struct Game {
    pub state: GameState,
    pub score: f32,
    pub best_score: u32,
    pub jumped_obstacles: u32,
    pub speed: f32,

    elapsed: f32,
    jump_time: Option<f32>,
    obstacles: Vec<Obstacle>,
    spawn_timer: f32,
    game_over_timer: f32,
    sun_progress: f32,
    dino_mesh: Object,
    sphere_mesh: Object,
    sun_mesh: Object,
    scenery: Vec<Object>,
}

impl Game {
    pub fn new() -> Self {
        let dino_material = Material::new([0.45, 0.45, 0.45], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], 8.0);
        let sphere_material = Material::new([0.2, 0.5, 0.25], [0.25, 0.65, 0.3], [0.5, 0.5, 0.5], 12.0);
        let sun_material = Material::new([3.6, 3.0, 0.4], [0.0; 3], [0.0; 3], 1.0);

        let half = DINO_SIZE / 2.0;

        Self {
            state: GameState::Ready,
            score: 0.0,
            best_score: 0,
            jumped_obstacles: 0,
            speed: INITIAL_SPEED,

            elapsed: 0.0,
            jump_time: None,
            obstacles: Vec::new(),
            spawn_timer: FIRST_SPAWN_DELAY,
            game_over_timer: 0.0,
            sun_progress: SUN_START_PROGRESS,

            dino_mesh: Object::cuboid(Vec3::from_element(-half), Vec3::from_element(half), &dino_material),
            sphere_mesh: Object::sphere(SPHERE_RADIUS, SPHERE_SLICES, SPHERE_STACKS, &sphere_material),
            sun_mesh: Object::sphere(SUN_RADIUS, 10, 8, &sun_material),
            scenery: Self::build_scenery(),
        }
    }

    fn build_scenery() -> Vec<Object> {
        let ground_material = Material::new([0.75, 0.65, 0.45], [0.8, 0.7, 0.5], [0.05, 0.05, 0.05], 2.0);
        let cactus_material = Material::new([0.2, 0.45, 0.2], [0.3, 0.6, 0.3], [0.1, 0.1, 0.1], 4.0);
        let cloud_material = Material::new([1.0, 1.0, 1.0], [0.6, 0.6, 0.6], [0.0; 3], 1.0);

        let mut scenery = Vec::new();

        scenery.push(Object::cuboid(
            Vec3::new(-GROUND_HALF_WIDTH, GROUND_Y - GROUND_THICKNESS, -GROUND_DEPTH / 2.0),
            Vec3::new(GROUND_HALF_WIDTH, GROUND_Y, GROUND_DEPTH / 2.0),
            &ground_material,
        ));

        let cactus = [
            Object::cuboid(Vec3::new(-0.2, 0.0, -0.2), Vec3::new(0.2, 1.6, 0.2), &cactus_material),
            Object::cuboid(Vec3::new(-0.55, 0.6, -0.12), Vec3::new(-0.2, 0.8, 0.12), &cactus_material),
            Object::cuboid(Vec3::new(-0.55, 0.8, -0.12), Vec3::new(-0.35, 1.2, 0.12), &cactus_material),
            Object::cuboid(Vec3::new(0.2, 0.85, -0.12), Vec3::new(0.5, 1.05, 0.12), &cactus_material),
            Object::cuboid(Vec3::new(0.3, 1.05, -0.12), Vec3::new(0.5, 1.35, 0.12), &cactus_material),
        ];
        for (x, s) in BACKGROUND_CACTI {
            let matrix = translation(&Vec3::new(x, GROUND_Y, BACKGROUND_Z)) * scale(&Vec3::from_element(s));
            for part in &cactus {
                scenery.push(part.transformed(&matrix));
            }
        }

        let cloud = Object::sphere(1.0, 8, 6, &cloud_material);
        for (x, y, sx, sy) in CLOUDS {
            let matrix = translation(&Vec3::new(x, y, CLOUD_Z)) * scale(&Vec3::new(sx, sy, 0.4));
            scenery.push(cloud.transformed(&matrix));
        }

        scenery
    }

    pub fn action(&mut self) {
        match self.state {
            GameState::Ready => self.start(),
            GameState::Running => {
                if self.jump_time.is_none() {
                    self.jump_time = Some(0.0);
                }
            }
            GameState::GameOver => {
                if self.game_over_timer >= RESTART_DELAY {
                    self.start();
                }
            }
        }
    }

    fn start(&mut self) {
        self.state = GameState::Running;
        self.score = 0.0;
        self.jumped_obstacles = 0;
        self.speed = INITIAL_SPEED;
        self.elapsed = 0.0;
        self.jump_time = None;
        self.obstacles.clear();
        self.spawn_timer = FIRST_SPAWN_DELAY;
    }

    pub fn update(&mut self, dt: f32) {
        self.sun_progress = (self.sun_progress + dt / SUN_PERIOD) % 1.0;

        match self.state {
            GameState::Running => self.update_running(dt),
            GameState::GameOver => self.game_over_timer += dt,
            GameState::Ready => {}
        }
    }

    fn update_running(&mut self, dt: f32) {
        self.elapsed += dt;
        self.speed = (INITIAL_SPEED + SPEED_GAIN * self.elapsed).min(MAX_SPEED);
        self.score += SCORE_RATE * dt;

        if let Some(t) = self.jump_time {
            let t = t + dt;
            self.jump_time = if Self::jump_height(t) > 0.0 { Some(t) } else { None };
        }

        let dx = self.speed * dt;
        for obstacle in &mut self.obstacles {
            obstacle.x -= dx;
            obstacle.angle += dx / obstacle.radius();
            obstacle.bounce_phase += obstacle.bounce_speed * dt;

            if !obstacle.passed && obstacle.x + obstacle.radius() < DINO_X - DINO_SIZE / 2.0 {
                obstacle.passed = true;
                self.jumped_obstacles += 1;
            }
        }
        self.obstacles.retain(|o| o.x > DESPAWN_X);

        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_obstacle();
        }

        if self.obstacles.iter().any(|o| self.collides(o)) {
            self.state = GameState::GameOver;
            self.game_over_timer = 0.0;
            self.best_score = self.best_score.max(self.score as u32);
        }
    }

    fn spawn_obstacle(&mut self) {
        let mut rng = rand::rng();
        self.obstacles.push(Obstacle {
            x: SPAWN_X,
            scale: rng.random_range(SPHERE_SCALE_MIN..SPHERE_SCALE_MAX),
            angle: 0.0,
            bounce_phase: rng.random_range(0.0..PI),
            bounce_speed: rng.random_range(BOUNCE_SPEED_MIN..BOUNCE_SPEED_MAX),
            bounce_height: rng.random_range(0.0..BOUNCE_HEIGHT_MAX),
            passed: false,
        });
        self.spawn_timer = rng.random_range(SPAWN_INTERVAL_MIN..SPAWN_INTERVAL_MAX);
    }

    fn jump_height(t: f32) -> f32 {
        JUMP_SPEED * t - 0.5 * GRAVITY * t * t
    }

    fn dino_height(&self) -> f32 {
        self.jump_time.map_or(0.0, |t| Self::jump_height(t).max(0.0))
    }

    fn collides(&self, obstacle: &Obstacle) -> bool {
        let half = DINO_SIZE / 2.0;
        let min = Vec3::new(DINO_X - half, GROUND_Y + self.dino_height(), -half);
        let max = min + Vec3::from_element(DINO_SIZE);

        let c = obstacle.center();
        let closest = Vec3::new(
            c.x.clamp(min.x, max.x),
            c.y.clamp(min.y, max.y),
            c.z.clamp(min.z, max.z),
        );
        let r = obstacle.radius() * COLLISION_TOLERANCE;
        (c - closest).norm_squared() < r * r
    }

    pub fn sun_position(&self) -> Vec3 {
        let u = self.sun_progress;
        Vec3::new(
            SUN_X_START + (SUN_X_END - SUN_X_START) * u,
            SUN_Y_MIN + (SUN_Y_MAX - SUN_Y_MIN) * (PI * u).sin(),
            SUN_Z,
        )
    }

    pub fn daylight(&self) -> f32 {
        (PI * self.sun_progress).sin()
    }

    pub fn scene(&self) -> Vec<Object> {
        let mut objects = self.scenery.clone();

        objects.push(self.sun_mesh.transformed(&translation(&self.sun_position())));

        let dino_center = Vec3::new(DINO_X, GROUND_Y + DINO_SIZE / 2.0 + self.dino_height(), 0.0);
        objects.push(self.dino_mesh.transformed(&translation(&dino_center)));

        for obstacle in &self.obstacles {
            let matrix = translation(&obstacle.center())
                * rotation_z(obstacle.angle)
                * scale(&Vec3::from_element(obstacle.scale));
            objects.push(self.sphere_mesh.transformed(&matrix));
        }

        objects
    }
}
