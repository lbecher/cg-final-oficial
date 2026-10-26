pub const GUI_VIEWPORT_WIDTH: f32 = 800.0;
pub const GUI_VIEWPORT_HEIGHT: f32 = 600.0;
pub const GUI_VIEWPORT_PADDING: f32 = 8.0;
pub const GUI_SIDEBAR_WIDTH: f32 = 300.0;

pub const MAX_FRAME_TIME: f32 = 0.05;

pub const GROUND_Y: f32 = -3.0;
pub const GROUND_HALF_WIDTH: f32 = 9.0;
pub const GROUND_THICKNESS: f32 = 4.0;
pub const GROUND_DEPTH: f32 = 6.0;
pub const BACKGROUND_Z: f32 = -2.5;
pub const CLOUD_Z: f32 = -3.0;
pub const BACKGROUND_CACTI: [(f32, f32); 5] = [(-6.5, 0.8), (-1.5, 1.3), (2.5, 0.6), (4.2, 1.0), (7.0, 1.5)];
pub const CLOUDS: [(f32, f32, f32, f32); 5] = [
    (-5.0, 4.2, 1.2, 0.4),
    (-4.0, 4.45, 0.8, 0.35),
    (1.0, 3.6, 1.4, 0.45),
    (2.1, 3.85, 0.9, 0.4),
    (5.8, 4.8, 1.1, 0.35),
];
pub const SKY_DAY: [f32; 3] = [160.0, 205.0, 245.0];
pub const SKY_NIGHT: [f32; 3] = [45.0, 55.0, 95.0];

pub const DINO_X: f32 = -5.0;
pub const DINO_SIZE: f32 = 1.0;
pub const JUMP_SPEED: f32 = 12.0;
pub const GRAVITY: f32 = 36.0;

pub const SPHERE_RADIUS: f32 = 0.5;
pub const SPHERE_SLICES: usize = 8;
pub const SPHERE_STACKS: usize = 6;
pub const SPHERE_SCALE_MIN: f32 = 0.8;
pub const SPHERE_SCALE_MAX: f32 = 1.15;

pub const BOUNCE_HEIGHT_MAX: f32 = 0.35;
pub const BOUNCE_SPEED_MIN: f32 = 5.0;
pub const BOUNCE_SPEED_MAX: f32 = 8.0;

pub const SPAWN_X: f32 = 10.0;
pub const DESPAWN_X: f32 = -10.0;
pub const FIRST_SPAWN_DELAY: f32 = 1.0;
pub const SPAWN_INTERVAL_MIN: f32 = 1.1;
pub const SPAWN_INTERVAL_MAX: f32 = 2.4;
pub const COLLISION_TOLERANCE: f32 = 0.85;

pub const INITIAL_SPEED: f32 = 7.0;
pub const MAX_SPEED: f32 = 13.0;
pub const SPEED_GAIN: f32 = 0.1;
pub const SCORE_RATE: f32 = 10.0;
pub const RESTART_DELAY: f32 = 0.4;

pub const SUN_RADIUS: f32 = 0.6;
pub const SUN_PERIOD: f32 = 60.0;
pub const SUN_START_PROGRESS: f32 = 0.3;
pub const SUN_X_START: f32 = 9.0;
pub const SUN_X_END: f32 = -9.0;
pub const SUN_Y_MIN: f32 = 1.5;
pub const SUN_Y_MAX: f32 = 5.5;
pub const SUN_Z: f32 = 6.0;

pub const MIN_DAYLIGHT: f32 = 0.3;
