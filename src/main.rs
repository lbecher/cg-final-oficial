mod app;
mod constants;
mod object;
mod render;
mod types;

use constants::*;
use eframe::{NativeOptions, Result, run_native};
use eframe::egui::{Vec2, ViewportBuilder};
use app::App;

fn main() -> Result {
    let title = "Aleluia";

    let width = GUI_VIEWPORT_WIDTH + GUI_SIDEBAR_WIDTH + GUI_VIEWPORT_PADDING * 2.0;
    let height = GUI_VIEWPORT_HEIGHT + GUI_VIEWPORT_PADDING * 2.0;

    let options = NativeOptions {
        viewport: ViewportBuilder {
            inner_size: Some(Vec2::new(width, height)),
            resizable: Some(false),
            maximize_button: Some(false),
            minimize_button: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };

    run_native(
        title,
        options,
        Box::new(|_cc| Ok(Box::<App>::default())),
    )
}