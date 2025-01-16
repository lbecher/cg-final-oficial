mod app;
mod constants;
mod object;
mod render;
mod types;
mod utils;

use eframe::{NativeOptions, Result, run_native};
use app::App;

fn main() -> Result {
    let title = "Aleluia";

    let options = NativeOptions::default();

    run_native(
        title,
        options,
        Box::new(|_cc| Ok(Box::<App>::default())),
    )
}