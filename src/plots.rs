//! Plain, standalone scientific plots of saved trial measurements.
use crate::{invalid, report::string, training::Observation, Result};
use plotters::{coord::Shift, prelude::*};
use serde::Deserialize;
use serde_json::Value;
use std::{fs, path::Path, sync::OnceLock};

#[derive(Deserialize)]
struct Trial {
    name: String,
    learning_rate: f64,
    val_loss: f64,
    history: Vec<Observation>,
}
fn plot_error(error: impl std::fmt::Debug) -> crate::Error {
    invalid(format!("Plot rendering failed: {error:?}"))
}
fn font() -> Result<()> {
    static FONT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    FONT.get_or_init(|| {
        plotters::style::register_font(
            "sans-serif",
            FontStyle::Normal,
            include_bytes!("../assets/DejaVuSans.ttf"),
        )
        .map_err(|_| "Bundled plot font is invalid".to_owned())
    })
    .clone()
    .map_err(invalid)
}
fn color(index: usize) -> RGBColor {
    [
        RGBColor(40, 90, 150),
        RGBColor(180, 65, 45),
        RGBColor(45, 120, 75),
        RGBColor(115, 80, 140),
    ][index % 4]
}
fn id(trial: &Trial) -> &str {
    trial.name.rsplit('-').next().unwrap_or(&trial.name)
}
fn curves<DB: DrawingBackend>(area: DrawingArea<DB, Shift>, trials: &[Trial]) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.fill(&WHITE).map_err(plot_error)?;
    let xmax = trials
        .iter()
        .flat_map(|t| &t.history)
        .map(|h| h.epoch)
        .max()
        .unwrap_or(1)
        .max(2);
    let ymax = trials
        .iter()
        .flat_map(|t| &t.history)
        .map(|h| h.val_loss)
        .fold(0.0, f64::max)
        .max(0.01)
        * 1.1;
    let mut chart = ChartBuilder::on(&area)
        .caption("Validation loss by epoch", ("sans-serif", 25))
        .margin(25)
        .x_label_area_size(52)
        .y_label_area_size(75)
        .build_cartesian_2d(1usize..xmax, 0f64..ymax)
        .map_err(plot_error)?;
    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Validation MSE")
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 14))
        .light_line_style(RGBColor(235, 235, 235))
        .draw()
        .map_err(plot_error)?;
    for (index, trial) in trials.iter().enumerate() {
        let ink = color(index);
        chart
            .draw_series(LineSeries::new(
                trial.history.iter().map(|h| (h.epoch, h.val_loss)),
                ink.stroke_width(2),
            ))
            .map_err(plot_error)?
            .label(format!("{}  lr={}", id(trial), trial.learning_rate))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ink.stroke_width(2)));
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9))
        .border_style(RGBColor(180, 180, 180))
        .label_font(("sans-serif", 14))
        .draw()
        .map_err(plot_error)?;
    area.present().map_err(plot_error)
}
fn results<DB: DrawingBackend>(area: DrawingArea<DB, Shift>, trials: &[Trial]) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.fill(&WHITE).map_err(plot_error)?;
    let ymax = trials
        .iter()
        .map(|t| t.val_loss)
        .fold(0., f64::max)
        .max(0.01)
        * 1.2;
    let mut chart = ChartBuilder::on(&area)
        .caption("Final validation loss by trial", ("sans-serif", 25))
        .margin(25)
        .x_label_area_size(65)
        .y_label_area_size(75)
        .build_cartesian_2d(0.5f64..trials.len() as f64 + 0.5, 0f64..ymax)
        .map_err(plot_error)?;
    let label = |x: &f64| {
        let i = x.round() as isize - 1;
        if i >= 0 && (i as usize) < trials.len() && (x - x.round()).abs() < 0.01 {
            id(&trials[i as usize]).to_owned()
        } else {
            String::new()
        }
    };
    chart
        .configure_mesh()
        .x_desc("Trial ID")
        .y_desc("Validation MSE")
        .x_labels(trials.len())
        .x_label_formatter(&label)
        .axis_desc_style(("sans-serif", 18))
        .label_style(("sans-serif", 14))
        .disable_x_mesh()
        .light_line_style(RGBColor(235, 235, 235))
        .draw()
        .map_err(plot_error)?;
    for (index, trial) in trials.iter().enumerate() {
        let x = index as f64 + 1.;
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.), (x, trial.val_loss)],
                color(index).stroke_width(2),
            )))
            .map_err(plot_error)?;
        chart
            .draw_series(std::iter::once(Circle::new(
                (x, trial.val_loss),
                5,
                color(index).filled(),
            )))
            .map_err(plot_error)?;
        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.6}", trial.val_loss),
                (x, trial.val_loss + ymax * 0.045),
                ("sans-serif", 14),
            )))
            .map_err(plot_error)?;
    }
    area.present().map_err(plot_error)
}
fn csv(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}
/// Export two PNG/SVG plots, numeric CSV, and a short text summary.
/// Reads recorded observations only; never reruns or simulates training.
pub fn write_plots(report: &Value, output: &Path) -> Result<()> {
    if report["winner_verified"] != true {
        return Err(invalid(
            "Only independently verified results can be plotted",
        ));
    }
    let trials: Vec<Trial> = serde_json::from_value(report["trials"].clone())?;
    if trials.is_empty() {
        return Err(invalid("No recorded trials"));
    }
    for trial in &trials {
        if !trial.learning_rate.is_finite()
            || !trial.val_loss.is_finite()
            || trial.val_loss < 0.
            || trial.history.is_empty()
            || trial
                .history
                .iter()
                .enumerate()
                .any(|(i, h)| h.epoch != i + 1 || !h.val_loss.is_finite() || h.val_loss < 0.)
            || trial
                .history
                .last()
                .is_none_or(|h| (h.val_loss - trial.val_loss).abs() > 1e-12)
        {
            return Err(invalid("Invalid or inconsistent recorded measurements"));
        }
    }
    let job = string(report, "/job")?;
    let winner = string(report, "/best/name")?;
    let best = trials
        .iter()
        .find(|trial| trial.name == winner)
        .ok_or_else(|| invalid("Winner is absent from recorded trials"))?;
    font()?;
    fs::create_dir_all(output)?;
    for name in [
        "loss-by-epoch.svg",
        "loss-by-epoch.png",
        "trial-results.svg",
        "trial-results.png",
        "results.csv",
        "summary.txt",
    ] {
        if fs::symlink_metadata(output.join(name)).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(invalid("Refusing to overwrite a plot-output symlink"));
        }
    }
    let path = output.join("loss-by-epoch.svg");
    curves(
        SVGBackend::new(&path, (1000, 620)).into_drawing_area(),
        &trials,
    )?;
    let path = output.join("trial-results.svg");
    results(
        SVGBackend::new(&path, (1000, 620)).into_drawing_area(),
        &trials,
    )?;
    let path = output.join("loss-by-epoch.png");
    curves(
        BitMapBackend::new(&path, (1000, 620)).into_drawing_area(),
        &trials,
    )?;
    let path = output.join("trial-results.png");
    results(
        BitMapBackend::new(&path, (1000, 620)).into_drawing_area(),
        &trials,
    )?;
    let raw = report["trials"]
        .as_array()
        .ok_or_else(|| invalid("Missing trials"))?;
    let mut table=String::from("trial,learning_rate,val_loss,weight,bias,gpu_device,cuda_elapsed_ms,cpu_reference_max_abs_error,metric_delivered\n");
    for (trial, row) in trials.iter().zip(raw) {
        let device = row
            .pointer("/gpu/device_name")
            .and_then(Value::as_str)
            .unwrap_or("CPU");
        let number = |pointer: &str| {
            row.pointer(pointer)
                .and_then(Value::as_f64)
                .map(|n| n.to_string())
                .unwrap_or_default()
        };
        table.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            csv(&trial.name),
            trial.learning_rate,
            trial.val_loss,
            number("/weight"),
            number("/bias"),
            csv(device),
            number("/gpu/cuda_elapsed_ms"),
            number("/gpu/cpu_reference_max_abs_error"),
            row["metric_delivered"].as_bool().unwrap_or(false)
        ));
    }
    fs::write(output.join("results.csv"), table)?;
    let mode = if report.pointer("/coverage/gpu").and_then(Value::as_bool) == Some(true) {
        "CUDA GPU"
    } else {
        "CPU"
    };
    fs::write(output.join("summary.txt"),format!("Job: {job}\nExecution: {mode}\nTrials: {}\nWinner: {winner}\nLearning rate: {}\nValidation MSE: {}\n\nMeasured observations: results.csv\nValidation curves: loss-by-epoch.png / .svg\nFinal trial losses: trial-results.png / .svg\n",trials.len(),best.learning_rate,best.val_loss))?;
    Ok(())
}
