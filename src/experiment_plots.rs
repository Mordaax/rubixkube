//! Static exports for arbitrary objective metrics, including negative scores.
use crate::{invalid, report::string, Result};
use plotters::{coord::Shift, prelude::*};
use serde_json::Value;
use std::path::Path;
fn error(e: impl std::fmt::Debug) -> crate::Error {
    invalid(format!("Plot rendering failed: {e:?}"))
}
fn draw<DB: DrawingBackend>(
    area: DrawingArea<DB, Shift>,
    values: &[(String, f64)],
    metric: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    area.fill(&WHITE).map_err(error)?;
    let scale = values.iter().map(|(_, v)| v.abs()).fold(1., f64::max);
    let minimum = values.iter().map(|(_, v)| v / scale).fold(0., f64::min);
    let maximum = values.iter().map(|(_, v)| v / scale).fold(0., f64::max);
    let span = (maximum - minimum).max(0.1);
    let mut chart = ChartBuilder::on(&area)
        .caption(format!("Final {metric} by trial"), ("sans-serif", 24))
        .margin(25)
        .x_label_area_size(60)
        .y_label_area_size(95)
        .build_cartesian_2d(
            0.5..values.len() as f64 + 0.5,
            minimum - span * 0.1..maximum + span * 0.2,
        )
        .map_err(error)?;
    let label = |x: &f64| {
        let i = x.round() as isize - 1;
        if i >= 0 && (i as usize) < values.len() && (x - x.round()).abs() < 0.01 {
            values[i as usize]
                .0
                .rsplit('-')
                .next()
                .unwrap_or("")
                .to_owned()
        } else {
            String::new()
        }
    };
    let y_label = |v: &f64| format!("{:.3e}", v * scale);
    chart
        .configure_mesh()
        .x_desc("Trial ID")
        .y_desc(metric)
        .x_labels(values.len().min(12))
        .x_label_formatter(&label)
        .y_label_formatter(&y_label)
        .disable_x_mesh()
        .light_line_style(RGBColor(245, 245, 245))
        .draw()
        .map_err(error)?;
    for (i, (_, value)) in values.iter().enumerate() {
        let x = i as f64 + 1.;
        chart
            .draw_series(std::iter::once(Circle::new(
                (x, value / scale),
                5,
                RGBColor(40, 90, 150).filled(),
            )))
            .map_err(error)?;
        if values.len() <= 12 {
            chart
                .draw_series(std::iter::once(Text::new(
                    format!("{value:.6}"),
                    (x, value / scale + span * 0.04),
                    ("sans-serif", 13),
                )))
                .map_err(error)?;
        }
    }
    area.present().map_err(error)
}
fn csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}
pub fn write(report: &Value, output: &Path) -> Result<()> {
    if report["winner_verified"] != true {
        return Err(invalid("Cannot export an unverified result"));
    }
    crate::plots::font()?;
    let rows = report["trials"]
        .as_array()
        .ok_or_else(|| invalid("Missing trials"))?;
    let metric = string(report, "/objective/metric")?;
    let mut values = Vec::new();
    let mut table = String::from("trial,objective,value,parameters,cuda_training_verified\n");
    for row in rows {
        let name = string(row, "/name")?;
        let value = row["value"]
            .as_f64()
            .ok_or_else(|| invalid("Missing objective value"))?;
        if !value.is_finite() {
            return Err(invalid("Nonfinite objective"));
        }
        values.push((name.to_owned(), value));
        table.push_str(&format!(
            "{},{},{},{},{}\n",
            csv(name),
            csv(metric),
            value,
            csv(&row["parameters"].to_string()),
            row["cuda_training_verified"].as_bool().unwrap_or(false)
        ));
    }
    if values.is_empty() {
        return Err(invalid("No trial results"));
    }
    crate::experiment::save(output, "results.csv", table)?;
    crate::experiment::save(output,"summary.txt",format!("Job: {}\nMetric: {} ({})\nTrials: {}\nWinner: {}\nValue: {}\nParameters: {}\nCUDA training verified: {}\n",string(report,"/job")?,metric,string(report,"/objective/direction")?,rows.len(),string(report,"/best/name")?,report["best"]["value"],report["best"]["parameters"],report["cuda_training_verified"].as_bool().unwrap_or(false)))?;
    for filename in ["objective.svg", "objective.png"] {
        let target = output.join(filename);
        if std::fs::symlink_metadata(&target).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(invalid("Refusing to overwrite plot symlink"));
        }
    }
    let path = output.join("objective.svg");
    draw(
        SVGBackend::new(&path, (1000, 620)).into_drawing_area(),
        &values,
        metric,
    )?;
    let path = output.join("objective.png");
    draw(
        BitMapBackend::new(&path, (1000, 620)).into_drawing_area(),
        &values,
        metric,
    )?;
    Ok(())
}
