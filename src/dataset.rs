//! CSV input for the optional built-in linear training workload.
use crate::{invalid, Result};
use std::{fs, path::Path};
pub fn embedded() -> Result<Vec<[f64; 2]>> {
    Ok(serde_json::from_str(include_str!(
        "../demo/fixtures/dataset.json"
    ))?)
}
pub fn validate(rows: &[[f64; 2]]) -> Result<()> {
    if rows.len() < 4 || rows.len() > 1_000_000 || rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("Dataset needs 4..1000000 finite x,y rows"));
    }
    Ok(())
}
pub fn read_csv(path: &Path) -> Result<Vec<[f64; 2]>> {
    if fs::metadata(path)?.len() > 32 * 1024 * 1024 {
        return Err(invalid("CSV exceeds 32 MiB"));
    }
    let text = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (index, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if index == 0 && line.split(',').map(str::trim).collect::<Vec<_>>() == ["x", "y"] {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != 2 {
            return Err(invalid(format!("CSV line {} needs exactly x,y", index + 1)));
        }
        let x = fields[0]
            .parse::<f64>()
            .map_err(|_| invalid(format!("Invalid x on CSV line {}", index + 1)))?;
        let y = fields[1]
            .parse::<f64>()
            .map_err(|_| invalid(format!("Invalid y on CSV line {}", index + 1)))?;
        rows.push([x, y]);
    }
    validate(&rows)?;
    Ok(rows)
}
