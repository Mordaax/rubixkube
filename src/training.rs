//! Deterministic CPU linear regression; scores are measured on held-out data.
use crate::{invalid, Result};
use serde::{Deserialize, Serialize};

/// A held-out validation measurement after one training epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation { pub epoch: usize, pub val_loss: f64 }
/// Fitted coefficients, measured loss, and the complete validation curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingResult {
    pub learning_rate: f64,
    pub epochs: usize,
    pub weight: f64,
    pub bias: f64,
    pub val_loss: f64,
    pub history: Vec<Observation>,
    pub execution: String,
}

/// Fit the original demo's fixed dataset using full-batch gradient descent.
///
/// ```
/// let fitted = optimizationjob::training::train(0.1, 40)?;
/// assert!(fitted.val_loss < 0.02);
/// # Ok::<(), optimizationjob::Error>(())
/// ```
pub fn train(learning_rate: f64, epochs: usize) -> Result<TrainingResult> {
    if !learning_rate.is_finite() || learning_rate <= 0. || learning_rate > 0.2 || epochs == 0 {
        return Err(invalid("Use a finite learning rate in (0, 0.2] and positive epochs"));
    }
    let examples: Vec<[f64; 2]> = serde_json::from_str(include_str!("../demo/fixtures/dataset.json"))?;
    let (training, validation) = examples.split_at(64);
    let (mut weight, mut bias) = (0., 0.);
    let mut history = Vec::with_capacity(epochs);
    for epoch in 1..=epochs {
        let errors: Vec<(f64, f64)> = training.iter().map(|[x, y]| (weight * x + bias - y, *x)).collect();
        weight -= learning_rate * 2. * errors.iter().map(|(e,x)| e*x).sum::<f64>() / training.len() as f64;
        bias -= learning_rate * 2. * errors.iter().map(|(e,_)| e).sum::<f64>() / training.len() as f64;
        let val_loss = validation.iter().map(|[x,y]| (weight*x+bias-y).powi(2)).sum::<f64>() / validation.len() as f64;
        history.push(Observation { epoch, val_loss });
    }
    let val_loss = history.last().ok_or_else(|| invalid("No training epochs"))?.val_loss;
    Ok(TrainingResult { learning_rate, epochs, weight, bias, val_loss, history,
        execution: "CPU, Rust; no GPU or simulated score".into() })
}
