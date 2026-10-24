//! Deterministic CPU linear regression; scores are measured on held-out data.
use crate::{invalid, Result};
use serde::{Deserialize, Serialize};

/// A held-out validation measurement after one training epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub epoch: usize,
    pub val_loss: f64,
}
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuEvidence>,
}

/// Fit the original demo's fixed dataset using full-batch gradient descent.
///
/// ```
/// let fitted = rubixkube::training::train(0.1, 40)?;
/// assert!(fitted.val_loss < 0.02);
/// # Ok::<(), rubixkube::Error>(())
/// ```
pub fn train(learning_rate: f64, epochs: usize) -> Result<TrainingResult> {
    train_on(learning_rate, epochs, &crate::dataset::embedded()?)
}
/// Fit supplied finite x,y samples; first half trains, second half validates.
pub fn train_on(
    learning_rate: f64,
    epochs: usize,
    examples: &[[f64; 2]],
) -> Result<TrainingResult> {
    crate::dataset::validate(examples)?;
    if !learning_rate.is_finite()
        || learning_rate <= 0.
        || learning_rate > 0.2
        || epochs == 0
        || epochs > 10000
    {
        return Err(invalid(
            "Use a finite learning rate in (0, 0.2] and positive epochs",
        ));
    }
    let (training, validation) = examples.split_at(examples.len() / 2);
    let (mut weight, mut bias) = (0., 0.);
    let mut history = Vec::with_capacity(epochs);
    for epoch in 1..=epochs {
        let errors: Vec<(f64, f64)> = training
            .iter()
            .map(|[x, y]| (weight * x + bias - y, *x))
            .collect();
        weight -= learning_rate * 2. * errors.iter().map(|(e, x)| e * x).sum::<f64>()
            / training.len() as f64;
        bias -=
            learning_rate * 2. * errors.iter().map(|(e, _)| e).sum::<f64>() / training.len() as f64;
        let val_loss = validation
            .iter()
            .map(|[x, y]| (weight * x + bias - y).powi(2))
            .sum::<f64>()
            / validation.len() as f64;
        if !weight.is_finite() || !bias.is_finite() || !val_loss.is_finite() {
            return Err(invalid(
                "Training diverged; normalize data or reduce the learning rate",
            ));
        }
        history.push(Observation { epoch, val_loss });
    }
    let val_loss = history
        .last()
        .ok_or_else(|| invalid("No training epochs"))?
        .val_loss;
    Ok(TrainingResult {
        learning_rate,
        epochs,
        weight,
        bias,
        val_loss,
        history,
        execution: "CPU, Rust; no GPU or simulated score".into(),
        gpu: None,
    })
}

/// Evidence collected from a completed CUDA kernel, checked against CPU results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuEvidence {
    pub device_name: String,
    pub compute_capability: [i32; 2],
    pub kernel_launches: usize,
    pub cuda_elapsed_ms: f64,
    pub cpu_reference_max_abs_error: f64,
}
