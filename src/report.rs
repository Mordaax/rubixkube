//! Validate Kubernetes observations independently of the upstream winner selection.
use crate::{invalid, Result};
use serde_json::{json, Value};
use std::collections::HashSet;

/// Require a JSON string at a resource field.
pub fn string<'a>(value: &'a Value, pointer: &str) -> Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("Missing string at {pointer}")))
}
/// Check a Kubernetes status condition.
pub fn condition(value: &Value, kind: &str) -> bool {
    value
        .pointer("/status/conditions")
        .and_then(Value::as_array)
        .is_some_and(|conditions| {
            conditions
                .iter()
                .any(|c| c["type"] == kind && c["status"] == "True")
        })
}
fn completed(value: &Value) -> bool {
    condition(value, "Complete") && !condition(value, "Failed")
}

/// Reject incomplete observations and verify finite metrics, budget and winner.
pub fn summarize(job: &Value, trials: &[Value]) -> Result<Value> {
    if !completed(job) {
        return Err(invalid("OptimizationJob has not completed successfully"));
    }
    let budget = job
        .pointer("/spec/numTrials")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("Missing trial budget"))?;
    if budget == 0 || trials.len() as u64 != budget {
        return Err(invalid("Observed trial count differs from job budget"));
    }
    let objectives = job
        .pointer("/spec/objectives")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("Missing objectives"))?;
    if objectives.len() != 1 {
        return Err(invalid("Exactly one objective is supported"));
    }
    let objective = &objectives[0];
    let metric = string(objective, "/metric")?;
    let direction = match string(objective, "/direction")? {
        "Minimize" => 1.,
        "Maximize" => -1.,
        _ => return Err(invalid("Invalid objective direction")),
    };
    let mut names = HashSet::new();
    let mut rows = Vec::with_capacity(trials.len());
    for trial in trials {
        if !completed(trial) {
            return Err(invalid("A trial has not completed successfully"));
        }
        let name = string(trial, "/metadata/name")?;
        if !names.insert(name) {
            return Err(invalid("Duplicate trial name"));
        }
        let created = string(trial, "/metadata/creationTimestamp")?;
        let time = chrono::DateTime::parse_from_rfc3339(created)
            .map_err(|_| invalid("Invalid trial creation timestamp"))?;
        let metrics = trial
            .pointer("/status/trainerStatus/metrics")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("Missing trial metrics"))?;
        let observation = metrics
            .iter()
            .rev()
            .find(|m| m["name"] == metric)
            .ok_or_else(|| invalid("Missing objective observation"))?;
        let loss: f64 = string(observation, "/value")?
            .parse()
            .map_err(|_| invalid("Objective is not numeric"))?;
        let env = trial
            .pointer("/spec/trainer/env")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("Missing trial environment"))?;
        let rate_env = env
            .iter()
            .find(|e| e["name"] == "KUBEFLOW_TRAINER_OPT_learning_rate")
            .ok_or_else(|| invalid("Missing learning rate"))?;
        let rate: f64 = string(rate_env, "/value")?
            .parse()
            .map_err(|_| invalid("Invalid learning rate"))?;
        if !loss.is_finite() || !rate.is_finite() {
            return Err(invalid("Nonfinite trial observation"));
        }
        rows.push((direction*loss, time, name.to_owned(), json!({"name":name,"created":created,"learning_rate":rate,"val_loss":loss,"status":"Complete"})));
    }
    rows.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let best = &rows[0].3;
    if string(job, "/status/result/trainJobName")? != string(best, "/name")? {
        return Err(invalid(
            "Controller winner differs from measured objective ranking",
        ));
    }
    Ok(
        json!({"job":string(job,"/metadata/name")?,"best":best,"trials":rows.iter().map(|r| &r.3).collect::<Vec<_>>(),"objective":objective,"winner_verified":true}),
    )
}
