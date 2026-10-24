//! Generic result verification for arbitrary metrics, parameters and workloads.
use crate::{
    invalid,
    report::{condition, string},
    Result,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
pub fn summarize(job: &Value, trials: &[Value]) -> Result<Value> {
    if !condition(job, "Complete") || condition(job, "Failed") {
        return Err(invalid("OptimizationJob has not completed successfully"));
    }
    let budget = job["spec"]["numTrials"]
        .as_u64()
        .ok_or_else(|| invalid("Missing trial budget"))?;
    if budget == 0 || budget as usize != trials.len() {
        return Err(invalid("Trial count differs from requested budget"));
    }
    let objective = &job["spec"]["objectives"][0];
    let metric = string(objective, "/metric")?;
    let sign = match string(objective, "/direction")? {
        "Minimize" => 1.,
        "Maximize" => -1.,
        _ => return Err(invalid("Invalid objective direction")),
    };
    let uid = string(job, "/metadata/uid")?;
    let namespace = string(job, "/metadata/namespace")?;
    let mut names = HashSet::new();
    let mut rows = Vec::new();
    for trial in trials {
        let name = string(trial, "/metadata/name")?;
        if !names.insert(name) || !condition(trial, "Complete") || condition(trial, "Failed") {
            return Err(invalid("Duplicate, failed or incomplete trial"));
        }
        if string(trial, "/metadata/namespace")? != namespace
            || !trial["metadata"]["ownerReferences"]
                .as_array()
                .is_some_and(|owners| {
                    owners
                        .iter()
                        .any(|o| o["kind"] == "OptimizationJob" && o["uid"] == uid)
                })
        {
            return Err(invalid(
                "Trial does not belong to this OptimizationJob UID and namespace",
            ));
        }
        let created = string(trial, "/metadata/creationTimestamp")?;
        let time = chrono::DateTime::parse_from_rfc3339(created)
            .map_err(|_| invalid("Invalid trial creation timestamp"))?;
        let observation = trial
            .pointer("/status/trainerStatus/metrics")
            .and_then(Value::as_array)
            .and_then(|m| m.iter().rev().find(|m| m["name"] == metric))
            .ok_or_else(|| invalid(format!("Trial {name} is missing objective {metric}")))?;
        let value: f64 = string(observation, "/value")?
            .parse()
            .map_err(|_| invalid("Objective is not numeric"))?;
        if !value.is_finite() {
            return Err(invalid("Objective must be finite"));
        }
        let mut parameters = BTreeMap::new();
        if let Some(env) = trial.pointer("/spec/trainer/env").and_then(Value::as_array) {
            for entry in env {
                if let Some(name) = entry["name"]
                    .as_str()
                    .and_then(|s| s.strip_prefix("KUBEFLOW_TRAINER_OPT_"))
                {
                    parameters.insert(name.to_owned(), string(entry, "/value")?.to_owned());
                }
            }
        }
        rows.push((sign*value,time,name.to_owned(),json!({"name":name,"created":created,"value":value,"parameters":parameters,"status":"Complete"})));
    }
    rows.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("Validated finite objective")
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    if string(job, "/status/result/trainJobName")? != rows[0].2 {
        return Err(invalid(
            "Controller winner differs from independently ranked objective",
        ));
    }
    Ok(
        json!({"schema_version":2,"job":string(job,"/metadata/name")?,"namespace":namespace,"job_uid":uid,"objective":objective,"best":rows[0].3,"trials":rows.iter().map(|r|&r.3).collect::<Vec<_>>(),"winner_verified":true}),
    )
}
