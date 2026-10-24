//! General workload lifecycle. Watching is resumable and never resubmits a job.
use crate::{
    cluster::{self, Api},
    config::{self, ExperimentConfig},
    experiment_report, invalid,
    report::{condition, string},
    shared, Result,
};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::Path,
    thread,
    time::{Duration, Instant},
};
fn name_ok(name: &str) -> Result<()> {
    if !config::dns_label(name) || name.len() > 28 {
        Err(invalid(
            "Invalid experiment name (DNS label, max 28 characters)",
        ))
    } else {
        Ok(())
    }
}
pub fn save(output: &Path, name: &str, bytes: impl AsRef<[u8]>) -> Result<()> {
    fs::create_dir_all(output)?;
    let path = output.join(name);
    if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(invalid("Refusing to overwrite an artifact symlink"));
    }
    let mut file = tempfile::NamedTempFile::new_in(output)?;
    file.write_all(bytes.as_ref())?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| crate::Error::Io(e.error))?;
    Ok(())
}
pub fn save_json(output: &Path, name: &str, value: &Value) -> Result<()> {
    save(output, name, serde_json::to_vec_pretty(value)?)
}
pub fn owned_job(api: &impl Api, name: &str) -> Result<Value> {
    name_ok(name)?;
    let job = cluster::get(api, "optimizationjob", name)?;
    if job.is_null() {
        return Err(invalid(format!(
            "Experiment {name} not found in this explicit context/namespace"
        )));
    }
    cluster::require_owned(&job, name)?;
    Ok(job)
}
pub fn submit(
    api: &impl Api,
    config: &ExperimentConfig,
    name: &str,
    output: &Path,
) -> Result<Value> {
    let manifest = config::manifest(config, name)?;
    if !cluster::get(api, "optimizationjob", name)?.is_null() {
        return Err(invalid(
            "Experiment already exists; use watch/results to resume, or choose another name",
        ));
    }
    save_json(output, "config.json", &serde_json::to_value(config)?)?;
    save_json(output, "manifest.json", &manifest)?;
    let resources = shared::prepare(api, config, name)?;
    if !resources.is_empty() {
        save_json(output, "shared-resources.json", &json!(resources))?;
    }
    let created = cluster::create(api, &manifest)?;
    save_json(output, "submitted-job.json", &created)?;
    Ok(created)
}
fn list_trials(api: &impl Api, name: &str, timeout: Duration) -> Result<Value> {
    api.json(
        &[
            "get",
            "trainjobs",
            "-l",
            &format!("trainer.kubeflow.org/optimization-job={name}"),
            "-o",
            "json",
        ],
        None,
        timeout,
    )
}
fn config_of(job: &Value) -> Result<ExperimentConfig> {
    let c: ExperimentConfig =
        serde_json::from_str(string(job, "/metadata/annotations/rubixkube.dev~1config")?)?;
    c.validate()?;
    if c.namespace != string(job, "/metadata/namespace")? {
        return Err(invalid(
            "Recorded configuration namespace differs from owning resource",
        ));
    }
    Ok(c)
}
pub fn status(api: &impl Api, name: &str) -> Result<Value> {
    let job = owned_job(api, name)?;
    let trials = list_trials(api, name, Duration::from_secs(20))?;
    let items = trials["items"].as_array().cloned().unwrap_or_default();
    let completed = items.iter().filter(|t| condition(t, "Complete")).count();
    let failed = items.iter().filter(|t| condition(t, "Failed")).count();
    Ok(
        json!({"name":name,"namespace":job["metadata"]["namespace"],"uid":job["metadata"]["uid"],"state":if condition(&job,"Failed"){"Failed"}else if condition(&job,"Complete"){"Complete"}else{"Running"},"trial_budget":job["spec"]["numTrials"],"observed_trials":items.len(),"completed_trials":completed,"failed_trials":failed,"conditions":job["status"]["conditions"],"result":job["status"]["result"]}),
    )
}
pub fn list(api: &impl Api) -> Result<Value> {
    api.json(
        &[
            "get",
            "optimizationjobs",
            "-l",
            "app.kubernetes.io/managed-by=rubixkube",
            "-o",
            "json",
        ],
        None,
        Duration::from_secs(20),
    )
}
fn snapshots(api: &impl Api, name: &str, job: &Value, output: &Path) {
    let _ = save_json(output, "failed-job.json", job);
    if let Ok(trials) = list_trials(api, name, Duration::from_secs(5)) {
        let _ = save_json(output, "trainjobs.json", &trials);
    }
    if let Ok(pods) = api.json(&["get", "pods", "-o", "json"], None, Duration::from_secs(5)) {
        let _ = save_json(output, "pods.json", &pods);
    }
    if let Ok(events) = api.json(
        &[
            "get",
            "events",
            "--field-selector",
            "type=Warning",
            "-o",
            "json",
        ],
        None,
        Duration::from_secs(5),
    ) {
        let _ = save_json(output, "warning-events.json", &events);
    }
}
pub fn watch(api: &impl Api, name: &str, timeout: Duration, output: &Path) -> Result<Value> {
    name_ok(name)?;
    let started = Instant::now();
    let mut previous = String::new();
    loop {
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(invalid(format!(
                "Watch timed out; experiment {name} continues running. Resume with watch {name}."
            )));
        }
        let job = api.json(
            &["get", "optimizationjob", name, "-o", "json"],
            None,
            remaining.min(Duration::from_secs(10)),
        )?;
        cluster::require_owned(&job, name)?;
        if condition(&job, "Failed") {
            snapshots(api, name, &job, output);
            return Err(invalid(format!(
                "Experiment {name} failed; diagnostics saved to {}",
                output.display()
            )));
        }
        if condition(&job, "Complete") {
            return collect(api, name, output);
        }
        let state = job["status"]["conditions"].to_string();
        if state != previous {
            eprintln!("{name}: {}", job["status"]["conditions"]);
            previous = state;
        }
        thread::sleep(Duration::from_millis(500).min(remaining));
    }
}
pub fn collect(api: &impl Api, name: &str, output: &Path) -> Result<Value> {
    let job = owned_job(api, name)?;
    let config = config_of(&job)?;
    let trials = list_trials(api, name, Duration::from_secs(20))?;
    let items = trials["items"]
        .as_array()
        .ok_or_else(|| invalid("Malformed TrainJob list"))?;
    let mut report = experiment_report::summarize(&job, items)?;
    let pods = api.json(
        &["get", "pods", "-o", "json"],
        None,
        Duration::from_secs(20),
    )?;
    let pod_items = pods["items"]
        .as_array()
        .ok_or_else(|| invalid("Malformed Pod list"))?;
    let rows = report["trials"]
        .as_array_mut()
        .ok_or_else(|| invalid("Missing trial rows"))?;
    let mut all_cuda = true;
    for row in rows.iter_mut() {
        let trial = string(row, "/name")?.to_owned();
        if !config::dns_label(&trial) {
            return Err(invalid("Invalid trial name"));
        }
        for parameter in config.parameters.keys() {
            if !row["parameters"][parameter].is_string() {
                return Err(invalid("Missing sampled parameter assignment"));
            }
        }
        let prefix = format!("{trial}-");
        let successful: Vec<&Value> = pod_items
            .iter()
            .filter(|p| {
                p["metadata"]["name"]
                    .as_str()
                    .is_some_and(|n| n.starts_with(&prefix))
                    && p["status"]["phase"] == "Succeeded"
            })
            .collect();
        if successful.len() != 1 {
            return Err(invalid(format!(
                "Expected one successful single-node worker Pod for {trial}"
            )));
        }
        let pod = successful[0];
        let pod_name = string(pod, "/metadata/name")?;
        let container = pod
            .pointer("/spec/containers")
            .and_then(Value::as_array)
            .and_then(|c| c.iter().find(|c| c["name"] == "node"))
            .ok_or_else(|| invalid("Expected trainer container node"))?;
        let allocated = container
            .pointer("/resources/limits/nvidia.com~1gpu")
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        if allocated < config.resources.gpus {
            return Err(invalid(
                "Worker GPU allocation differs from requested resources",
            ));
        }
        let logs = api.execute(
            &["logs", pod_name, "-c", "node"],
            None,
            Duration::from_secs(20),
        )?;
        save(output, &format!("{trial}.log"), &logs)?;
        let events: Vec<Value> = logs
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let training = events.iter().find(|e| e["event"] == "training_complete");
        let mut cuda = false;
        if let Some(event) = training {
            let objective = row["value"]
                .as_f64()
                .ok_or_else(|| invalid("Missing objective"))?;
            if allocated > 0 {
                if let Ok(proof) = cuda_evidence(event, &config.objective.metric, objective) {
                    row["gpu"] = serde_json::to_value(proof)?;
                    cuda = true;
                }
            }
            if event["objective_metric"].as_str().unwrap_or("val_loss") == config.objective.metric {
                row["history"] = event["history"].clone();
                row["dataset_rows"] = event["dataset_rows"].clone();
                row["dataset_source"] = event["dataset_source"].clone();
                row["epochs"] = event["epochs"].clone();
                row["weight"] = event["weight"].clone();
                row["bias"] = event["bias"].clone();
            }
        }
        if config.require_cuda_proof && !cuda {
            return Err(invalid(
                "Requested CUDA training proof missing or invalid; CPU fallback is not accepted",
            ));
        }
        all_cuda &= cuda;
        row["pod"] = json!(pod_name);
        row["gpu_allocation"] = json!(allocated);
        row["cuda_training_verified"] = json!(cuda);
        row["metric_delivery_observed"] =
            json!(events.iter().any(|e| e["event"] == "metric_delivered"));
    }
    report["best"] = report["trials"][0].clone();
    report["recorded_at"] = json!(chrono::Utc::now().to_rfc3339());
    report["cuda_training_verified"] = json!(all_cuda);
    save_json(output, "optimizationjob.json", &job)?;
    save_json(output, "trainjobs.json", &trials)?;
    save_json(output, "pods.json", &pods)?;
    save_json(output, "report.json", &report)?;
    crate::experiment_plots::write(&report, output)?;
    Ok(report)
}
pub fn logs(api: &impl Api, name: &str) -> Result<Value> {
    owned_job(api, name)?;
    let trials = list_trials(api, name, Duration::from_secs(20))?;
    let mut result = serde_json::Map::new();
    for trial in trials["items"]
        .as_array()
        .ok_or_else(|| invalid("Malformed trials"))?
    {
        let trial_name = string(trial, "/metadata/name")?;
        let pods = api.json(
            &[
                "get",
                "pods",
                "-l",
                &format!("jobset.sigs.k8s.io/jobset-name={trial_name}"),
                "-o",
                "json",
            ],
            None,
            Duration::from_secs(20),
        )?;
        for pod in pods["items"]
            .as_array()
            .ok_or_else(|| invalid("Malformed pods"))?
        {
            let name = string(pod, "/metadata/name")?;
            let text = api.execute(&["logs", name, "-c", "node"], None, Duration::from_secs(20))?;
            result.insert(name.to_owned(), json!(text));
        }
    }
    Ok(Value::Object(result))
}
pub fn cancel(api: &impl Api, name: &str, purge: bool) -> Result<Value> {
    name_ok(name)?;
    let job = cluster::get(api, "optimizationjob", name)?;
    let init_name = format!("{name}-init");
    let init = cluster::get(api, "job", &init_name)?;
    let claim_name = format!("{name}-data");
    let claim = if purge {
        cluster::get(api, "pvc", &claim_name)?
    } else {
        Value::Null
    };
    if job.is_null() && init.is_null() && claim.is_null() {
        return Err(invalid("Managed experiment/initializer/data not found"));
    }
    let mut resources = Vec::new();
    for (kind, resource, object) in [
        ("optimizationjob".to_owned(), name.to_owned(), job.clone()),
        ("job".to_owned(), init_name, init.clone()),
    ] {
        if !object.is_null() {
            cluster::require_owned(&object, name)?;
            resources.push((kind, resource));
        }
    }
    let recorded = if !job.is_null() {
        Some(config_of(&job)?)
    } else if !init.is_null() {
        Some(config_of(&init)?)
    } else {
        None
    };
    if let Some(c) = recorded.filter(|c| c.shared_data.is_some()) {
        let runtime_name = format!("{name}-runtime");
        let object = cluster::get(api, &c.runtime.kind, &runtime_name)?;
        if !object.is_null() {
            cluster::require_owned(&object, name)?;
            let expected = serde_json::to_string(&c)?;
            if object["metadata"]["annotations"]["rubixkube.dev/config"] != expected {
                return Err(invalid(
                    "Derived runtime belongs to another namespace/configuration; cleanup refused",
                ));
            }
            resources.push((c.runtime.kind, runtime_name));
        }
    }
    if !claim.is_null() {
        cluster::require_owned(&claim, name)?;
        resources.push(("pvc".to_owned(), claim_name));
    }
    let mut deleted = Vec::new();
    for (kind, resource) in resources {
        api.execute(
            &["delete", &kind, &resource, "--wait=true", "--timeout=45s"],
            None,
            Duration::from_secs(50),
        )?;
        deleted.push(format!("{kind}/{resource}"));
    }
    Ok(json!({"name":name,"deleted":deleted,"shared_data_retained":!purge}))
}

pub fn doctor(api: &impl Api, config: Option<&ExperimentConfig>) -> Result<Value> {
    let mut checks = Vec::new();
    for (kind, name) in [
        ("crd", "optimizationjobs.trainer.kubeflow.org"),
        ("crd", "trainjobs.trainer.kubeflow.org"),
    ] {
        let check = cluster::get(api, kind, name);
        let ready = check.as_ref().is_ok_and(|v| !v.is_null());
        checks.push(json!({"check":name,"ready":ready,"error":check.err().map(|e|e.to_string())}));
    }
    let permission = api.execute(
        &[
            "auth",
            "can-i",
            "create",
            "optimizationjobs.trainer.kubeflow.org",
        ],
        None,
        Duration::from_secs(15),
    );
    let ready = permission.as_ref().is_ok_and(|s| s.trim() == "yes");
    checks.push(json!({"check":"create OptimizationJob permission","ready":ready,"error":permission.err().map(|e|e.to_string())}));
    if let Some(c) = config {
        let runtime = cluster::get(api, &c.runtime.kind, &c.runtime.name);
        checks.push(json!({"check":"configured runtime","ready":runtime.as_ref().is_ok_and(|v|!v.is_null()),"error":runtime.err().map(|e|e.to_string())}));
        if c.resources.gpus > 0 {
            let nodes = api.json(
                &["get", "nodes", "-o", "json"],
                None,
                Duration::from_secs(20),
            );
            let available = nodes
                .as_ref()
                .ok()
                .and_then(|n| n["items"].as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|n| {
                            n.pointer("/status/allocatable/nvidia.com~1gpu")
                                .and_then(Value::as_str)
                                .and_then(|s| s.parse::<u32>().ok())
                        })
                        .sum::<u32>()
                })
                .unwrap_or(0);
            checks.push(json!({"check":"advertised GPU capacity","ready":available>=c.resources.gpus,"allocatable":available,"error":nodes.err().map(|e|e.to_string())}));
        }
    }
    Ok(
        json!({"ready":checks.iter().all(|c|c["ready"]==true),"checks":checks,"note":"Admission, image availability, PVC binding and workload behavior are validated during submission/execution; this is not a production security certification."}),
    )
}

/// Validate the built-in CUDA evidence contract against a named final objective.
pub fn cuda_evidence(
    event: &Value,
    metric: &str,
    objective: f64,
) -> Result<crate::training::GpuEvidence> {
    let result: crate::training::TrainingResult = serde_json::from_value(event.clone())?;
    let name = event["objective_metric"].as_str().unwrap_or("val_loss");
    let proof = result
        .gpu
        .ok_or_else(|| invalid("Missing CUDA device evidence"))?;
    if name != metric
        || result.execution != "GPU, Rust CUDA; device training verified"
        || !objective.is_finite()
        || !result.val_loss.is_finite()
        || (result.val_loss - objective).abs() > 1e-10
        || result.epochs == 0
        || result.history.len() != result.epochs
        || result
            .history
            .iter()
            .enumerate()
            .any(|(i, h)| h.epoch != i + 1 || !h.val_loss.is_finite())
        || result
            .history
            .last()
            .is_none_or(|h| (h.val_loss - objective).abs() > 1e-10)
        || !result.weight.is_finite()
        || !result.bias.is_finite()
        || proof.device_name.is_empty()
        || proof.compute_capability[0] <= 0
        || proof.kernel_launches == 0
        || !proof.cuda_elapsed_ms.is_finite()
        || proof.cuda_elapsed_ms <= 0.
        || !proof.cpu_reference_max_abs_error.is_finite()
        || !(0.0..=1e-10).contains(&proof.cpu_reference_max_abs_error)
    {
        return Err(invalid(
            "CUDA evidence does not prove this metric/history and finite device computation",
        ));
    }
    Ok(proof)
}
