//! A run-specific initializer Job and retained PVC shared read-only by trials.
use crate::{
    cluster::{self, Api},
    config::{self, ExperimentConfig},
    invalid,
    report::condition,
    Result,
};
use serde_json::{json, Value};
use std::{
    thread,
    time::{Duration, Instant},
};
fn metadata(config: &ExperimentConfig, run: &str, name: &str) -> Result<Value> {
    Ok(
        json!({"name":name,"namespace":config.namespace,"labels":{config::MANAGED:"rubixkube",config::RUN_LABEL:run},"annotations":{"rubixkube.dev/config":serde_json::to_string(config)?}}),
    )
}
pub fn pvc(config: &ExperimentConfig, run: &str) -> Result<Value> {
    let shared = config
        .shared_data
        .as_ref()
        .ok_or_else(|| invalid("Missing shared data"))?;
    let mut claim = json!({"apiVersion":"v1","kind":"PersistentVolumeClaim","metadata":metadata(config,run,&format!("{run}-data"))?,"spec":{"accessModes":["ReadWriteOnce"],"resources":{"requests":{"storage":shared.storage}}}});
    if let Some(class) = &shared.storage_class {
        claim["spec"]["storageClassName"] = json!(class);
    }
    Ok(claim)
}
pub fn initializer(config: &ExperimentConfig, run: &str) -> Result<Value> {
    let shared = config
        .shared_data
        .as_ref()
        .ok_or_else(|| invalid("Missing shared data"))?;
    let mut container = json!({"name":"initializer","image":shared.initializer.image,"env":shared.initializer.environment(),"resources":{"requests":{"cpu":"100m","memory":"64Mi"},"limits":{"cpu":"1","memory":"512Mi"}},"securityContext":{"allowPrivilegeEscalation":false,"readOnlyRootFilesystem":true,"capabilities":{"drop":["ALL"]}},"volumeMounts":[{"name":"data","mountPath":shared.mount_path},{"name":"tmp","mountPath":"/tmp"}]});
    if !shared.initializer.command.is_empty() {
        container["command"] = json!(shared.initializer.command);
    }
    if !shared.initializer.args.is_empty() {
        container["args"] = json!(shared.initializer.args);
    }
    Ok(
        json!({"apiVersion":"batch/v1","kind":"Job","metadata":metadata(config,run,&format!("{run}-init"))?,"spec":{"backoffLimit":0,"activeDeadlineSeconds":config.timeout_seconds,"template":{"metadata":{"labels":{config::MANAGED:"rubixkube",config::RUN_LABEL:run}},"spec":{"restartPolicy":"Never","securityContext":{"runAsNonRoot":true,"runAsUser":65532,"fsGroup":65532,"seccompProfile":{"type":"RuntimeDefault"}},"containers":[container],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":format!("{run}-data")}},{"name":"tmp","emptyDir":{"sizeLimit":"128Mi"}}]}}}}),
    )
}
pub fn runtime(config: &ExperimentConfig, run: &str, source: &Value) -> Result<Value> {
    let shared = config
        .shared_data
        .as_ref()
        .ok_or_else(|| invalid("Missing shared data"))?;
    let mut result = source.clone();
    let mut meta = metadata(config, run, &format!("{run}-runtime"))?;
    if config.runtime.kind == "ClusterTrainingRuntime" {
        meta.as_object_mut()
            .ok_or_else(|| invalid("Bad metadata"))?
            .remove("namespace");
    }
    // Keep upstream runtime labels/annotations that control runtime behavior.
    for key in ["labels", "annotations"] {
        if let Some(original) = source["metadata"][key].as_object() {
            for (k, v) in original {
                if !k.starts_with("rubixkube.dev/") && k != config::MANAGED {
                    meta[key][k] = v.clone();
                }
            }
        }
    }
    result["metadata"] = meta;
    result
        .as_object_mut()
        .ok_or_else(|| invalid("Bad runtime"))?
        .remove("status");
    let jobs = result
        .pointer_mut("/spec/template/spec/replicatedJobs")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| invalid("Runtime has no replicated jobs"))?;
    if jobs.len() != 1 {
        return Err(invalid(
            "Shared-data v0.2 requires one node replicatedJob in the runtime",
        ));
    }
    let spec = jobs[0]
        .pointer_mut("/template/spec/template/spec")
        .ok_or_else(|| invalid("Runtime lacks a Pod template"))?;
    if spec["volumes"].is_null() {
        spec["volumes"] = json!([]);
    }
    let volumes = spec["volumes"]
        .as_array_mut()
        .ok_or_else(|| invalid("Invalid runtime volumes"))?;
    if volumes.iter().any(|v| v["name"] == "rubixkube-shared") {
        return Err(invalid("Runtime already has a rubixkube-shared volume"));
    }
    volumes.push(json!({"name":"rubixkube-shared","persistentVolumeClaim":{"claimName":format!("{run}-data"),"readOnly":true}}));
    let containers = spec["containers"]
        .as_array_mut()
        .ok_or_else(|| invalid("Runtime lacks containers"))?;
    let container = containers
        .iter_mut()
        .find(|c| c["name"] == "node")
        .ok_or_else(|| invalid("Shared-data runtime needs a trainer container named node"))?;
    if container["volumeMounts"].is_null() {
        container["volumeMounts"] = json!([]);
    }
    let mounts = container["volumeMounts"]
        .as_array_mut()
        .ok_or_else(|| invalid("Invalid runtime volume mounts"))?;
    if mounts.iter().any(|m| m["mountPath"] == shared.mount_path) {
        return Err(invalid(
            "Shared-data mount path is already used by the runtime",
        ));
    }
    mounts.push(json!({"name":"rubixkube-shared","mountPath":shared.mount_path,"readOnly":true}));
    Ok(result)
}
fn ensure(api: &impl Api, run: &str, expected: &Value) -> Result<Value> {
    let kind = expected["kind"]
        .as_str()
        .ok_or_else(|| invalid("Missing kind"))?;
    let name = expected["metadata"]["name"]
        .as_str()
        .ok_or_else(|| invalid("Missing name"))?;
    let existing = cluster::get(api, kind, name)?;
    if existing.is_null() {
        cluster::create(api, expected)
    } else {
        cluster::require_owned(&existing, run)?;
        if existing["metadata"]["annotations"]["rubixkube.dev/config"]
            != expected["metadata"]["annotations"]["rubixkube.dev/config"]
        {
            return Err(invalid(
                "Existing shared resource has another configuration; use a new run name",
            ));
        }
        Ok(existing)
    }
}
pub fn prepare(api: &impl Api, config: &ExperimentConfig, run: &str) -> Result<Vec<Value>> {
    if config.shared_data.is_none() {
        return Ok(Vec::new());
    }
    let source = cluster::get(api, &config.runtime.kind, &config.runtime.name)?;
    if source.is_null() {
        return Err(invalid("Base runtime was not found"));
    }
    let derived = runtime(config, run, &source)?;
    let claim = ensure(api, run, &pvc(config, run)?)?;
    let init = ensure(api, run, &initializer(config, run)?)?;
    let start = Instant::now();
    let name = format!("{run}-init");
    let mut observed = init;
    loop {
        if condition(&observed, "Failed") {
            return Err(invalid(format!("Initializer failed; inspect Job {name} and its logs. Optimization was not submitted.")));
        }
        if condition(&observed, "Complete") {
            break;
        }
        let remaining = Duration::from_secs(config.timeout_seconds).saturating_sub(start.elapsed());
        if remaining.is_zero() {
            return Err(invalid("Initializer deadline exceeded; OptimizationJob not submitted. Shared PVC retained; retry same run name to resume initialization."));
        }
        thread::sleep(Duration::from_millis(500).min(remaining));
        observed = api.json(
            &["get", "job", &name, "-o", "json"],
            None,
            remaining.min(Duration::from_secs(10)),
        )?;
    }
    let runtime = ensure(api, run, &derived)?;
    Ok(vec![claim, observed, runtime])
}
