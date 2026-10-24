//! Explicit-context Kubernetes transport; does not use the user's implicit context.
use crate::{config, invalid, process::checked, Result};
use serde_json::Value;
use std::{path::PathBuf, process::Command, time::Duration};
pub trait Api {
    fn execute(&self, args: &[&str], payload: Option<&Value>, timeout: Duration) -> Result<String>;
    fn json(&self, args: &[&str], payload: Option<&Value>, timeout: Duration) -> Result<Value> {
        let text = self.execute(args, payload, timeout)?;
        if text.trim().is_empty() {
            Ok(Value::Null)
        } else {
            Ok(serde_json::from_str(&text)?)
        }
    }
}
#[derive(Debug, Clone)]
pub struct Cluster {
    pub kubeconfig: PathBuf,
    pub context: String,
    pub namespace: String,
}
impl Cluster {
    pub fn new(kubeconfig: PathBuf, context: String, namespace: String) -> Result<Self> {
        if context.is_empty()
            || context.starts_with('-')
            || context.chars().any(char::is_control)
            || !config::dns_label(&namespace)
        {
            return Err(invalid("Specify an explicit context and a valid namespace"));
        }
        Ok(Self {
            kubeconfig: kubeconfig.canonicalize()?,
            context,
            namespace,
        })
    }
}
impl Api for Cluster {
    fn execute(&self, args: &[&str], payload: Option<&Value>, timeout: Duration) -> Result<String> {
        let bytes = payload.map(serde_json::to_vec).transpose()?;
        checked(
            Command::new("kubectl")
                .arg("--kubeconfig")
                .arg(&self.kubeconfig)
                .args(["--context", &self.context, "--namespace", &self.namespace])
                .args(args),
            bytes.as_deref(),
            timeout,
        )
    }
}
pub fn get(api: &impl Api, kind: &str, name: &str) -> Result<Value> {
    api.json(
        &["get", kind, name, "--ignore-not-found", "-o", "json"],
        None,
        Duration::from_secs(20),
    )
}
pub fn create(api: &impl Api, object: &Value) -> Result<Value> {
    api.json(
        &["create", "-f", "-", "-o", "json"],
        Some(object),
        Duration::from_secs(30),
    )
}
pub fn owned(object: &Value, name: &str) -> bool {
    object["metadata"]["labels"][config::MANAGED] == "rubixkube"
        && object["metadata"]["labels"][config::RUN_LABEL] == name
}
pub fn require_owned(object: &Value, name: &str) -> Result<()> {
    if !owned(object, name) {
        Err(invalid(
            "Resource is not owned by this RubixKube run; refusing to mutate it",
        ))
    } else {
        Ok(())
    }
}
