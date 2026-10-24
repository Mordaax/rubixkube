//! Versioned experiment configuration and typed upstream manifest generation.
use crate::{invalid, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
pub const MANAGED: &str = "app.kubernetes.io/managed-by";
pub const RUN_LABEL: &str = "rubixkube.dev/run";
fn one() -> u32 {
    1
}
fn trials() -> u32 {
    4
}
fn timeout() -> u64 {
    600
}
fn cpu() -> String {
    "100m".into()
}
fn memory() -> String {
    "256Mi".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    #[serde(default = "one")]
    pub version: u32,
    pub namespace: String,
    pub runtime: Runtime,
    pub workload: Workload,
    pub objective: Objective,
    #[serde(default = "trials")]
    pub trials: u32,
    #[serde(default = "one")]
    pub parallelism: u32,
    #[serde(default)]
    pub seed: u32,
    pub parameters: BTreeMap<String, Space>,
    #[serde(default)]
    pub resources: Resources,
    #[serde(default)]
    pub require_cuda_proof: bool,
    #[serde(default)]
    pub shared_data: Option<SharedData>,
    #[serde(default = "timeout")]
    pub timeout_seconds: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub name: String,
    #[serde(default = "cluster_kind")]
    pub kind: String,
}
fn cluster_kind() -> String {
    "ClusterTrainingRuntime".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub image: String,
    #[serde(default)]
    pub command: Vec<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub secret_env: BTreeMap<String, SecretRef>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecretRef {
    pub name: String,
    pub key: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    pub metric: String,
    #[serde(default = "minimize")]
    pub direction: String,
}
fn minimize() -> String {
    "Minimize".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Space {
    Categorical {
        choices: Vec<String>,
    },
    Uniform {
        min: f64,
        max: f64,
        #[serde(default)]
        integer: bool,
    },
    LogUniform {
        min: f64,
        max: f64,
        #[serde(default)]
        integer: bool,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resources {
    #[serde(default = "cpu")]
    pub cpu: String,
    #[serde(default = "memory")]
    pub memory: String,
    #[serde(default)]
    pub gpus: u32,
}
impl Default for Resources {
    fn default() -> Self {
        Self {
            cpu: cpu(),
            memory: memory(),
            gpus: 0,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedData {
    pub storage: String,
    #[serde(default)]
    pub storage_class: Option<String>,
    pub mount_path: String,
    pub initializer: Workload,
}
pub fn dns_label(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 63
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
}
pub fn dns_name(name: &str) -> bool {
    name.len() <= 253 && name.split('.').all(dns_label)
}
pub fn env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
pub fn metric_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
}
fn quantity(value: &str) -> bool {
    if value.is_empty() || value.len() > 32 {
        return false;
    }
    if let Ok(number) = value.parse::<f64>() {
        return number.is_finite() && number > 0.;
    }
    let split = value
        .find(|c: char| c.is_ascii_alphabetic())
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(split);
    number.parse::<f64>().is_ok_and(|n| n.is_finite() && n > 0.)
        && matches!(
            suffix,
            "n" | "u"
                | "m"
                | "k"
                | "K"
                | "M"
                | "G"
                | "T"
                | "P"
                | "E"
                | "Ki"
                | "Mi"
                | "Gi"
                | "Ti"
                | "Pi"
                | "Ei"
        )
}

impl Workload {
    pub fn validate(&self) -> Result<()> {
        if self.image.is_empty()
            || self.image.len() > 500
            || self.image.chars().any(char::is_whitespace)
            || self.command.len() > 128
            || self.args.len() > 128
        {
            return Err(invalid(
                "Workload needs an image and at most 128 command/args entries",
            ));
        }
        for name in self.env.keys().chain(self.secret_env.keys()) {
            if !env_name(name) || name.starts_with("KUBEFLOW_") {
                return Err(invalid(format!(
                    "Invalid or reserved environment variable {name}"
                )));
            }
        }
        if self.env.len() + self.secret_env.len() > 100
            || self.env.keys().any(|k| self.secret_env.contains_key(k))
        {
            return Err(invalid("Environment keys must be unique and at most 100"));
        }
        for secret in self.secret_env.values() {
            if !dns_name(&secret.name)
                || secret.key.is_empty()
                || secret.key.len() > 253
                || !secret
                    .key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            {
                return Err(invalid("Invalid Secret reference"));
            }
        }
        Ok(())
    }
    pub fn environment(&self) -> Vec<Value> {
        let mut env: Vec<Value> = self
            .env
            .iter()
            .map(|(name, value)| json!({"name":name,"value":value}))
            .collect();
        env.extend(self.secret_env.iter().map(|(name,s)|json!({"name":name,"valueFrom":{"secretKeyRef":{"name":s.name,"key":s.key}}})));
        env
    }
}
impl ExperimentConfig {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() > 65536 {
            return Err(invalid("Configuration exceeds 64 KiB"));
        }
        let config: Self = serde_json::from_slice(&bytes)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        if self.version != 1 {
            return Err(invalid(
                "Unsupported config version; supported version is 1",
            ));
        }
        if !dns_label(&self.namespace)
            || !dns_name(&self.runtime.name)
            || !matches!(
                self.runtime.kind.as_str(),
                "ClusterTrainingRuntime" | "TrainingRuntime"
            )
        {
            return Err(invalid("Invalid namespace or runtime reference"));
        }
        self.workload.validate()?;
        if !metric_name(&self.objective.metric)
            || !matches!(self.objective.direction.as_str(), "Minimize" | "Maximize")
        {
            return Err(invalid(
                "Objective requires a metric and Minimize/Maximize direction",
            ));
        }
        if !(1..=100).contains(&self.trials)
            || self.parallelism == 0
            || self.parallelism > self.trials
        {
            return Err(invalid(
                "Trials must be 1..100; parallelism must be 1..trials",
            ));
        }
        if self.parameters.is_empty() || self.parameters.len() > 100 {
            return Err(invalid("Specify 1..100 parameters"));
        }
        for (name, space) in &self.parameters {
            if !env_name(name) || name.len() > 64 {
                return Err(invalid(format!(
                    "Parameter {name} must be an environment-compatible identifier"
                )));
            }
            match space {
                Space::Categorical { choices } => {
                    let unique: std::collections::BTreeSet<_> = choices.iter().collect();
                    if choices.is_empty()
                        || choices.len() > 100
                        || unique.len() != choices.len()
                        || choices
                            .iter()
                            .any(|v| v.is_empty() || v.len() > 64 || v.contains('\0'))
                    {
                        return Err(invalid("Categorical choices must be unique, 1..100 values, at most 64 bytes each"));
                    }
                }
                Space::Uniform { min, max, integer } | Space::LogUniform { min, max, integer } => {
                    if !min.is_finite()
                        || !max.is_finite()
                        || min >= max
                        || (*integer && (min.fract() != 0. || max.fract() != 0.))
                    {
                        return Err(invalid(
                            "Search bounds must be finite, ordered and integral for integer spaces",
                        ));
                    }
                    if matches!(space, Space::LogUniform { .. })
                        && (*min <= 0. || (!*integer && *min < 1e-10))
                    {
                        return Err(invalid("Pinned Optuna backend requires floating log-uniform min >= 1e-10 (integer min > 0)"));
                    }
                }
            }
        }
        if !quantity(&self.resources.cpu)
            || !quantity(&self.resources.memory)
            || self.resources.gpus > 16
        {
            return Err(invalid("Invalid compute resources"));
        }
        if self.require_cuda_proof && self.resources.gpus == 0 {
            return Err(invalid("CUDA proof requires at least one requested GPU"));
        }
        if self.timeout_seconds == 0 || self.timeout_seconds > 86400 {
            return Err(invalid("Timeout must be 1..86400 seconds"));
        }
        if let Some(shared) = &self.shared_data {
            shared.initializer.validate()?;
            if !quantity(&shared.storage)
                || !shared.mount_path.starts_with('/')
                || shared.mount_path == "/"
                || shared.mount_path.contains("..")
                || shared.mount_path.starts_with("/usr")
                || shared.mount_path.starts_with("/var/run")
                || shared.mount_path.starts_with("/proc")
                || shared.mount_path.starts_with("/dev")
                || shared.mount_path.starts_with("/etc")
            {
                return Err(invalid(
                    "Shared data needs storage and a safe absolute mount_path, such as /dataset",
                ));
            }
            if shared.storage_class.as_ref().is_some_and(|s| !dns_name(s)) {
                return Err(invalid("Invalid storage class"));
            }
        }
        if serde_json::to_vec(self)?.len() > 65536 {
            return Err(invalid("Configuration exceeds 64 KiB"));
        }
        Ok(())
    }
}
pub fn manifest(config: &ExperimentConfig, name: &str) -> Result<Value> {
    config.validate()?;
    if !dns_label(name) || name.len() > 28 {
        return Err(invalid("Experiment name must be a DNS label of at most 28 characters to fit upstream Pod suffixes"));
    }
    let parameters:Vec<Value>=config.parameters.iter().map(|(name,space)|{
        let space=match space {Space::Categorical{choices}=>json!({"categorical":{"choices":choices}}),Space::Uniform{min,max,integer}=>json!({"uniform":{"min":min.to_string(),"max":max.to_string(),"type":if *integer{"Int"}else{"Float"}}}),Space::LogUniform{min,max,integer}=>json!({"logUniform":{"min":min.to_string(),"max":max.to_string(),"type":if *integer{"Int"}else{"Float"}}})};json!({"name":name,"searchSpace":space})}).collect();
    let mut trainer = json!({"image":config.workload.image,"numNodes":1,"env":config.workload.environment(),"resourcesPerNode":{"requests":{"cpu":config.resources.cpu,"memory":config.resources.memory},"limits":{"cpu":config.resources.cpu,"memory":config.resources.memory}}});
    if !config.workload.command.is_empty() {
        trainer["command"] = json!(config.workload.command);
    }
    if !config.workload.args.is_empty() {
        trainer["args"] = json!(config.workload.args);
    }
    if config.resources.gpus > 0 {
        let value = config.resources.gpus.to_string();
        trainer["resourcesPerNode"]["requests"]["nvidia.com/gpu"] = json!(value);
        trainer["resourcesPerNode"]["limits"]["nvidia.com/gpu"] = json!(value);
    }
    Ok(
        json!({"apiVersion":"trainer.kubeflow.org/v1alpha1","kind":"OptimizationJob","metadata":{"name":name,"namespace":config.namespace,"labels":{MANAGED:"rubixkube",RUN_LABEL:name},"annotations":{"rubixkube.dev/config":serde_json::to_string(config)?}},"spec":{"objectives":[{"metric":config.objective.metric,"direction":config.objective.direction}],"searchAlgorithm":{"random":{"seed":config.seed}},"parameters":parameters,"numTrials":config.trials,"parallelTrials":config.parallelism,"trainJobTemplate":{"spec":{"runtimeRef":{"name":if config.shared_data.is_some(){format!("{name}-runtime")}else{config.runtime.name.clone()},"kind":config.runtime.kind,"apiGroup":"trainer.kubeflow.org"},"trainer":trainer}}}}),
    )
}
pub fn example(gpu: bool) -> ExperimentConfig {
    serde_json::from_value(json!({"version":1,"namespace":"optimizationjob-demo","runtime":{"name":if gpu{"rubixkube-demo-gpu"}else{"optimizationjob-demo-cpu"}},"workload":{"image":if gpu{"rubixkube-training:cuda"}else{"optimizationjob-demo-training:rust"}},"objective":{"metric":"val_loss","direction":"Minimize"},"trials":4,"parallelism":1,"seed":42,"parameters":{"learning_rate":{"type":"categorical","choices":["0.005","0.02","0.05","0.1"]}},"resources":{"cpu":"100m","memory":"256Mi","gpus":if gpu{1}else{0}},"require_cuda_proof":gpu,"timeout_seconds":600})).expect("Built-in example configuration is valid")
}
