//! Authenticated final-metric reporting using the pinned Trainer status protocol.
use crate::{invalid, Result};
use reqwest::{blocking::Client, redirect::Policy, Certificate, StatusCode, Url};
use serde_json::json;
use std::{env, fs, path::PathBuf, thread, time::Duration};

/// Explicit transport and retry settings; tokens are read from disk per attempt.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    pub url: String,
    pub ca_cert: PathBuf,
    pub token_path: PathBuf,
    pub attempts: usize,
    pub retry_delay: Duration,
    pub request_timeout: Duration,
}
impl MetricsConfig {
    /// Read the three variables projected by Trainer's TrainJobStatus plugin.
    pub fn from_env() -> Result<Self> {
        fn required(name: &str) -> Result<String> {
            env::var(name).map_err(|_| {
                invalid(format!(
                    "Missing {name}; run inside a TrainJob with TrainJobStatus enabled"
                ))
            })
        }
        Ok(Self {
            url: required("KUBEFLOW_TRAINER_SERVER_URL")?,
            ca_cert: required("KUBEFLOW_TRAINER_SERVER_CA_CERT")?.into(),
            token_path: required("KUBEFLOW_TRAINER_SERVER_TOKEN")?.into(),
            attempts: 8,
            retry_delay: Duration::from_millis(5500),
            request_timeout: Duration::from_secs(5),
        })
    }
}
/// Deliver one finite final objective; only HTTP 200 is an acknowledgement.
///
/// Certificate and hostname verification are mandatory. Redirects and proxies
/// are disabled so projected credentials go only to the configured endpoint.
pub fn deliver(config: &MetricsConfig, loss: f64) -> Result<usize> {
    if !loss.is_finite()
        || config.attempts == 0
        || config.attempts > 8
        || config.request_timeout.is_zero()
    {
        return Err(invalid("Invalid metric delivery configuration"));
    }
    let url = Url::parse(&config.url).map_err(|_| invalid("Invalid Trainer status URL"))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid(
            "Trainer status endpoint must be HTTPS without embedded credentials, query or fragment",
        ));
    }
    let certificate = Certificate::from_pem(&fs::read(&config.ca_cert)?)?;
    let client = Client::builder()
        .https_only(true)
        .tls_built_in_root_certs(false)
        .add_root_certificate(certificate)
        .no_proxy()
        .redirect(Policy::none())
        .timeout(config.request_timeout)
        .build()?;
    let mut failure = "No request attempted".to_owned();
    for attempt in 1..=config.attempts {
        let token = fs::read_to_string(&config.token_path)?;
        let token = token.trim();
        if token.is_empty() {
            return Err(invalid("Projected token is empty"));
        }
        let payload = json!({"trainerStatus":{
            "lastUpdatedTime":chrono::Utc::now().to_rfc3339(),
            "progressPercentage":100,
            "metrics":[{"name":"val_loss","value":loss.to_string()}]
        }});
        match client
            .post(url.clone())
            .bearer_auth(token)
            .json(&payload)
            .send()
        {
            Ok(response) if response.status() == StatusCode::OK => return Ok(attempt),
            Ok(response) => failure = format!("HTTP {}", response.status().as_u16()),
            // Do not print reqwest's full error chain: credentials and endpoint
            // configuration need not be exposed in public Pod logs.
            Err(_) => failure = "TLS or transport error".into(),
        }
        if attempt < config.attempts {
            thread::sleep(config.retry_delay);
        }
    }
    Err(invalid(format!(
        "Metric not acknowledged after {} attempts ({failure})",
        config.attempts
    )))
}
