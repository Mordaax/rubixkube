use rubixkube::orchestrate::validate_worker;
use serde_json::json;
use std::process::Command;
#[test]
fn cli_help_and_invalid_commands() {
    let executable = env!("CARGO_BIN_EXE_optimizationjob");
    let output = Command::new(executable).arg("--help").output().unwrap();
    assert!(output.status.success());
    for command in ["train", "run", "start", "smoke", "plot"] {
        assert!(String::from_utf8_lossy(&output.stdout).contains(command));
    }
    assert!(!Command::new(executable)
        .arg("invalid-subcommand")
        .status()
        .unwrap()
        .success());
    assert!(!Command::new(executable)
        .args(["train", "--local", "--learning-rate", "5"])
        .status()
        .unwrap()
        .success());
}
#[test]
fn local_cli_emits_real_training_without_claiming_delivery() {
    let output = Command::new(env!("CARGO_BIN_EXE_optimizationjob"))
        .args(["train", "--local", "--learning-rate", "0.1"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["event"], "training_complete");
    assert!(result["val_loss"].as_f64().unwrap() < 0.02);
    assert!(!String::from_utf8_lossy(&output.stdout).contains("metric_delivered"));
}
#[test]
fn worker_evidence_requires_training_acknowledgement_and_consistent_loss() {
    let trained = rubixkube::training::train(0.1, 40).unwrap();
    let mut event = serde_json::to_value(&trained).unwrap();
    event["event"] = json!("training_complete");
    let logs = format!(
        "{event}\n{}\n",
        json!({"event":"metric_delivered","attempt":1})
    );
    let row = json!({"learning_rate":0.1,"val_loss":trained.val_loss});
    assert!(validate_worker(&row, &logs).is_ok());
    assert!(validate_worker(&row, &event.to_string()).is_err());
    let bad_row = json!({"learning_rate":0.1,"val_loss":0.8});
    assert!(validate_worker(&bad_row, &logs).is_err());
    event["execution"] = json!("CPU, Python standard library; no GPU or simulated score");
    assert!(validate_worker(
        &row,
        &format!("{event}\n{}", json!({"event":"metric_delivered"}))
    )
    .is_err());
}
#[test]
fn generated_job_names_leave_room_for_upstream_trial_and_pod_suffixes() {
    let name = rubixkube::orchestrate::new_job_name();
    assert!(name.starts_with("rust-hpo-"));
    // Observed upstream suffix: -trial-xxxxxxxx-node-0-0-xxxxx (29 chars).
    assert!(name.len() + 29 <= 63, "{name}");
    assert!(name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'));
}
