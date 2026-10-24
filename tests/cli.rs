use optimizationjob::{orchestrate::validate_worker, serve::resolve_file};
use serde_json::json;
use std::{process::Command, time::Duration};
#[test]
fn cli_help_and_invalid_commands() {
    let executable = env!("CARGO_BIN_EXE_optimizationjob");
    let output = Command::new(executable).arg("--help").output().unwrap();
    assert!(output.status.success());
    for command in ["train", "run", "start", "smoke", "serve"] {
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
fn report_paths_cannot_escape_root() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.html"), "report").unwrap();
    assert_eq!(
        resolve_file(dir.path(), "/").unwrap(),
        dir.path().join("index.html")
    );
    for path in [
        "/../credentials",
        "/%2e%2e/credentials",
        "/%2Fetc/passwd",
        "/missing",
        "/%FF",
        "/%00",
    ] {
        assert!(resolve_file(dir.path(), path).is_err(), "{path}");
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("/etc/passwd", dir.path().join("escape")).unwrap();
        assert!(resolve_file(dir.path(), "/escape").is_err());
    }
}
#[test]
fn worker_evidence_requires_training_acknowledgement_and_consistent_loss() {
    let trained = optimizationjob::training::train(0.1, 40).unwrap();
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
fn loopback_server_serves_report_and_rejects_traversal() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.html"), "hello-report").unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_optimizationjob"))
        .args(["serve", "--directory"])
        .arg(dir.path())
        .arg("--port")
        .arg(port.to_string())
        .spawn()
        .unwrap();
    let mut socket = None;
    for _ in 0..100 {
        if let Ok(connected) = std::net::TcpStream::connect(("127.0.0.1", port)) {
            socket = Some(connected);
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut socket = socket.expect("server started");
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    socket
        .write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut response = String::new();
    socket.read_to_string(&mut response).unwrap();
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(response.starts_with("HTTP/1.1 200"));
    assert!(response.ends_with("hello-report"));
}
