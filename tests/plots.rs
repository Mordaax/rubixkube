use rubixkube::plots::write_plots;
use serde_json::{json, Value};

fn report() -> Value {
    let mut rows = Vec::new();
    for (name, rate, losses) in [
        ("trial-fast", 0.1, vec![1.6, 0.5, 0.01]),
        ("trial-slow", 0.005, vec![2.0, 1.8, 1.2]),
    ] {
        rows.push(json!({"name":name,"learning_rate":rate,"val_loss":losses[2],"weight":1.9,"bias":1.0,"metric_delivered":true,"history":losses.iter().enumerate().map(|(i,loss)|json!({"epoch":i+1,"val_loss":loss})).collect::<Vec<_>>()}));
    }
    json!({"job":"saved-run","winner_verified":true,"objective":{"metric":"val_loss","direction":"Minimize"},"best":rows[0],"trials":rows,"coverage":{"gpu":false},"mode":"Real Kubernetes CPU run"})
}
#[test]
fn creates_standalone_graphs_and_exact_csv_without_a_web_app() {
    let dir = tempfile::tempdir().unwrap();
    write_plots(&report(), dir.path()).unwrap();
    for file in ["loss-by-epoch.svg", "trial-results.svg"] {
        let svg = std::fs::read_to_string(dir.path().join(file)).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Validation MSE"));
        assert!(!svg.contains("<script"));
    }
    for name in ["loss-by-epoch.png", "trial-results.png"] {
        let image = std::fs::read(dir.path().join(name)).unwrap();
        assert_eq!(&image[..8], b"\x89PNG\r\n\x1a\n");
    }
    let csv = std::fs::read_to_string(dir.path().join("results.csv")).unwrap();
    assert!(csv.contains("trial-fast,0.1,0.01"));
    assert!(csv.contains("trial-slow,0.005,1.2"));
    let summary = std::fs::read_to_string(dir.path().join("summary.txt")).unwrap();
    assert!(summary.contains("trial-fast"));
    assert!(!dir.path().join("index.html").exists());
}
#[test]
fn rejects_unverified_missing_nonfinite_or_inconsistent_observations() {
    let dir = tempfile::tempdir().unwrap();
    let mut data = report();
    data["winner_verified"] = json!(false);
    assert!(write_plots(&data, dir.path()).is_err());
    let mut data = report();
    data["trials"] = json!([]);
    assert!(write_plots(&data, dir.path()).is_err());
    let mut data = report();
    data["trials"][0]["history"][0]["val_loss"] = json!("NaN");
    assert!(write_plots(&data, dir.path()).is_err());
    let mut data = report();
    data["trials"][0]["val_loss"] = json!(5.);
    assert!(write_plots(&data, dir.path()).is_err());
    assert!(!dir.path().join("loss-by-epoch.svg").exists());
}
#[test]
fn cli_plots_saved_report_without_contacting_a_cluster() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("report.json");
    std::fs::write(&input, report().to_string()).unwrap();
    let output = dir.path().join("plots");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_optimizationjob"))
        .arg("plot")
        .arg("--report")
        .arg(input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.join("trial-results.svg").exists());
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_optimizationjob"))
        .arg("serve")
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "The web server should no longer be available"
    );
}
