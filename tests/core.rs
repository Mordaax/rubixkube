use optimizationjob::{report::summarize, training::train};
use serde_json::{json, Value};

#[test]
fn training_matches_all_original_python_results() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("../demo/fixtures/parity.json")).unwrap();
    for case in cases {
        let result = serde_json::to_value(train(case["learning_rate"].as_f64().unwrap(), 40).unwrap()).unwrap();
        for field in ["val_loss", "weight", "bias"] {
            assert!((result[field].as_f64().unwrap() - case[field].as_f64().unwrap()).abs() < 1e-12);
        }
        assert_eq!(result["history"].as_array().unwrap().len(), 40);
        for (actual, expected) in result["history"].as_array().unwrap().iter().zip(case["history"].as_array().unwrap()) {
            assert!((actual["val_loss"].as_f64().unwrap() - expected["val_loss"].as_f64().unwrap()).abs() < 1e-12);
        }
    }
}

#[test]
fn training_learns_and_is_reproducible() {
    let fast = train(0.1, 40).unwrap();
    assert!(fast.val_loss < 0.02);
    assert!(fast.val_loss < train(0.005, 40).unwrap().val_loss);
    assert!((fast.weight - 2.).abs() < 0.2);
    assert!((fast.bias - 1.).abs() < 0.1);
    assert_eq!(train(0.02, 10).unwrap(), train(0.02, 10).unwrap());
}

#[test]
fn invalid_training_inputs_fail() {
    for rate in [0., -0.1, f64::NAN, f64::INFINITY, 10.] {
        assert!(train(rate, 40).is_err());
    }
    assert!(train(0.1, 0).is_err());
}

fn fixture() -> (Value, Vec<Value>) {
    let job = json!({"metadata":{"name":"example"},"spec":{"numTrials":2,"objectives":[{"metric":"val_loss","direction":"Minimize"}]},"status":{"conditions":[{"type":"Complete","status":"True"}],"result":{"trainJobName":"trial-b"}}});
    let trials = [("trial-a","0.005","0.8"),("trial-b","0.1","0.003")].into_iter().map(|(name,rate,loss)| json!({"metadata":{"name":name,"creationTimestamp":"2026-10-24T10:00:00Z"},"spec":{"trainer":{"env":[{"name":"KUBEFLOW_TRAINER_OPT_learning_rate","value":rate}]}},"status":{"conditions":[{"type":"Complete","status":"True"}],"trainerStatus":{"metrics":[{"name":"val_loss","value":loss}]}}})).collect();
    (job, trials)
}

#[test]
fn report_verifies_measured_winner_and_both_directions() {
    let (mut job, trials) = fixture();
    assert_eq!(summarize(&job, &trials).unwrap()["best"]["name"], "trial-b");
    job["spec"]["objectives"][0]["direction"] = json!("Maximize");
    job["status"]["result"]["trainJobName"] = json!("trial-a");
    assert_eq!(summarize(&job, &trials).unwrap()["best"]["name"], "trial-a");
}

#[test]
fn report_rejects_incomplete_nonfinite_and_wrong_winner() {
    let (mut job, mut trials) = fixture();
    assert!(summarize(&job, &trials[..1]).is_err());
    job["status"]["result"]["trainJobName"] = json!("trial-a");
    assert!(summarize(&job, &trials).is_err());
    job["status"]["result"]["trainJobName"] = json!("trial-b");
    trials[0]["status"]["trainerStatus"]["metrics"][0]["value"] = json!("NaN");
    assert!(summarize(&job, &trials).is_err());
    let (mut job, trials) = fixture();
    job["status"]["conditions"] = json!([]);
    assert!(summarize(&job, &trials).is_err());
}

#[test]
fn report_ties_use_creation_time_then_name_and_reject_duplicates() {
    let (mut job, mut trials) = fixture();
    trials[0]["status"]["trainerStatus"]["metrics"][0]["value"] = json!("0.003");
    job["status"]["result"]["trainJobName"] = json!("trial-a");
    assert!(summarize(&job, &trials).is_ok());
    trials[0]["metadata"]["creationTimestamp"] = json!("2026-10-24T11:00:00Z");
    job["status"]["result"]["trainJobName"] = json!("trial-b");
    assert!(summarize(&job, &trials).is_ok());
    trials[1] = trials[0].clone();
    assert!(summarize(&job, &trials).is_err());
}

#[test]
fn report_rejects_invalid_direction_missing_metrics_and_failed_trials() {
    let (mut job, mut trials) = fixture();
    job["spec"]["objectives"][0]["direction"] = json!("Unknown");
    assert!(summarize(&job, &trials).is_err());
    let (job, _) = fixture();
    trials[0]["status"]["trainerStatus"]["metrics"] = json!([]);
    assert!(summarize(&job, &trials).is_err());
    let (job, mut trials) = fixture();
    trials[0]["status"]["conditions"] = json!([{"type":"Failed","status":"True"}]);
    assert!(summarize(&job, &trials).is_err());
}
