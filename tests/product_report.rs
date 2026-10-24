use rubixkube::experiment_report::summarize;
use serde_json::{json, Value};
fn fixture() -> (Value, Vec<Value>) {
    let job = json!({"metadata":{"name":"test-fit","namespace":"tests","uid":"parent-uid"},"spec":{"numTrials":2,"objectives":[{"metric":"score","direction":"Maximize"}]},"status":{"conditions":[{"type":"Complete","status":"True"}],"result":{"trainJobName":"trial-b"}}});
    let trials=[("trial-a","-2"),("trial-b","-1")].into_iter().map(|(name,value)|json!({"metadata":{"name":name,"namespace":"tests","creationTimestamp":"2026-10-24T12:00:00Z","ownerReferences":[{"kind":"OptimizationJob","uid":"parent-uid"}]},"spec":{"trainer":{"env":[{"name":"KUBEFLOW_TRAINER_OPT_depth","value":"3"},{"name":"KUBEFLOW_TRAINER_OPT_optimizer","value":"adam"}]}},"status":{"conditions":[{"type":"Complete","status":"True"}],"trainerStatus":{"metrics":[{"name":"score","value":value}]}}})).collect();
    (job, trials)
}
#[test]
fn generic_metric_and_all_parameter_assignments_are_verified() {
    let (j, t) = fixture();
    let r = summarize(&j, &t).unwrap();
    assert_eq!(r["best"]["name"], "trial-b");
    assert_eq!(r["best"]["value"], -1.0);
    assert_eq!(r["best"]["parameters"]["optimizer"], "adam");
    assert_eq!(r["schema_version"], 2);
}
#[test]
fn zero_ties_match_upstream_and_wrong_winner_is_rejected() {
    let (mut j, mut t) = fixture();
    t[0]["status"]["trainerStatus"]["metrics"][0]["value"] = json!("-0");
    t[1]["status"]["trainerStatus"]["metrics"][0]["value"] = json!("0");
    j["status"]["result"]["trainJobName"] = json!("trial-a");
    assert!(summarize(&j, &t).is_ok());
    j["status"]["result"]["trainJobName"] = json!("trial-b");
    assert!(summarize(&j, &t).is_err());
}
#[test]
fn foreign_duplicate_incomplete_or_nonfinite_trials_fail() {
    let (j, mut t) = fixture();
    t[0]["metadata"]["ownerReferences"][0]["uid"] = json!("foreign");
    assert!(summarize(&j, &t).is_err());
    let (j, mut t) = fixture();
    t[1] = t[0].clone();
    assert!(summarize(&j, &t).is_err());
    let (j, t) = fixture();
    assert!(summarize(&j, &t[..1]).is_err());
    let (j, mut t) = fixture();
    t[0]["status"]["trainerStatus"]["metrics"][0]["value"] = json!("NaN");
    assert!(summarize(&j, &t).is_err());
}

#[test]
fn cuda_evidence_must_match_the_named_objective_and_full_history() {
    let result = rubixkube::training::train(0.1, 10).unwrap();
    let loss = result.val_loss;
    let mut event = serde_json::to_value(result).unwrap();
    event["event"] = json!("training_complete");
    event["execution"] = json!("GPU, Rust CUDA; device training verified");
    event["objective_metric"] = json!("validation_error");
    event["gpu"] = json!({"device_name":"NVIDIA GPU","compute_capability":[12,0],"kernel_launches":1,"cuda_elapsed_ms":1.0,"cpu_reference_max_abs_error":0.0});
    assert!(rubixkube::experiment::cuda_evidence(&event, "validation_error", loss).is_ok());
    assert!(rubixkube::experiment::cuda_evidence(&event, "accuracy", loss).is_err());
    event["history"][0]["val_loss"] = json!("bad");
    assert!(rubixkube::experiment::cuda_evidence(&event, "validation_error", loss).is_err());
}
