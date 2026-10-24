use rubixkube::config::{manifest, ExperimentConfig};
use serde_json::{json, Value};
fn data() -> Value {
    json!({"version":1,"namespace":"optimizationjob-demo","runtime":{"name":"optimizationjob-demo-cpu"},"workload":{"image":"example/model:1"},"objective":{"metric":"accuracy","direction":"Maximize"},"trials":4,"parallelism":2,"parameters":{"rate":{"type":"log_uniform","min":0.001,"max":0.1},"depth":{"type":"uniform","min":1.0,"max":8.0,"integer":true},"optimizer":{"type":"categorical","choices":["adam","sgd"]}}})
}
#[test]
fn compiles_typed_multi_parameter_custom_workload_manifest() {
    let config: ExperimentConfig = serde_json::from_value(data()).unwrap();
    config.validate().unwrap();
    let job = manifest(&config, "custom-fit").unwrap();
    assert_eq!(job["apiVersion"], "trainer.kubeflow.org/v1alpha1");
    assert_eq!(job["spec"]["objectives"][0]["metric"], "accuracy");
    assert_eq!(
        job["spec"]["trainJobTemplate"]["spec"]["trainer"]["image"],
        "example/model:1"
    );
    assert_eq!(
        job["metadata"]["labels"]["app.kubernetes.io/managed-by"],
        "rubixkube"
    );
    assert_eq!(job["spec"]["parameters"].as_array().unwrap().len(), 3);
    assert_eq!(
        job["spec"]["parameters"][0]["searchSpace"]["uniform"]["type"],
        "Int"
    );
}
#[test]
fn rejects_invalid_budgets_names_bounds_and_reserved_environment() {
    for (path, value) in [
        ("trials", json!(0)),
        ("parallelism", json!(5)),
        ("namespace", json!("--all")),
    ] {
        let mut d = data();
        d[path] = value;
        assert!(serde_json::from_value::<ExperimentConfig>(d)
            .unwrap()
            .validate()
            .is_err());
    }
    let mut d = data();
    d["parameters"]["rate"]["min"] = json!(-1.0);
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
    let mut d = data();
    d["parameters"]["depth"]["min"] = json!(1.5);
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
    let mut d = data();
    d["workload"]["env"] = json!({"KUBEFLOW_TRAINER_SERVER_URL":"https://attacker.invalid"});
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
    let c: ExperimentConfig = serde_json::from_value(data()).unwrap();
    assert!(manifest(&c, &"a".repeat(40)).is_err());
}
#[test]
fn gpu_and_shared_data_are_explicit_and_secrets_use_references() {
    let mut d = data();
    d["resources"] = json!({"gpus":1});
    d["require_cuda_proof"] = json!(true);
    d["workload"]["secret_env"] =
        json!({"MODEL_TOKEN":{"name":"training-credentials","key":"token"}});
    d["shared_data"] = json!({"storage":"1Gi","mount_path":"/dataset","initializer":{"image":"example/fetch:1","command":["fetch"]}});
    let c: ExperimentConfig = serde_json::from_value(d).unwrap();
    c.validate().unwrap();
    let j = manifest(&c, "custom-fit").unwrap();
    assert_eq!(
        j["spec"]["trainJobTemplate"]["spec"]["trainer"]["resourcesPerNode"]["limits"]
            ["nvidia.com/gpu"],
        "1"
    );
    assert_eq!(
        j["spec"]["trainJobTemplate"]["spec"]["runtimeRef"]["name"],
        "custom-fit-runtime"
    );
    assert_eq!(
        j["spec"]["trainJobTemplate"]["spec"]["trainer"]["env"][0]["valueFrom"]["secretKeyRef"]
            ["name"],
        "training-credentials"
    );
}
#[test]
fn unknown_config_fields_are_errors() {
    let mut d = data();
    d["paralellism"] = json!(2);
    assert!(serde_json::from_value::<ExperimentConfig>(d).is_err());
}

#[test]
fn rejects_search_spaces_the_pinned_provider_cannot_honor() {
    for max in [1e-6, 1e-11] {
        let mut d = data();
        d["parameters"]["rate"] = json!({"type":"log_uniform","min":1e-12,"max":max});
        assert!(serde_json::from_value::<ExperimentConfig>(d)
            .unwrap()
            .validate()
            .is_err());
    }
    let mut d = data();
    d["parameters"]["optimizer"] = json!({"type":"categorical","choices":[""]});
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
    let mut d = data();
    d["resources"] = json!({"cpu":"-1"});
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
}

#[test]
fn rejects_empty_category_before_the_controller_cannot_persist_its_assignment() {
    let mut d = data();
    d["parameters"]["optimizer"] = json!({"type":"categorical","choices":[""]});
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
}
#[test]
fn compute_quantities_must_be_positive() {
    let mut d = data();
    d["resources"] = json!({"cpu":"-1"});
    assert!(serde_json::from_value::<ExperimentConfig>(d)
        .unwrap()
        .validate()
        .is_err());
}
