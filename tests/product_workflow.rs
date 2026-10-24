use rubixkube::{cluster::Api, config, experiment};
use serde_json::{json, Value};
use std::{cell::RefCell, time::Duration};
struct Fake {
    job: Value,
    trials: Value,
    pods: Value,
    calls: RefCell<Vec<String>>,
}
impl Api for Fake {
    fn execute(&self, args: &[&str], _: Option<&Value>, _: Duration) -> rubixkube::Result<String> {
        self.calls.borrow_mut().push(args.join(" "));
        Ok(if args.starts_with(&["get", "optimizationjob"]) {
            self.job.to_string()
        } else if args.starts_with(&["get", "trainjobs"]) {
            self.trials.to_string()
        } else if args.starts_with(&["get", "pods"]) {
            self.pods.to_string()
        } else if args.first() == Some(&"logs") {
            String::from("{}\n")
        } else {
            json!({"items":[]}).to_string()
        })
    }
}
fn fake() -> Fake {
    let mut c = config::example(false);
    c.trials = 1;
    c.objective.metric = "score".into();
    let mut job = config::manifest(&c, "custom").unwrap();
    job["metadata"]["uid"] = json!("parent");
    job["status"] = json!({"conditions":[{"type":"Complete","status":"True"}],"result":{"trainJobName":"custom-trial-a"}});
    Fake {
        job,
        trials: json!({"items":[{"metadata":{"name":"custom-trial-a","namespace":"optimizationjob-demo","creationTimestamp":"2026-10-24T12:00:00Z","ownerReferences":[{"kind":"OptimizationJob","uid":"parent"}]},"spec":{"trainer":{"env":[{"name":"KUBEFLOW_TRAINER_OPT_learning_rate","value":"0.1"}]}},"status":{"conditions":[{"type":"Complete","status":"True"}],"trainerStatus":{"metrics":[{"name":"score","value":"-0.3"}]}}}]}),
        pods: json!({"items":[{"metadata":{"name":"custom-trial-a-node-0-0-abcde"},"spec":{"containers":[{"name":"node","image":"example/model"}]},"status":{"phase":"Succeeded"}}]}),
        calls: RefCell::new(Vec::new()),
    }
}
#[test]
fn resume_collects_generic_results_without_resubmitting() {
    let api = fake();
    let dir = tempfile::tempdir().unwrap();
    let report = experiment::watch(&api, "custom", Duration::from_secs(1), dir.path()).unwrap();
    assert_eq!(report["best"]["value"], -0.3);
    assert!(dir.path().join("pods.json").exists());
    assert!(dir.path().join("report.json").exists());
    assert!(!api.calls.borrow().iter().any(|c| c.starts_with("create")));
}
#[test]
fn failure_is_saved_without_claiming_complete_results() {
    let mut api = fake();
    api.job["status"]["conditions"] = json!([{"type":"Failed","status":"True"}]);
    let dir = tempfile::tempdir().unwrap();
    assert!(experiment::watch(&api, "custom", Duration::from_secs(1), dir.path()).is_err());
    assert!(dir.path().join("failed-job.json").exists());
    assert!(!dir.path().join("report.json").exists());
}
#[test]
fn cancellation_cannot_delete_a_foreign_resource() {
    let mut api = fake();
    api.job["metadata"]["labels"] = json!({});
    assert!(experiment::cancel(&api, "custom", false).is_err());
    assert!(!api.calls.borrow().iter().any(|c| c.starts_with("delete")));
}

#[test]
fn retained_shared_data_can_be_purged_after_the_job_was_cancelled() {
    struct Retained {
        deleted: RefCell<Vec<String>>,
    }
    impl Api for Retained {
        fn execute(
            &self,
            args: &[&str],
            _: Option<&Value>,
            _: Duration,
        ) -> rubixkube::Result<String> {
            if args[0] == "get" && args[1] == "pvc" {
                return Ok(json!({"metadata":{"name":"custom-data","labels":{"app.kubernetes.io/managed-by":"rubixkube","rubixkube.dev/run":"custom"}}}).to_string());
            }
            if args[0] == "delete" {
                self.deleted.borrow_mut().push(args.join(" "));
            }
            Ok(Value::Null.to_string())
        }
    }
    let api = Retained {
        deleted: RefCell::new(Vec::new()),
    };
    assert!(experiment::cancel(&api, "custom", true).is_ok());
    assert!(api
        .deleted
        .borrow()
        .iter()
        .any(|c| c.starts_with("delete pvc custom-data")));
}

#[test]
fn cancelling_a_nonshared_run_never_touches_a_same_named_shared_runtime() {
    struct CrossNamespace {
        job: Value,
        calls: RefCell<Vec<String>>,
    }
    impl Api for CrossNamespace {
        fn execute(
            &self,
            args: &[&str],
            _: Option<&Value>,
            _: Duration,
        ) -> rubixkube::Result<String> {
            self.calls.borrow_mut().push(args.join(" "));
            Ok(if args.starts_with(&["get","optimizationjob"]){self.job.clone()}else if args.starts_with(&["get","clustertrainingruntime"]){json!({"metadata":{"name":"custom-runtime","labels":{"app.kubernetes.io/managed-by":"rubixkube","rubixkube.dev/run":"custom"},"annotations":{"rubixkube.dev/config":"foreign namespace configuration"}}})}else{Value::Null}.to_string())
        }
    }
    let api = CrossNamespace {
        job: fake().job,
        calls: RefCell::new(Vec::new()),
    };
    experiment::cancel(&api, "custom", false).unwrap();
    assert!(!api
        .calls
        .borrow()
        .iter()
        .any(|c| c.contains("clustertrainingruntime")));
}
