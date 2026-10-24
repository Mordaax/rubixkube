use rubixkube::{
    cluster::Api,
    config,
    shared::{initializer, pvc, runtime},
};
use serde_json::{json, Value};
use std::{cell::RefCell, time::Duration};
struct Fake {
    calls: RefCell<Vec<Vec<String>>>,
}
impl Api for Fake {
    fn execute(&self, args: &[&str], _: Option<&Value>, _: Duration) -> rubixkube::Result<String> {
        self.calls
            .borrow_mut()
            .push(args.iter().map(|s| s.to_string()).collect());
        Ok("{}".into())
    }
}
#[test]
fn shared_initializer_and_readonly_trial_mount_use_one_claim() {
    let mut c = config::example(false);
    c.shared_data=Some(serde_json::from_value(json!({"storage":"1Gi","mount_path":"/dataset","initializer":{"image":"busybox:1.37.0","command":["sh","-c","echo x,y > /dataset/data.csv"]}})).unwrap());
    let claim = pvc(&c, "fit-one").unwrap();
    assert_eq!(claim["metadata"]["name"], "fit-one-data");
    let init = initializer(&c, "fit-one").unwrap();
    assert_eq!(
        init["spec"]["template"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"],
        "fit-one-data"
    );
    assert_eq!(init["spec"]["backoffLimit"], 0);
    let source = json!({"apiVersion":"trainer.kubeflow.org/v1alpha1","kind":"ClusterTrainingRuntime","metadata":{"name":"base","uid":"old","resourceVersion":"3"},"spec":{"template":{"spec":{"replicatedJobs":[{"name":"node","template":{"spec":{"template":{"spec":{"containers":[{"name":"node","image":"old"}]}}}}}]}}}});
    let derived = runtime(&c, "fit-one", &source).unwrap();
    assert_eq!(derived["metadata"]["name"], "fit-one-runtime");
    assert!(derived["metadata"]["uid"].is_null());
    assert_eq!(derived.pointer("/spec/template/spec/replicatedJobs/0/template/spec/template/spec/containers/0/volumeMounts/0/readOnly"),Some(&json!(true)));
    assert_eq!(source["metadata"]["name"], "base");
}
#[test]
fn api_json_preserves_failure_instead_of_using_empty_success() {
    let fake = Fake {
        calls: RefCell::new(Vec::new()),
    };
    let value = fake
        .json(
            &["get", "nodes", "-o", "json"],
            None,
            Duration::from_secs(1),
        )
        .unwrap();
    assert_eq!(value, json!({}));
    assert_eq!(fake.calls.borrow().len(), 1);
}

#[test]
fn successful_shared_initialization_is_reused_and_failed_initialization_blocks_trials() {
    use std::collections::BTreeMap;
    struct Memory {
        objects: RefCell<BTreeMap<(String, String), Value>>,
        created: RefCell<Vec<String>>,
        fail: bool,
    }
    impl Api for Memory {
        fn execute(
            &self,
            args: &[&str],
            payload: Option<&Value>,
            _: Duration,
        ) -> rubixkube::Result<String> {
            if args[0] == "get" {
                return Ok(self
                    .objects
                    .borrow()
                    .get(&(args[1].to_ascii_lowercase(), args[2].into()))
                    .cloned()
                    .unwrap_or(Value::Null)
                    .to_string());
            }
            let mut object = payload.unwrap().clone();
            let kind = object["kind"].as_str().unwrap().to_ascii_lowercase();
            let name = object["metadata"]["name"].as_str().unwrap().to_owned();
            if kind == "job" {
                object["status"] = json!({"conditions":[{"type":if self.fail{"Failed"}else{"Complete"},"status":"True"}]});
            }
            self.created.borrow_mut().push(kind.clone());
            self.objects
                .borrow_mut()
                .insert((kind, name), object.clone());
            Ok(object.to_string())
        }
    }
    let mut c = config::example(false);
    c.shared_data=Some(serde_json::from_value(json!({"storage":"1Gi","mount_path":"/dataset","initializer":{"image":"busybox:1.37.0"}})).unwrap());
    let source = json!({"apiVersion":"trainer.kubeflow.org/v1alpha1","kind":"ClusterTrainingRuntime","metadata":{"name":c.runtime.name},"spec":{"template":{"spec":{"replicatedJobs":[{"name":"node","template":{"spec":{"template":{"spec":{"containers":[{"name":"node"}]}}}}}]}}}});
    for fail in [false, true] {
        let api = Memory {
            objects: RefCell::new(BTreeMap::from([(
                ("clustertrainingruntime".into(), c.runtime.name.clone()),
                source.clone(),
            )])),
            created: RefCell::new(Vec::new()),
            fail,
        };
        let result = rubixkube::shared::prepare(&api, &c, "one-init");
        if fail {
            assert!(result.is_err());
            assert!(!api
                .created
                .borrow()
                .contains(&"clustertrainingruntime".to_owned()));
        } else {
            result.unwrap();
            rubixkube::shared::prepare(&api, &c, "one-init").unwrap();
            assert_eq!(
                api.created
                    .borrow()
                    .iter()
                    .filter(|kind| kind.as_str() == "job")
                    .count(),
                1
            );
        }
    }
}
