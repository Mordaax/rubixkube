use rubixkube::{orchestrate::validate_worker_mode, training::train};
use serde_json::json;

#[test]
fn gpu_mode_rejects_cpu_execution_and_missing_device_proof() {
    let cpu = train(0.1, 40).unwrap();
    let mut event = serde_json::to_value(&cpu).unwrap();
    event["event"] = json!("training_complete");
    let row = json!({"learning_rate":0.1,"val_loss":cpu.val_loss});
    let logs = |event: &serde_json::Value| {
        format!(
            "{event}\n{}",
            json!({"event":"metric_delivered","attempt":1})
        )
    };
    assert!(validate_worker_mode(&row, &logs(&event), true).is_err());
    event["execution"] = json!("GPU, Rust CUDA; device training verified");
    assert!(validate_worker_mode(&row, &logs(&event), true).is_err());
    event["gpu"] = json!({"device_name":"NVIDIA GeForce RTX 5060 Ti","compute_capability":[12,0],"kernel_launches":1,"cuda_elapsed_ms":0.1,"cpu_reference_max_abs_error":0.0});
    assert!(validate_worker_mode(&row, &logs(&event), true).is_ok());
    event["gpu"]["kernel_launches"] = json!(0);
    assert!(validate_worker_mode(&row, &logs(&event), true).is_err());
    event["gpu"]["kernel_launches"] = json!(1);
    event["gpu"]["cpu_reference_max_abs_error"] = json!(1.0);
    assert!(validate_worker_mode(&row, &logs(&event), true).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires explicitly supplied CUDA GPU and NVRTC libraries"]
fn real_cuda_training_matches_cpu_for_every_candidate() {
    for rate in [0.005, 0.02, 0.05, 0.1] {
        let gpu = rubixkube::gpu::train_cuda(rate, 40).unwrap();
        let cpu = train(rate, 40).unwrap();
        assert!((gpu.val_loss - cpu.val_loss).abs() < 1e-10);
        let proof = gpu.gpu.unwrap();
        assert_eq!(proof.kernel_launches, 1);
        assert!(proof.cuda_elapsed_ms > 0.);
        assert!(proof.cpu_reference_max_abs_error < 1e-10);
        println!(
            "GPU {} rate={} loss={} kernel_ms={} parity_error={}",
            proof.device_name,
            rate,
            gpu.val_loss,
            proof.cuda_elapsed_ms,
            proof.cpu_reference_max_abs_error
        );
    }
}
