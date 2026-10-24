use rubixkube::{dataset, training::train_on};
#[test]
fn custom_csv_trains_on_real_input_instead_of_the_fixture() {
    let rows: Vec<[f64; 2]> = (0..256)
        .map(|i| {
            let x = (i % 64) as f64 / 32. - 1.;
            [x, 3. * x + 0.5]
        })
        .collect();
    let model = train_on(0.1, 100, &rows).unwrap();
    assert!((model.weight - 3.).abs() < 0.01);
    assert!((model.bias - 0.5).abs() < 0.01);
    assert!(model.val_loss < 0.0001);
}
#[test]
fn csv_parser_validates_shape_and_finite_values() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("data.csv");
    std::fs::write(&p, "x,y\n0,1\n1,3\n2,5\n3,7\n").unwrap();
    assert_eq!(dataset::read_csv(&p).unwrap().len(), 4);
    for bad in ["x,y\n0,NaN\n1,2\n2,3\n3,4\n", "x,y\n0,1,2\n", "x,y\n0,1\n"] {
        std::fs::write(&p, bad).unwrap();
        assert!(dataset::read_csv(&p).is_err());
    }
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires actual CUDA device and NVRTC"]
fn cuda_trains_dynamic_dataset_and_epochs() {
    let rows: Vec<[f64; 2]> = (0..256)
        .map(|i| {
            let x = (i % 64) as f64 / 32. - 1.;
            [x, 3. * x + 0.5]
        })
        .collect();
    let model = rubixkube::gpu::train_cuda_on(0.1, 100, &rows).unwrap();
    assert!((model.weight - 3.).abs() < 0.01);
    assert_eq!(model.history.len(), 100);
    assert!(model.gpu.unwrap().cpu_reference_max_abs_error < 1e-10);
}
