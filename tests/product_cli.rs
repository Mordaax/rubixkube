use std::process::Command;
#[test]
fn installable_binary_has_product_commands_and_noninteractive_init() {
    let bin = env!("CARGO_BIN_EXE_rubixkube");
    let help = Command::new(bin).arg("--help").output().unwrap();
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stdout);
    for command in [
        "init",
        "validate",
        "submit",
        "status",
        "list",
        "watch",
        "results",
        "logs",
        "cancel",
        "doctor",
        "report",
        "completions",
    ] {
        assert!(help.contains(command), "missing {command}");
    }
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("experiment.json");
    assert!(Command::new(bin)
        .arg("init")
        .arg("--file")
        .arg(&file)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(bin)
        .arg("validate")
        .arg("--config")
        .arg(&file)
        .status()
        .unwrap()
        .success());
    assert!(
        !Command::new(bin)
            .arg("init")
            .arg("--file")
            .arg(&file)
            .status()
            .unwrap()
            .success(),
        "init must not overwrite user config"
    );
}
