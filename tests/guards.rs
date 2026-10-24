use optimizationjob::process::{checked, verify_checkout};
use std::{
    process::Command,
    time::{Duration, Instant},
};
fn git(path: &std::path::Path, args: &[&str]) -> String {
    checked(
        Command::new("git").arg("-C").arg(path).args(args),
        None,
        Duration::from_secs(5),
    )
    .unwrap()
}
fn repo() -> (tempfile::TempDir, String) {
    let root = tempfile::tempdir().unwrap();
    git(root.path(), &["init", "-q"]);
    git(
        root.path(),
        &[
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
            "-q",
        ],
    );
    let sha = git(root.path(), &["rev-parse", "HEAD"]).trim().to_owned();
    (root, sha)
}
#[test]
fn clean_checkout_is_accepted() {
    let (root, sha) = repo();
    verify_checkout(root.path(), &sha).unwrap();
}
#[test]
fn mismatch_dirty_checkout_and_corrupt_index_are_rejected() {
    let (root, sha) = repo();
    assert!(verify_checkout(root.path(), &"0".repeat(40)).is_err());
    std::fs::write(root.path().join("untracked"), "user work").unwrap();
    assert!(verify_checkout(root.path(), &sha).is_err());
    std::fs::remove_file(root.path().join("untracked")).unwrap();
    std::fs::write(root.path().join(".git/index"), b"not a git index").unwrap();
    assert!(verify_checkout(root.path(), &sha).is_err());
}
#[test]
fn failed_commands_are_not_success() {
    assert!(checked(
        Command::new("git").arg("invalid-command-name"),
        None,
        Duration::from_secs(5)
    )
    .is_err());
}
#[cfg(unix)]
#[test]
fn subprocess_deadline_is_enforced() {
    let start = Instant::now();
    assert!(checked(
        Command::new("sleep").arg("5"),
        None,
        Duration::from_millis(50)
    )
    .is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}
#[cfg(unix)]
#[test]
fn drains_large_output_without_pipe_deadlock() {
    let result = checked(
        Command::new("dd").args(["if=/dev/zero", "bs=65536", "count=32"]),
        None,
        Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(result.len(), 2 * 1024 * 1024);
}
