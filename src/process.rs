//! Bounded external commands and fail-closed source revision checks.
use crate::{invalid, Result};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

/// Captured external process output, including failures for evidence recording.
pub struct ProcessOutput {
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
}
fn read(mut file: File) -> Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Ok(text)
}
/// Run a process with file-backed I/O to avoid pipe deadlocks and bounded wait.
pub fn capture(
    command: &mut Command,
    input: Option<&[u8]>,
    timeout: Duration,
) -> Result<ProcessOutput> {
    let output = tempfile::tempfile()?;
    let errors = tempfile::tempfile()?;
    let stdin = if let Some(bytes) = input {
        let mut file = tempfile::tempfile()?;
        file.write_all(bytes)?;
        file.seek(SeekFrom::Start(0))?;
        Stdio::from(file)
    } else {
        Stdio::null()
    };
    command
        .stdin(stdin)
        .stdout(output.try_clone()?)
        .stderr(errors.try_clone()?);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn()?;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= timeout {
            #[cfg(unix)]
            {
                use nix::{
                    sys::signal::{killpg, Signal},
                    unistd::Pid,
                };
                let _ = killpg(Pid::from_raw(child.id() as i32), Signal::SIGKILL);
            }
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid(format!(
                "Command {:?} exceeded deadline of {} seconds",
                command.get_program(),
                timeout.as_secs_f64()
            )));
        }
        thread::sleep(Duration::from_millis(20));
    };
    Ok(ProcessOutput {
        status,
        stdout: read(output)?,
        stderr: read(errors)?,
    })
}
/// Return stdout only for a successful command; all spawn/status errors propagate.
pub fn checked(command: &mut Command, input: Option<&[u8]>, timeout: Duration) -> Result<String> {
    let output = capture(command, input, timeout)?;
    if !output.status.success() {
        let diagnostic = format!("{}{}", output.stderr, output.stdout);
        return Err(invalid(format!(
            "Command {:?} failed ({}): {}",
            command.get_program(),
            output.status,
            diagnostic.chars().take(4000).collect::<String>()
        )));
    }
    Ok(output.stdout)
}
/// Verify both exact revision and a successfully read clean Git status.
pub fn verify_checkout(checkout: &Path, expected: &str) -> Result<()> {
    let timeout = Duration::from_secs(30);
    let actual = checked(
        Command::new("git")
            .arg("-C")
            .arg(checkout)
            .args(["rev-parse", "HEAD"]),
        None,
        timeout,
    )?;
    if actual.trim() != expected {
        return Err(invalid(format!(
            "{} revision differs from upstream.lock.json",
            checkout.display()
        )));
    }
    // Match the Windows-created checkouts without changing global Git settings.
    let status = checked(
        Command::new("git")
            .args(["-c", "core.autocrlf=true", "-C"])
            .arg(checkout)
            .args(["status", "--porcelain"]),
        None,
        timeout,
    )?;
    if !status.is_empty() {
        return Err(invalid(format!("{} has local changes", checkout.display())));
    }
    Ok(())
}
