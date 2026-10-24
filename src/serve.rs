//! A loopback-only static report server with canonical path containment.
use crate::{invalid, Result};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Component, Path, PathBuf},
    time::Duration,
};

/// Resolve a URL path without permitting traversal or symlink escape.
pub fn resolve_file(root: &Path, request: &str) -> Result<PathBuf> {
    let root = root.canonicalize()?;
    let path = request.split('?').next().unwrap_or("/");
    let decoded = percent_encoding::percent_decode_str(path)
        .decode_utf8()
        .map_err(|_| invalid("Invalid UTF-8 path"))?;
    let relative = decoded
        .strip_prefix('/')
        .ok_or_else(|| invalid("Expected absolute URL path"))?;
    let relative = if relative.is_empty() {
        "index.html"
    } else {
        relative
    };
    let path = Path::new(relative);
    if path
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid("Invalid report path"));
    }
    let target = root.join(path).canonicalize()?;
    if !target.starts_with(&root) || !target.is_file() {
        return Err(invalid("Report path escapes directory or is not a file"));
    }
    Ok(target)
}
fn respond(mut socket: TcpStream, root: &Path) -> Result<()> {
    socket.set_read_timeout(Some(Duration::from_secs(3)))?;
    socket.set_write_timeout(Some(Duration::from_secs(3)))?;
    let mut request = Vec::new();
    let mut chunk = [0u8; 1024];
    while request.len() < 8192 && !request.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = socket.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        request.extend_from_slice(&chunk[..n]);
    }
    let request = std::str::from_utf8(&request).map_err(|_| invalid("Invalid HTTP request"))?;
    let line: Vec<&str> = request
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .collect();
    if line.len() != 3 || !matches!(line[0], "GET" | "HEAD") {
        socket.write_all(
            b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?;
        return Ok(());
    }
    match resolve_file(root, line[1]) {
        Ok(path) => {
            let body = fs::read(&path)?;
            let mime = match path.extension().and_then(|s| s.to_str()) {
                Some("html") => "text/html; charset=utf-8",
                Some("json") => "application/json",
                _ => "text/plain; charset=utf-8",
            };
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",body.len())?;
            if line[0] == "GET" {
                socket.write_all(&body)?;
            }
        }
        Err(_) => socket.write_all(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )?,
    }
    Ok(())
}
/// Serve saved reports only on IPv4 loopback until the process is stopped.
pub fn serve(root: &Path, port: u16) -> Result<()> {
    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err(invalid("Report directory does not exist"));
    }
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!(
        "Serving {} at http://127.0.0.1:{}/",
        root.display(),
        listener.local_addr()?.port()
    );
    for socket in listener.incoming() {
        if let Err(error) = respond(socket?, &root) {
            eprintln!("Report request rejected: {error}");
        }
    }
    Ok(())
}
