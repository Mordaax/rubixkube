use rubixkube::metrics::{deliver, MetricsConfig};
use serde_json::Value;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};
use tempfile::TempDir;

struct Server {
    dir: TempDir,
    config: MetricsConfig,
    seen: mpsc::Receiver<String>,
    worker: thread::JoinHandle<()>,
}
fn server(statuses: Vec<u16>, hostname: &str) -> Server {
    let dir = tempfile::tempdir().unwrap();
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    std::fs::write(dir.path().join("ca.pem"), cert.cert.pem()).unwrap();
    std::fs::write(dir.path().join("token"), "projected-secret\n").unwrap();
    let tls = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(
        vec![cert.cert.der().clone()],
        rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
            cert.key_pair.serialize_der(),
        )),
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    let (send, seen) = mpsc::channel();
    let worker = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(4);
        let mut responses = statuses.into_iter();
        while Instant::now() < deadline {
            match listener.accept() {
                Ok((socket, _)) => {
                    socket
                        .set_read_timeout(Some(Duration::from_secs(1)))
                        .unwrap();
                    socket
                        .set_write_timeout(Some(Duration::from_secs(1)))
                        .unwrap();
                    let conn = rustls::ServerConnection::new(Arc::new(tls.clone())).unwrap();
                    let mut stream = rustls::StreamOwned::new(conn, socket);
                    let mut bytes = Vec::new();
                    let mut buf = [0u8; 4096];
                    loop {
                        let n = match stream.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        bytes.extend_from_slice(&buf[..n]);
                        if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&bytes[..end]);
                            let length = head
                                .lines()
                                .find_map(|line| {
                                    line.to_lowercase()
                                        .strip_prefix("content-length:")
                                        .and_then(|s| s.trim().parse::<usize>().ok())
                                })
                                .unwrap_or(0);
                            if bytes.len() >= end + 4 + length {
                                break;
                            }
                        }
                    }
                    if bytes.is_empty() {
                        return;
                    }
                    send.send(String::from_utf8(bytes).unwrap()).unwrap();
                    let status = responses.next().unwrap_or(500);
                    let response = format!("HTTP/1.1 {status} Test\r\nContent-Length: 0\r\nConnection: close\r\nLocation: https://localhost:{port}/redirected\r\n\r\n");
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                    if responses.len() == 0 {
                        return;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10))
                }
                Err(_) => return,
            }
        }
    });
    let config = MetricsConfig {
        url: format!("https://{hostname}:{port}/status"),
        ca_cert: dir.path().join("ca.pem"),
        token_path: dir.path().join("token"),
        attempts: 1,
        retry_delay: Duration::ZERO,
        request_timeout: Duration::from_secs(2),
    };
    Server {
        dir,
        config,
        seen,
        worker,
    }
}

#[test]
fn pushes_exact_trainer_payload_with_token_and_verified_tls() {
    let s = server(vec![200], "localhost");
    assert_eq!(deliver(&s.config, 0.007).unwrap(), 1);
    let request = s.seen.recv().unwrap();
    assert!(request
        .to_lowercase()
        .contains("authorization: bearer projected-secret"));
    assert!(request.starts_with("POST /status HTTP/1.1"));
    let payload: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["trainerStatus"]["progressPercentage"], 100);
    assert_eq!(payload["trainerStatus"]["metrics"][0]["value"], "0.007");
    assert!(chrono::DateTime::parse_from_rfc3339(
        payload["trainerStatus"]["lastUpdatedTime"]
            .as_str()
            .unwrap()
    )
    .is_ok());
    s.worker.join().unwrap();
}

#[test]
fn retries_rejection_but_requires_exact_200_acknowledgement() {
    let mut s = server(vec![503, 200], "localhost");
    s.config.attempts = 2;
    assert_eq!(deliver(&s.config, 0.1).unwrap(), 2);
    assert_eq!(s.seen.iter().count(), 2);
    s.worker.join().unwrap();
    let s = server(vec![201], "localhost");
    assert!(deliver(&s.config, 0.1).is_err());
    s.worker.join().unwrap();
}

#[test]
fn refuses_redirects_and_exhausts_retry_budget() {
    for status in [302, 401, 503] {
        let s = server(vec![status], "localhost");
        assert!(deliver(&s.config, 0.1).is_err());
        assert_eq!(s.seen.iter().count(), 1);
        s.worker.join().unwrap();
    }
}

#[test]
fn rejects_wrong_hostname_untrusted_ca_and_missing_credentials() {
    let s = server(vec![200], "127.0.0.1");
    assert!(deliver(&s.config, 0.1).is_err());
    s.worker.join().unwrap();
    let s = server(vec![200], "localhost");
    let other = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    std::fs::write(&s.config.ca_cert, other.cert.pem()).unwrap();
    assert!(deliver(&s.config, 0.1).is_err());
    s.worker.join().unwrap();
    let s = server(vec![200], "localhost");
    std::fs::remove_file(&s.config.token_path).unwrap();
    assert!(deliver(&s.config, 0.1).is_err());
    drop(s.dir);
}

#[test]
fn validates_configuration_before_network_access() {
    let mut s = server(vec![200], "localhost");
    assert!(deliver(&s.config, f64::NAN).is_err());
    s.config.attempts = 0;
    assert!(deliver(&s.config, 0.1).is_err());
    s.config.attempts = 1;
    s.config.url = "http://localhost/status".into();
    assert!(deliver(&s.config, 0.1).is_err());
    s.config.url = "https://localhost/status".into();
    std::fs::write(&s.config.ca_cert, "invalid certificate").unwrap();
    assert!(deliver(&s.config, 0.1).is_err());
}

#[test]
fn custom_metric_uses_the_same_authenticated_protocol() {
    let s = server(vec![200], "localhost");
    assert_eq!(
        rubixkube::metrics::deliver_metric(&s.config, "accuracy", 0.94).unwrap(),
        1
    );
    let request = s.seen.recv().unwrap();
    let payload: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(payload["trainerStatus"]["metrics"][0]["name"], "accuracy");
    assert_eq!(payload["trainerStatus"]["metrics"][0]["value"], "0.94");
    s.worker.join().unwrap();
}
