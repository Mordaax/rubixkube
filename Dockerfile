FROM rust@sha256:cfb3f582db21e4b4168bffa96397db118d288f1c55026cf016911e147476184e AS build
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml LICENSE README.md ./
COPY src ./src
COPY assets ./assets
COPY demo/fixtures ./demo/fixtures
RUN cargo build --locked --release --features cuda --bin rubixkube

# CPU worker and metric reporter. Cluster lifecycle commands need host kubectl.
FROM gcr.io/distroless/cc-debian12@sha256:9dac0a79194e45a7da0158a9c6da57b217585af0786db3845d1f0ec1a0dd182f AS cpu
COPY --from=build /build/target/release/rubixkube /app/rubixkube
COPY LICENSE assets/FONT-LICENSE.txt /licenses/
USER 65532:65532
ENTRYPOINT ["/app/rubixkube"]

# CUDA worker; supply the host NVIDIA driver/device through your GPU runtime.
FROM nvidia/cuda@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9 AS cuda
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64
COPY --from=build /build/target/release/rubixkube /app/rubixkube
COPY LICENSE assets/FONT-LICENSE.txt /licenses/
USER 65532:65532
ENTRYPOINT ["/app/rubixkube"]
