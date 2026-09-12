fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/reference.proto");
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(&["proto/reference.proto"], &["proto"])?;
    Ok(())
}
