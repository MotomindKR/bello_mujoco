fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/reference.proto");
    println!("cargo:rerun-if-changed=proto/operator.proto");
    tonic_prost_build::configure().compile_protos(&["proto/reference.proto"], &["proto"])?;
    tonic_prost_build::configure().compile_protos(&["proto/operator.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=../../src/bello_mujoco/hardware/hand.proto");
    tonic_prost_build::configure().compile_protos(
        &["../../src/bello_mujoco/hardware/hand.proto"],
        &["../../src/bello_mujoco/hardware"],
    )?;
    Ok(())
}
