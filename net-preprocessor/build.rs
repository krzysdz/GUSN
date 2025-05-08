extern crate flatc_rust; // or just `use flatc_rust;` with Rust 2018 edition.

use std::path::Path;

fn main() {
    println!(
        "cargo:rerun-if-changed=../tflite-micro/tensorflow/compiler/mlir/lite/schema/schema.fbs"
    );
    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new(
            "../tflite-micro/tensorflow/compiler/mlir/lite/schema/schema.fbs",
        )],
        out_dir: Path::new("target/flatbuffers/"),
        ..Default::default()
    })
    .expect("flatc");
}
