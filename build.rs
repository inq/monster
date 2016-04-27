fn main() {
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cudnn");
}
