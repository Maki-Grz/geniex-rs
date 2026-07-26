# GenieX Rust Binding Example

This directory contains a functional example demonstrating how to use the GenieX Rust bindings (`geniex` crate).

## Prerequisites

1. Build and install the GenieX C SDK (`geniex.dll` / `geniex.lib` / plugins):
   ```powershell
   # Point CARGO_GENIEX_LIB_DIR to the directory containing geniex.lib and geniex.dll
   $env:CARGO_GENIEX_LIB_DIR="C:\path\to\GenieX\sdk\pkg-geniex\lib"
   ```

2. (Optional) If running outside Cargo build automation, add DLL paths to your environment (Windows PowerShell):
   ```powershell
   $env:PATH="$env:PATH;$env:CARGO_GENIEX_LIB_DIR;$env:CARGO_GENIEX_LIB_DIR\llama_cpp"
   ```

## Running the Example

### 1. Basic SDK Status & Discovery Check

Run without parameters to test runtime initialization, plugin discovery, and device resolution:

From the workspace root (`geniex-rs`):
```bash
cargo run -p geniex-rust-example
```

Or from within the `examples/` directory:
```bash
cargo run
```

### 2. LLM Model Inference

Pass the path to a GGUF model file to run text generation:

```bash
cargo run -p geniex-rust-example -- path/to/model.gguf
```
