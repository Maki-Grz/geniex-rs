# GenieX Rust Bindings - Windows ARM64 Guide

This document provides a comprehensive guide to building, testing, and using the standalone GenieX Rust bindings (`geniex` crate) on **Windows ARM64** (`aarch64-pc-windows-msvc`).

---

## 1. Prerequisites & System Requirements

To build and run the Rust bindings on Windows ARM64, the following tools are required:

1. **Rust Toolchain**:
   - Target: `aarch64-pc-windows-msvc`
   - Edition: Rust 2021 (e.g., `rustc 1.97+`)
2. **LLVM / Clang**:
   - Required by `bindgen` to parse the C ABI header `vendor/include/geniex.h` (or from `CARGO_GENIEX_INCLUDE_DIR`).
3. **C/C++ Compiler & Build System**:
   - Visual Studio Build Tools / MSVC ARM64 + CMake (3.16+) + Ninja (for building the native C SDK if compiling from source).
4. **GenieX Native C SDK**:
   - `geniex.lib` and `geniex.dll` (and plugin DLLs such as `geniex_plugin_llama_cpp.dll`).

---

## 2. Step 1: Configuring Environment Variables

The Rust build script [`build.rs`](../build.rs) locates the native C library using flexible environment variables and fallback paths:
- `CARGO_GENIEX_LIB_DIR`: Directory containing `geniex.lib` and `geniex.dll` (e.g., `C:\path\to\GenieX\sdk\pkg-geniex\lib`).
- `CARGO_GENIEX_INCLUDE_DIR` (Optional): Custom path to `geniex.h` if not using the bundled `vendor/include/geniex.h`.

In PowerShell:
```powershell
# Specify path to geniex.lib / geniex.dll
$env:CARGO_GENIEX_LIB_DIR="C:\path\to\GenieX\sdk\pkg-geniex\lib"

# (Optional) Add DLL locations to system PATH if running custom binaries outside cargo
$env:PATH="$env:PATH;$env:CARGO_GENIEX_LIB_DIR;$env:CARGO_GENIEX_LIB_DIR\llama_cpp"
```

---

## 3. Step 2: Compiling and Running Tests

From the root of `geniex-rs`:
```powershell
cargo build

cargo test -- --nocapture
```

### Test Suite Coverage
The integration test suite validates:
- `test_version`: Verifies GenieX SDK version retrieval.
- `test_config_defaults`: Ensures default configuration values (`ModelConfig`, `SamplerConfig`, `GenerationConfig`).
- `test_chat_message`: Validates chat message data structures.
- `test_error`: Verifies C error code mapping to `GeniexError`.
- `test_init_and_plugins`: Tests SDK initialization (`geniex_init`), dynamic scanning/loading of `llama_cpp`, and teardown (`geniex_deinit`).
- `test_resolve_device`: Validates compute-unit resolution (`geniex_resolve_device`).

---

## 4. Step 3: Running the Example

```powershell
# Run basic status check
cargo run -p geniex-rust-example

# Run LLM text generation with model
cargo run -p geniex-rust-example -- path/to/model.gguf
```

---

## 5. Troubleshooting on Windows ARM64

| Error | Root Cause | Solution |
|---|---|---|
| `LINK : fatal error LNK1181: cannot open input file 'geniex.lib'` | Cargo cannot locate the C import library `geniex.lib`. | Ensure `CARGO_GENIEX_LIB_DIR` is set to the directory containing `geniex.lib`. |
| `Unable to generate bindings` during `cargo build` | `bindgen` cannot find Clang/LLVM installations. | Install LLVM (ARM64 build) and ensure `clang.exe` is in your `%PATH%`. |
| Error `0xc0000135` when executing binary | Runtime loader cannot find `geniex.dll` or plugin DLLs. | Add the directory containing `geniex.dll` and `llama_cpp` to system `%PATH%`. |
