# geniex-rs — Rust Bindings for Qualcomm GenieX

> ⚠️ **Disclaimer / Community Notice**: `geniex-rs` is an open-source, community-maintained Rust binding for Qualcomm's GenieX C API. It is not an officially maintained product of Qualcomm Technologies, Inc. For official documentation and C/C++ runtime releases, visit [qualcomm/GenieX on GitHub](https://github.com/qualcomm/GenieX).

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qualcomm/GenieX/main/GenieX-Logo-Hor-1-White.png" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/qualcomm/GenieX/main/GenieX-Logo-Hor-1-Black.png" />
  <img src="https://raw.githubusercontent.com/qualcomm/GenieX/main/GenieX-Logo-Hor-1-Black.png" width="420" alt="Qualcomm AI Hub GenieX" />
</picture>

### Safe, high-level Rust bindings to run frontier LLMs & VLMs locally on Qualcomm devices

[![Status: Developer Preview](https://img.shields.io/badge/status-developer%20preview-FF6C2C?style=flat-square)](#)
[![Docs](https://img.shields.io/badge/docs-geniex.aihub.qualcomm.com-2A2AEA?style=flat-square&logo=readthedocs&logoColor=white)](https://geniex.aihub.qualcomm.com)
[![Upstream Release](https://img.shields.io/github/v/release/qualcomm/GenieX?style=flat-square&color=2A2AEA&label=upstream%20release)](https://github.com/qualcomm/GenieX/releases)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)](LICENSE)
[![Slack](https://img.shields.io/badge/Slack-join%20the%20community-4A154B?style=flat-square&logo=slack&logoColor=white)](https://aihub.qualcomm.com/community/slack)

[**Upstream Docs**](https://geniex.aihub.qualcomm.com) · [**Quickstart**](#-quickstart) · [**Windows ARM64**](#-windows-arm64-support-snapdragon-x-plus--x-elite) · [**Usage Example**](#-usage-example) · [**Troubleshooting**](#-troubleshooting--faq) · [**Credits & Disclaimer**](#-credits--disclaimer)

</div>

---

## 🌟 Overview

GenieX is an **on-device Gen AI inference runtime for Qualcomm devices**. `geniex-rs` provides safe, idiomatic, high-level Rust wrappers for the **GenieX** C API (`geniex.h`). Bring almost any GGUF model from Hugging Face — or a pre-compiled bundle from [Qualcomm AI Hub](https://aihub.qualcomm.com/models/) — and run it locally in Rust on the **Hexagon NPU, Adreno GPU, or CPU** in a few lines of code.

<div align="center">
  <img src="https://raw.githubusercontent.com/qualcomm/GenieX/main/docs/Mintlify-image/geniex_arch_v2.png" width="820" alt="GenieX architecture: Rust bindings, CLI, Python, Java, Docker, and OpenAI-compatible Serve interfaces sit on a single GenieX SDK, which dispatches to the llama.cpp runtime (GGML over CPU / GPU / Hexagon HTP kernels) or the Qualcomm AI Engine Direct runtime on the NPU — across Windows, Android, and Linux." />
</div>

---

## 🖥️ Supported Platforms

GenieX runs natively on **Qualcomm Snapdragon** chipsets. The `geniex-rs` bindings support the following target architectures:

| Target Platform | Target Triple | Target Devices | Runtimes Supported |
| --- | --- | --- | --- |
| 🪟 **Windows ARM64** *(Compute)* | `aarch64-pc-windows-msvc` | Snapdragon X Plus · X Elite | NPU (`qairt`), GPU (`llama_cpp`), CPU |
| 🐧 **Linux ARM64** *(IoT / Edge)* | `aarch64-unknown-linux-gnu` | Dragonwing QCS9075 · RB5 | NPU (`qairt`), GPU, CPU |
| 💻 **x86_64 Host / Simulation** | `x86_64-pc-windows-msvc` / `x86_64-unknown-linux-gnu` | Dev Workstations / CI | CPU (`llama_cpp` fallback) |

> 💡 **No Qualcomm device on hand?** Spin up a remote session on [Qualcomm Device Cloud](https://qdc.qualcomm.com/).

---

## ⚙️ Prerequisites

To build and use `geniex-rs`, ensure your environment has:

1. **Rust Toolchain**: 2021 edition (Rust 1.70 or higher).
2. **LLVM / Clang**: Required by `bindgen` to parse `geniex.h`:
   - **Windows**: `winget install LLVM.LLVM` or `choco install llvm`
   - **Linux**: `sudo apt install libclang-dev clang`
3. **GenieX C SDK Binaries**: Download the native C SDK (`geniex.lib`/`geniex.dll` on Windows or `libgeniex.so` on Linux) from the [Qualcomm GenieX Releases](https://github.com/qualcomm/GenieX/releases).

---

## 🚀 Quickstart

### 1. Configure the Native SDK Location

Set the `CARGO_GENIEX_LIB_DIR` environment variable to point to your extracted GenieX C SDK libraries:

**PowerShell (Windows):**
```powershell
$env:CARGO_GENIEX_LIB_DIR="C:\path\to\GenieX\sdk\pkg-geniex\lib"
```

**Bash (Linux):**
```bash
export CARGO_GENIEX_LIB_DIR="/path/to/GenieX/sdk/pkg-geniex/lib"
```

*(Alternatively, copy `geniex.lib`/`geniex.dll` into `vendor/lib/` inside your project directory).*

### 2. Build and Test

```bash
# Compile the crate and build scripts
cargo build

# Run unit and integration tests
cargo test --workspace
```

### 3. Run the Included Example

Use `cargo run` to execute the bundled example crate (`geniex-rust-example`):

```bash
# Check runtime initialization & plugin discovery
cargo run -p geniex-rust-example

# Run LLM prompt inference using a GGUF model file (e.g. Qwen, Llama, Gemma)
cargo run -p geniex-rust-example -- path/to/model.gguf
```

---

## 💻 Usage Example

Here is a complete, safe Rust program using `geniex-rs` to load an LLM and stream response tokens:

```rust
use geniex::*;

fn main() -> Result<()> {
    // 1. Initialize the GenieX C runtime
    init()?;
    println!("GenieX SDK version: {}", version());

    // 2. Discover available plugins (e.g. llama_cpp, qairt)
    let plugins = get_plugin_list()?;
    println!("Available plugins: {:?}", plugins);

    // 3. Create an LLM instance with default configurations
    let config = ModelConfig::default();
    let mut llm = Llm::create(
        "models/gemma-2b-it.gguf",
        "llama_cpp",
        &config,
        None, // SamplerConfig
        None, // GenerationConfig
        None, // DialogConfig
    )?;

    // 4. Format prompt using the model's native chat template
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Explain quantum computing in one sentence.".to_string(),
    }];
    let prompt = llm.apply_chat_template(&messages, None, false, true)?;

    // 5. Generate text with a streaming token callback
    println!("--- Prompt response ---");
    let (response, profile) = llm.generate::<fn(&str) -> bool>(
        Some(&prompt),
        None,
        None,
        Some(|token| {
            print!("{}", token);
            true // Continue token generation
        }),
    )?;

    println!("\n\nTokens generated: {}", profile.token_count);
    println!("Speed: {:.2} tokens/sec", profile.decoding_speed);

    // 6. Clean up the GenieX runtime
    deinit()?;
    Ok(())
}
```

---

## 🪟 Windows ARM64 Support (Snapdragon X Plus / X Elite)

`geniex-rs` includes first-class support for **Windows ARM64** (`aarch64-pc-windows-msvc`) on Snapdragon X Plus and Snapdragon X Elite devices.

### Automated DLL Resolution
On Windows, native binaries depend on `geniex.dll` and dynamic plugin libraries (such as `llama_cpp.dll`). In standard Cargo workflows, missing runtime DLLs cause immediate execution crashes with code `0xc0000135` (`STATUS_DLL_NOT_FOUND`).

`geniex-rs` eliminates this friction:
- **Automatic Target Copying**: Our `build.rs` automatically scans your SDK paths (or `CARGO_GENIEX_LIB_DIR`) and copies `geniex.dll` together with all required plugin DLL directories directly into Cargo's output build directory (`target/debug` or `target/release`).
- **Zero-Setup `cargo run`**: You can run `cargo run` and `cargo test` directly out of the box without manually managing DLL PATH environment variables.

For detailed building and troubleshooting instructions on Snapdragon laptops, read the [Windows ARM64 Troubleshooting Guide](docs/rust_windows_arm64.md).

---

## ❓ Troubleshooting & FAQ

| Error / Issue | Root Cause | Resolution |
|---|---|---|
| `Unable to generate bindings` during `cargo build` | `bindgen` cannot locate Clang/LLVM. | Install LLVM (`winget install LLVM.LLVM` or `apt install libclang-dev`) and ensure `clang` is in your `PATH`. |
| `LINK : fatal error LNK1181: cannot open input file 'geniex.lib'` | Cargo cannot locate the native import library. | Set `$env:CARGO_GENIEX_LIB_DIR` to the directory containing `geniex.lib` / `geniex.dll`. |
| `Could not locate required C header file 'geniex.h'` | `geniex.h` is missing from include lookup paths. | Ensure `vendor/include/geniex.h` exists or set `CARGO_GENIEX_INCLUDE_DIR`. |

---

## 📦 Features

- **Safe RAII Memory Management**: `Llm` and `Vlm` structs handle underlying native C handles automatically via standard `Drop` semantics.
- **Multimodal Support (VLM)**: Safe API for Vision-Language Models processing image inputs alongside text prompts.
- **KV Cache Management**: Load and save KV cache states across sessions.
- **Dynamic Device Alias Resolution**: Select hardware acceleration backends (`cpu`, `gpu`, `npu`, or plugin defaults) at runtime.

---

## 📄 License

This repository is dual-licensed under the **BSD 3-Clause License** to strictly match the upstream Qualcomm GenieX project. See [LICENSE](LICENSE) for full details.

---

## 👏 Credits & Disclaimer

- **Upstream Repository**: [Qualcomm GenieX (C/C++ Engine)](https://github.com/qualcomm/GenieX)
- **Copyright**: Copyright (c) 2024-2026, Qualcomm Technologies, Inc. and/or its subsidiaries.
- **Maintenance**: This Rust binding is maintained by the open-source community. Contributions and PRs are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and DCO sign-off requirements.
