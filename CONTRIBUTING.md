# Contributing to `geniex-rs`

Thank you for your interest in contributing to **`geniex-rs`**, the community-maintained Rust bindings for Qualcomm's GenieX C API! We welcome contributions from developers, researchers, and open-source enthusiasts.

---

## 📜 Developer Certificate of Origin (DCO) & Commit Sign-Off

To align with upstream Qualcomm GenieX requirements, **all contributions to this project must include a Developer Certificate of Origin (DCO) sign-off**.

By including the `Signed-off-by` line in your commit message, you certify the following statement:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


DEVELOPER CERTIFICATE OF ORIGIN 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b), or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to Sign Off Your Commits

You can automatically add the sign-off line to your commit using the `-s` or `--signoff` flag with `git commit`:

```bash
git commit -s -m "feat: add support for streaming KV cache state"
```

Your commit message must end with a line formatted as:

```text
Signed-off-by: Random J Developer <random@developer.example.org>
```

---

## 🛠️ Local Development & Testing Workflow

### Prerequisites

1. **Rust Toolchain**: 2021 Edition (Rust 1.70+ recommended).
2. **LLVM / Clang**: Required by `bindgen` to parse `vendor/include/geniex.h`. Ensure `clang` is installed and available in your `PATH`.
3. **GenieX Native Libraries**: Set `CARGO_GENIEX_LIB_DIR` to the location of `geniex.lib` / `geniex.dll` (Windows) or `libgeniex.so` (Linux).

```bash
# Example (PowerShell)
$env:CARGO_GENIEX_LIB_DIR="C:\path\to\GenieX\sdk\pkg-geniex\lib"

# Example (Bash)
export CARGO_GENIEX_LIB_DIR="/path/to/GenieX/sdk/pkg-geniex/lib"
```

### Formatting & Linting

We enforce strict formatting and linting rules in CI. Before submitting a PR, verify your changes pass `cargo fmt` and `cargo clippy`:

```bash
# Check code formatting
cargo fmt --check

# Run Clippy linter with strict warning checks
cargo clippy -- -D warnings
```

### Running Tests

Run the integration and unit test suite:

```bash
cargo test -- --nocapture
```

---

## 🔀 Submitting a Pull Request

1. **Fork the repository** and create a feature branch (`git checkout -b feat/my-new-feature`).
2. **Write clean, idiomatic Rust code** and ensure tests pass.
3. **Commit your changes with DCO sign-off** (`git commit -s`).
4. **Push to your fork** and submit a Pull Request targeting the `main` branch.
5. Provide a clear explanation of your changes in the PR description.

---

## 💬 Disclaimer & Community

`geniex-rs` is an independent, community-driven Rust binding for Qualcomm GenieX. For upstream C/C++ runtime development, please refer to the official [Qualcomm GenieX Repository](https://github.com/qualcomm/GenieX).
