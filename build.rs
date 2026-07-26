use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let manifest_path = PathBuf::from(&manifest_dir);

    // 1. Locate C header file (geniex.h)
    println!("cargo:rerun-if-env-changed=CARGO_GENIEX_INCLUDE_DIR");
    let header_path = if let Ok(include_dir) = env::var("CARGO_GENIEX_INCLUDE_DIR") {
        let p = PathBuf::from(&include_dir);
        if p.is_file() {
            p
        } else {
            p.join("geniex.h")
        }
    } else {
        let candidates = vec![
            manifest_path.join("vendor/include/geniex.h"),
            manifest_path.join("vendor/geniex.h"),
            manifest_path.join("include/geniex.h"),
            manifest_path.join("../../sdk/include/geniex.h"),
            manifest_path.join("../GenieX/sdk/include/geniex.h"),
        ];

        candidates
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| {
                panic!(
                    "\n\n================================================================================\n\
                    ERROR: Could not locate required C header file 'geniex.h'.\n\n\
                    To resolve this:\n\
                    1. Place 'geniex.h' in the 'vendor/include/' directory of geniex-rs, OR\n\
                    2. Set the environment variable CARGO_GENIEX_INCLUDE_DIR to point to 'geniex.h'.\n\n\
                    Upstream GenieX C SDK: https://github.com/qualcomm/GenieX/releases\n\
                    ================================================================================\n\n"
                )
            })
    };

    println!("cargo:rerun-if-changed={}", header_path.display());

    // 2. Locate native C library search paths
    println!("cargo:rerun-if-env-changed=CARGO_GENIEX_LIB_DIR");
    let mut search_dirs = Vec::new();

    if let Ok(lib_dir) = env::var("CARGO_GENIEX_LIB_DIR") {
        let p = PathBuf::from(&lib_dir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
            search_dirs.push(p);
        } else {
            println!(
                "cargo:warning=CARGO_GENIEX_LIB_DIR specified path '{}' does not exist.",
                p.display()
            );
        }
    } else {
        let candidates = vec![
            manifest_path.join("vendor/lib"),
            manifest_path.join("vendor"),
            manifest_path.join("lib"),
            manifest_path.join("../../sdk/pkg-geniex/lib"),
            manifest_path.join("../../sdk/build/src"),
            manifest_path.join("../GenieX/sdk/pkg-geniex/lib"),
            manifest_path.join("../GenieX/sdk/build/src"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                println!("cargo:rustc-link-search=native={}", candidate.display());
                search_dirs.push(candidate);
            }
        }
    }

    if search_dirs.is_empty() {
        println!(
            "cargo:warning=================================================================================\n\
             cargo:warning=WARNING: Native GenieX SDK library directory was not found in standard paths.\n\
             cargo:warning=Please download the GenieX C SDK from https://github.com/qualcomm/GenieX/releases\n\
             cargo:warning=and set CARGO_GENIEX_LIB_DIR to the folder containing geniex.lib / libgeniex.so.\n\
             cargo:warning================================================================================="
        );
    }

    println!("cargo:rustc-link-lib=geniex");

    // 3. Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(
            header_path
                .to_str()
                .expect("Header path must be valid UTF-8"),
        )
        .generate_comments(false)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings via bindgen. Ensure LLVM/Clang is installed.");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // 4. On Windows, copy geniex.dll to target_dir so binaries run without DLL PATH errors
    if cfg!(target_os = "windows") {
        if let Some(target_dir) = out_path.ancestors().nth(3) {
            for search_dir in &search_dirs {
                let dll_src = search_dir.join("geniex.dll");
                if dll_src.exists() {
                    let _ = std::fs::copy(&dll_src, target_dir.join("geniex.dll"));
                }
                // Copy llama_cpp plugin DLLs if available
                let llama_plugin_dir = search_dir.join("llama_cpp");
                if llama_plugin_dir.exists() {
                    let plugin_target = target_dir.join("llama_cpp");
                    let _ = std::fs::create_dir_all(&plugin_target);
                    if let Ok(entries) = std::fs::read_dir(&llama_plugin_dir) {
                        for entry in entries.flatten() {
                            if entry.path().extension().is_some_and(|ext| ext == "dll") {
                                let _ = std::fs::copy(
                                    entry.path(),
                                    plugin_target.join(entry.file_name()),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
