use geniex::*;
use std::env;
use std::path::{Path, PathBuf};

fn find_gguf_in_dir(dir: &Path) -> Option<PathBuf> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
                return Some(path);
            } else if path.is_dir() {
                if let Some(found) = find_gguf_in_dir(&path) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn main() -> Result<()> {
    println!("=== GenieX Rust Binding Functional Example ===");

    init()?;
    println!("[+] SDK Initialized successfully.");
    println!("[+] GenieX Version: {}", version());

    let plugins = get_plugin_list()?;
    println!("[+] Installed plugins: {:?}", plugins);

    if plugins.contains(&"llama_cpp".to_string()) {
        let resolve_input = ResolveDeviceInput {
            plugin_id: "llama_cpp".to_string(),
            model_name: Some("example.gguf".to_string()),
            mode: Some("cpu".to_string()),
            ngl_default: -1,
        };
        let dev_output = resolve_device(&resolve_input)?;
        println!(
            "[+] Device resolution: device_id={:?}, ngl={}",
            dev_output.device_id, dev_output.ngl
        );
        if let Some(warn) = dev_output.warning {
            println!("[!] Device warning: {}", warn);
        }
    }

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let raw_path = PathBuf::from(&args[1]);
        let model_path = if raw_path.is_dir() {
            println!(
                "\n[i] Provided argument is a directory: {}",
                raw_path.display()
            );
            println!("[i] Searching for .gguf model files within directory...");
            if let Some(discovered) = find_gguf_in_dir(&raw_path) {
                println!("[+] Discovered GGUF model: {}", discovered.display());
                discovered
            } else {
                println!("[!] No .gguf file found inside directory!");
                raw_path
            }
        } else {
            raw_path
        };

        let path_str = model_path.to_str().unwrap_or(&args[1]);
        println!("\n[+] Loading LLM model from: {}", path_str);

        let config = ModelConfig::default();
        let mut llm = Llm::create(path_str, "llama_cpp", &config, None, None)?;

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello! Describe what GenieX is in one sentence.".to_string(),
        }];

        println!("[+] Applying chat template...");
        let prompt = llm.apply_chat_template(&messages, None, false, true)?;

        println!("[+] Generating response...");
        let (response_text, _profile) =
            llm.generate::<fn(&str) -> bool>(Some(&prompt), None, None, None)?;

        println!("\n--- Generated Response ---");
        println!("{}", response_text);
        println!("--------------------------");
    } else {
        println!("\n[i] Note: Pass a GGUF model file path as an argument to run LLM inference:");
        println!("    cargo run -- <path_to_model.gguf>");
    }

    deinit()?;
    println!("\n[+] SDK De-initialized successfully.");

    Ok(())
}
