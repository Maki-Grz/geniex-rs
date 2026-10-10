use geniex::*;
use std::env;
use std::io::Write;
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

    // 1. Configure logging before initialization (reduces noise and speeds up QNN backend)
    set_log_level(LogLevel::Warn)?;

    // 2. Initialize native runtime
    init()?;
    println!("[+] SDK Initialized successfully.");
    println!("[+] GenieX Version: {}", version());

    let qairt_path = get_qairt_runtime_path();
    if !qairt_path.is_empty() {
        println!("[+] QAIRT runtime path: {}", qairt_path);
    }

    // 3. Plugin & device discovery
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

        // 4. Configure model options including HTP power mode
        let config = ModelConfig {
            power_mode: PowerMode::Burst,
            ..Default::default()
        };
        println!("[+] HTP Power Mode: {}", config.power_mode);

        let mut llm = Llm::create(path_str, "llama_cpp", &config, None, None)?;

        let messages = vec![ChatMessage::new(
            "user",
            "Hello! Describe what GenieX is in one sentence.",
        )];

        println!("[+] Applying chat template...");
        let prompt = llm.apply_chat_template(&messages, None, false, true)?;

        // 5. Generate with streaming callback
        print!("\n[+] Streaming response:\n> ");
        let (_response_text, profile) = llm.generate(
            Some(&prompt),
            None,
            None,
            Some(|token: &str| {
                print!("{}", token);
                let _ = std::io::stdout().flush();
                true
            }),
        )?;

        println!("\n\n--- Performance Telemetry ---");
        println!("    Prompt tokens:      {}", profile.prompt_tokens);
        println!("    Generated tokens:   {}", profile.generated_tokens);
        println!(
            "    Time to first token: {:.2} ms",
            (profile.ttft as f64) / 1000.0
        );
        println!(
            "    Prefill speed:      {:.2} tokens/s",
            profile.prefill_speed
        );
        println!(
            "    Decoding speed:     {:.2} tokens/s",
            profile.decoding_speed
        );

        // 6. Demonstrate single-token candidate scoring (llm.score)
        println!("\n[+] Candidate next-token scoring via llm.score():");
        let score_prompt = "The official language of France is";
        let candidates = [" French", " English", " Spanish", " German"];
        match llm.score(score_prompt, &candidates) {
            Ok(score_output) => {
                println!("    Prompt: \"{}\"", score_prompt);
                for (cand, logit) in candidates.iter().zip(score_output.logits.iter()) {
                    println!("    - Candidate {:10} => logit: {:+.4}", cand, logit);
                }
            }
            Err(e) => {
                println!("    [i] Score not supported by backend: {}", e);
            }
        }
    } else {
        println!("\n[i] Note: Pass a GGUF model file path as an argument to run LLM inference:");
        println!("    cargo run -- <path_to_model.gguf>");
    }

    deinit()?;
    println!("\n[+] SDK De-initialized successfully.");

    Ok(())
}
