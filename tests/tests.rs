use geniex::*;

#[test]
fn test_version() {
    let _v = version();
}

#[test]
fn test_error() {
    let err = GeniexError::CommonInvalidInput;
    assert_ne!(err.message(), "");
    let res = GeniexError::check(0);
    assert!(res.is_ok());
}

#[test]
fn test_config_defaults() {
    let model_cfg = ModelConfig::default();
    assert_eq!(model_cfg.n_ctx, 0);

    let sampler_cfg = SamplerConfig::default();
    assert_eq!(sampler_cfg.seed, -1);

    let gen_cfg = GenerationConfig::default();
    assert_eq!(gen_cfg.max_tokens, 0);
}

#[test]
fn test_chat_message() {
    let msg = ChatMessage::new("user", "hello");
    assert_eq!(msg.role, "user");
    assert_eq!(msg.content, "hello");

    let tc = ToolCall::new(Some("call_1"), "get_weather", r#"{"location":"Paris"}"#);
    assert_eq!(tc.name, "get_weather");

    let assistant_msg = ChatMessage::assistant_with_tool_calls("", vec![tc]);
    assert_eq!(assistant_msg.role, "assistant");
    assert_eq!(assistant_msg.tool_calls.len(), 1);

    let tool_msg = ChatMessage::tool_response("22°C", "call_1", "get_weather");
    assert_eq!(tool_msg.role, "tool");
    assert_eq!(tool_msg.tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(tool_msg.tool_name.as_deref(), Some("get_weather"));
}

#[test]
fn test_init_and_plugins() -> Result<()> {
    init()?;
    let ver = version();
    assert!(!ver.is_empty(), "Version string should not be empty");

    let plugins = get_plugin_list()?;
    println!("Loaded plugins: {:?}", plugins);

    deinit()?;
    Ok(())
}

#[test]
fn test_resolve_device() -> Result<()> {
    let input = ResolveDeviceInput {
        plugin_id: "llama_cpp".to_string(),
        model_name: Some("test.gguf".to_string()),
        mode: Some("cpu".to_string()),
        ngl_default: -1,
    };
    let output = resolve_device(&input)?;
    assert_eq!(output.ngl, 0); // cpu forces ngl = 0
    Ok(())
}

#[test]
fn test_vlm_media_conversions() {
    use std::path::PathBuf;

    let txt = VlmMedia::Text("hello".to_string());
    let img = VlmMedia::Image(PathBuf::from("image.jpg"));
    let aud = VlmMedia::Audio(PathBuf::from("audio.wav"));

    let content_txt = VlmContent::from(txt);
    assert_eq!(content_txt.r#type, "text");
    assert_eq!(content_txt.text, "hello");

    let content_img = VlmContent::from(img);
    assert_eq!(content_img.r#type, "image");
    assert_eq!(content_img.text, "image.jpg");

    let content_aud = VlmContent::from(aud);
    assert_eq!(content_aud.r#type, "audio");
    assert_eq!(content_aud.text, "audio.wav");
}

#[test]
fn test_chat_session_history() {
    // We can test session state and manipulation without model execution
    let dummy_history = [ChatMessage::new("user", "test message")];

    assert_eq!(dummy_history.len(), 1);
    assert_eq!(dummy_history[0].role, "user");
    assert_eq!(dummy_history[0].content, "test message");
}

#[test]
fn test_power_mode() {
    assert_eq!(ModelConfig::default().power_mode, PowerMode::Burst);

    // Resolving standard aliases
    assert_eq!(
        PowerMode::resolve("low_power_saver").unwrap(),
        PowerMode::LowPowerSaver
    );
    assert_eq!(
        PowerMode::resolve("power_saver").unwrap(),
        PowerMode::PowerSaver
    );
    assert_eq!(
        PowerMode::resolve("high_power_saver").unwrap(),
        PowerMode::HighPowerSaver
    );
    assert_eq!(
        PowerMode::resolve("low_balanced").unwrap(),
        PowerMode::LowBalanced
    );
    assert_eq!(PowerMode::resolve("balanced").unwrap(), PowerMode::Balanced);
    assert_eq!(
        PowerMode::resolve("high_performance").unwrap(),
        PowerMode::HighPerformance
    );
    assert_eq!(
        PowerMode::resolve("sustained_high_performance").unwrap(),
        PowerMode::SustainedHighPerformance
    );
    assert_eq!(PowerMode::resolve("burst").unwrap(), PowerMode::Burst);

    // Resolving defaults and case insensitivity
    assert_eq!(PowerMode::resolve("").unwrap(), PowerMode::Burst);
    assert_eq!(PowerMode::resolve("default").unwrap(), PowerMode::Burst);
    assert_eq!(PowerMode::resolve("  BURST  ").unwrap(), PowerMode::Burst);
    assert_eq!(
        PowerMode::resolve("Low_Balanced").unwrap(),
        PowerMode::LowBalanced
    );

    // Error on invalid
    assert!(PowerMode::resolve("invalid_mode").is_err());

    // FromStr & Display
    let parsed: PowerMode = "high_performance".parse().unwrap();
    assert_eq!(parsed, PowerMode::HighPerformance);
    assert_eq!(parsed.to_string(), "high_performance");
}

#[test]
fn test_vlm_prefix_reuse_failed_error() {
    let err = GeniexError::from_i32(-201202);
    assert_eq!(err, GeniexError::VlmPrefixReuseFailed);
    assert!(!err.message().is_empty());
}

#[test]
fn test_log_level_and_qairt_runtime_path() {
    // Calling before init should succeed
    let res = set_log_level(LogLevel::Info);
    assert!(res.is_ok());

    let initial = get_qairt_runtime_path();
    assert_eq!(initial, "");

    let res = set_qairt_runtime_path(Some("custom/qairt/lib"));
    assert!(res.is_ok());
    assert_eq!(get_qairt_runtime_path(), "custom/qairt/lib");

    // Resetting back to bundled
    let res = set_qairt_runtime_path(None);
    assert!(res.is_ok());
    assert_eq!(get_qairt_runtime_path(), "");
}

#[test]
fn test_vlm_create_options_and_score_output() {
    let cfg = ModelConfig::default();
    let options = VlmCreateOptions {
        model_path: "model.gguf",
        plugin_id: "llama_cpp",
        config: &cfg,
        mmproj_path: Some("mmproj.gguf"),
        tokenizer_path: None,
        device_id: Some("npu"),
        vit_device_id: Some("gpu"),
    };
    assert_eq!(options.vit_device_id, Some("gpu"));

    let score = ScoreOutput {
        logits: vec![1.5, -0.5],
        input_tokens: 12,
    };
    assert_eq!(score.logits.len(), 2);
    assert_eq!(score.input_tokens, 12);
}
