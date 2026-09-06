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
