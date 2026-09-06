use crate::ffi;
use std::ffi::CString;
use std::os::raw::c_char;

/// Logging severity levels used by the native SDK.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Highly detailed tracing output.
    Trace = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_TRACE,
    /// Debugging information.
    Debug = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_DEBUG,
    /// Informational operational messages.
    Info = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_INFO,
    /// Warning conditions.
    Warn = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_WARN,
    /// Error conditions.
    Error = ffi::geniex_LogLevel_GENIEX_LOG_LEVEL_ERROR,
}

/// Performance and timing telemetry data captured during generation.
#[derive(Debug, Clone, Default)]
pub struct ProfileData {
    /// Time to first token in microseconds.
    pub ttft: i64,
    /// Image/audio encoder time in microseconds; 0 for text-only runs.
    pub media_time: i64,
    /// Prefill time in microseconds (includes media-token prefill, excludes encoder).
    pub prompt_time: i64,
    /// Total decoding time in microseconds.
    pub decode_time: i64,
    /// Number of prompt tokens processed (text + media tokens).
    pub prompt_tokens: i64,
    /// Number of generated tokens produced.
    pub generated_tokens: i64,
    /// Prefill speed in tokens per second.
    pub prefill_speed: f64,
    /// Decoding speed in tokens per second.
    pub decoding_speed: f64,
    /// Speculative decoding: draft tokens generated (0 when disabled).
    pub draft_n_total: i64,
    /// Speculative decoding: draft tokens accepted by the target model.
    pub draft_n_accepted: i64,
    /// Reason generation stopped (e.g., "eos", "length").
    pub stop_reason: String,
}

impl From<&ffi::geniex_ProfileData> for ProfileData {
    fn from(raw: &ffi::geniex_ProfileData) -> Self {
        let stop_reason = if raw.stop_reason.is_null() {
            String::new()
        } else {
            // SAFETY: raw.stop_reason is checked for non-null before constructing CStr slice.
            unsafe {
                std::ffi::CStr::from_ptr(raw.stop_reason)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        Self {
            ttft: raw.ttft,
            media_time: raw.media_time,
            prompt_time: raw.prompt_time,
            decode_time: raw.decode_time,
            prompt_tokens: raw.prompt_tokens,
            generated_tokens: raw.generated_tokens,
            prefill_speed: raw.prefill_speed,
            decoding_speed: raw.decoding_speed,
            draft_n_total: raw.draft_n_total,
            draft_n_accepted: raw.draft_n_accepted,
            stop_reason,
        }
    }
}

/// Sampler configuration options controlling token generation randomness and grammar.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature scaling for logit sampling (0.0 to 2.0).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k token candidate limit.
    pub top_k: i32,
    /// Min-p sampling threshold.
    pub min_p: f32,
    /// Penalty for repeating recent tokens.
    pub repetition_penalty: f32,
    /// Presence penalty.
    pub presence_penalty: f32,
    /// Frequency penalty.
    pub frequency_penalty: f32,
    /// Random seed (-1 for dynamic seed).
    pub seed: i32,
    /// Optional path to GBNF grammar file.
    pub grammar_path: Option<String>,
    /// Optional raw GBNF grammar string.
    pub grammar_string: Option<String>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: -1,
            grammar_path: None,
            grammar_string: None,
        }
    }
}

pub(crate) struct RawSamplerConfig {
    pub raw: ffi::geniex_SamplerConfig,
    _grammar_path: Option<CString>,
    _grammar_string: Option<CString>,
}

fn safe_c_string(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::new("").unwrap())
}

impl SamplerConfig {
    pub(crate) fn to_raw(&self) -> RawSamplerConfig {
        let grammar_path_c = self.grammar_path.as_ref().map(|s| safe_c_string(s));
        let grammar_string_c = self.grammar_string.as_ref().map(|s| safe_c_string(s));

        let raw = ffi::geniex_SamplerConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            grammar_path: grammar_path_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            grammar_string: grammar_string_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
        };

        RawSamplerConfig {
            raw,
            _grammar_path: grammar_path_c,
            _grammar_string: grammar_string_c,
        }
    }
}

/// Generation parameters passed during model inference requests.
#[derive(Debug, Clone, Default)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: i32,
    /// List of stop sequence strings.
    pub stop: Vec<String>,
    /// Optional sampler settings.
    pub sampler_config: Option<SamplerConfig>,
    /// Paths to image inputs for multimodal generation.
    pub image_paths: Vec<String>,
    /// Paths to audio inputs for multimodal generation.
    pub audio_paths: Vec<String>,
    /// Enable sliding window context attention.
    pub sliding_window: bool,
    /// Number of tokens to retain during sliding window shift.
    pub sliding_window_n_keep: i32,
}

pub(crate) struct RawGenerationConfig {
    pub raw: ffi::geniex_GenerationConfig,
    _raw_sampler: Option<Box<RawSamplerConfig>>,
    _stops: Vec<CString>,
    _stop_ptrs: Vec<*const c_char>,
    _image_paths: Vec<CString>,
    _image_path_ptrs: Vec<*const c_char>,
    _audio_paths: Vec<CString>,
    _audio_path_ptrs: Vec<*const c_char>,
}

impl GenerationConfig {
    pub(crate) fn to_raw(&self) -> RawGenerationConfig {
        let raw_sampler = self.sampler_config.as_ref().map(|s| Box::new(s.to_raw()));
        let sampler_ptr = raw_sampler
            .as_ref()
            .map_or(std::ptr::null_mut(), |s| &s.raw as *const _ as *mut _);

        let stops: Vec<CString> = self.stop.iter().map(|s| safe_c_string(s)).collect();
        let stop_ptrs: Vec<*const c_char> = stops.iter().map(|s| s.as_ptr()).collect();

        let image_paths: Vec<CString> = self.image_paths.iter().map(|s| safe_c_string(s)).collect();
        let image_path_ptrs: Vec<*const c_char> = image_paths.iter().map(|s| s.as_ptr()).collect();

        let audio_paths: Vec<CString> = self.audio_paths.iter().map(|s| safe_c_string(s)).collect();
        let audio_path_ptrs: Vec<*const c_char> = audio_paths.iter().map(|s| s.as_ptr()).collect();

        let raw = ffi::geniex_GenerationConfig {
            max_tokens: self.max_tokens,
            stop: if stop_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                stop_ptrs.as_ptr() as *mut *const c_char
            },
            stop_count: stop_ptrs.len() as i32,
            sampler_config: sampler_ptr,
            image_paths: if image_path_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                image_path_ptrs.as_ptr() as *mut *const c_char
            },
            image_count: image_path_ptrs.len() as i32,
            audio_paths: if audio_path_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                audio_path_ptrs.as_ptr() as *mut *const c_char
            },
            audio_count: audio_path_ptrs.len() as i32,
            sliding_window: self.sliding_window,
            sliding_window_n_keep: self.sliding_window_n_keep,
        };

        RawGenerationConfig {
            raw,
            _raw_sampler: raw_sampler,
            _stops: stops,
            _stop_ptrs: stop_ptrs,
            _image_paths: image_paths,
            _image_path_ptrs: image_path_ptrs,
            _audio_paths: audio_paths,
            _audio_path_ptrs: audio_path_ptrs,
        }
    }
}

/// Model initialization options.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Context window size (in tokens).
    pub n_ctx: i32,
    /// Number of CPU threads for inference.
    pub n_threads: i32,
    /// Number of CPU threads for batch processing.
    pub n_threads_batch: i32,
    /// Maximum batch size for prompt processing.
    pub n_batch: i32,
    /// Micro-batch size.
    pub n_ubatch: i32,
    /// Maximum sequence length.
    pub n_seq_max: i32,
    /// Number of layers to offload to GPU/NPU.
    pub n_gpu_layers: i32,
    /// Path to custom Jinja chat template.
    pub chat_template_path: Option<String>,
    /// Raw Jinja chat template content string.
    pub chat_template_content: Option<String>,
    pub spec_type: Option<String>,
    pub spec_draft_model: Option<String>,
    pub spec_n_max: i32,
    pub spec_n_min: i32,
    pub spec_p_min: f32,
}

pub(crate) struct RawModelConfig {
    pub raw: ffi::geniex_ModelConfig,
    _chat_template_path: Option<CString>,
    _chat_template_content: Option<CString>,
    _spec_type: Option<CString>,
    _spec_draft_model: Option<CString>,
}

impl ModelConfig {
    pub(crate) fn to_raw(&self) -> RawModelConfig {
        let chat_template_path_c = self.chat_template_path.as_ref().map(|s| safe_c_string(s));
        let chat_template_content_c = self
            .chat_template_content
            .as_ref()
            .map(|s| safe_c_string(s));
        let spec_type_c = self.spec_type.as_ref().map(|s| safe_c_string(s));
        let spec_draft_model_c = self.spec_draft_model.as_ref().map(|s| safe_c_string(s));

        let raw = ffi::geniex_ModelConfig {
            n_ctx: self.n_ctx,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            n_batch: self.n_batch,
            n_ubatch: self.n_ubatch,
            n_seq_max: self.n_seq_max,
            n_gpu_layers: self.n_gpu_layers,
            chat_template_path: chat_template_path_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            chat_template_content: chat_template_content_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            spec_type: spec_type_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            spec_draft_model: spec_draft_model_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            spec_n_max: self.spec_n_max,
            spec_n_min: self.spec_n_min,
            spec_p_min: self.spec_p_min,
        };

        RawModelConfig {
            raw,
            _chat_template_path: chat_template_path_c,
            _chat_template_content: chat_template_content_c,
            _spec_type: spec_type_c,
            _spec_draft_model: spec_draft_model_c,
        }
    }
}

/// A function call the model issued on a prior "assistant" turn.
///
/// Chat templates need these structurally: many render a tool response only when
/// the preceding assistant message carries `tool_calls`, and match the response
/// back to the call by `id`. Flattening a call into assistant `content` text
/// drops the following "tool" message from the prompt entirely.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// Call id echoed by the matching "tool" message (optional).
    pub id: Option<String>,
    /// Function name.
    pub name: String,
    /// Function arguments as a JSON string.
    pub arguments: String,
}

impl ToolCall {
    /// Creates a new `ToolCall`.
    pub fn new(
        id: Option<impl Into<String>>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.map(Into::into),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// Representation of a single chat message (role and content, plus optional tool calling metadata).
#[derive(Debug, Clone, Default)]
pub struct ChatMessage {
    /// Message sender role (e.g., "user", "assistant", "system", "tool").
    pub role: String,
    /// Message body content.
    pub content: String,
    /// "assistant": calls issued this turn (optional).
    pub tool_calls: Vec<ToolCall>,
    /// "tool": id of the call this responds to (optional).
    pub tool_call_id: Option<String>,
    /// "tool": name of the function that ran (optional).
    pub tool_name: Option<String>,
}

impl ChatMessage {
    /// Creates a simple chat message with role and content.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Creates an assistant message with tool calls.
    pub fn assistant_with_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            tool_calls,
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Creates a tool response message.
    pub fn tool_response(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            tool_name: Some(tool_name.into()),
        }
    }
}

pub(crate) struct RawLlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_LlmChatMessage>,
    _roles: Vec<CString>,
    _contents: Vec<CString>,
    _tool_call_ids: Vec<Option<CString>>,
    _tool_names: Vec<Option<CString>>,
    _tool_calls_structs: Vec<Vec<ffi::geniex_ToolCall>>,
    _tool_calls_strings: Vec<Vec<(Option<CString>, CString, CString)>>,
}

impl ChatMessage {
    pub(crate) fn vec_to_raw(messages: &[ChatMessage]) -> RawLlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut contents = Vec::with_capacity(messages.len());
        let mut tool_call_ids = Vec::with_capacity(messages.len());
        let mut tool_names = Vec::with_capacity(messages.len());
        let mut tool_calls_structs = Vec::with_capacity(messages.len());
        let mut tool_calls_strings = Vec::with_capacity(messages.len());
        let mut raw_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role_c = safe_c_string(&msg.role);
            let content_c = safe_c_string(&msg.content);

            let tool_call_id_c = msg.tool_call_id.as_ref().map(|s| safe_c_string(s));
            let tool_name_c = msg.tool_name.as_ref().map(|s| safe_c_string(s));

            let mut tc_strings = Vec::with_capacity(msg.tool_calls.len());
            let mut tc_structs = Vec::with_capacity(msg.tool_calls.len());

            for tc in &msg.tool_calls {
                let id_c = tc.id.as_ref().map(|s| safe_c_string(s));
                let name_c = safe_c_string(&tc.name);
                let args_c = safe_c_string(&tc.arguments);

                tc_structs.push(ffi::geniex_ToolCall {
                    id: id_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                    name: name_c.as_ptr(),
                    arguments: args_c.as_ptr(),
                });

                tc_strings.push((id_c, name_c, args_c));
            }

            raw_messages.push(ffi::geniex_LlmChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
                tool_calls: if tc_structs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    tc_structs.as_ptr() as *mut _
                },
                tool_call_count: tc_structs.len() as i32,
                tool_call_id: tool_call_id_c
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
                tool_name: tool_name_c
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
            });

            roles.push(role_c);
            contents.push(content_c);
            tool_call_ids.push(tool_call_id_c);
            tool_names.push(tool_name_c);
            tool_calls_structs.push(tc_structs);
            tool_calls_strings.push(tc_strings);
        }

        RawLlmChatMessages {
            raw_messages,
            _roles: roles,
            _contents: contents,
            _tool_call_ids: tool_call_ids,
            _tool_names: tool_names,
            _tool_calls_structs: tool_calls_structs,
            _tool_calls_strings: tool_calls_strings,
        }
    }
}

/// Content element for Vision-Language Models (text or media type).
#[derive(Debug, Clone)]
pub struct VlmContent {
    /// Content type (e.g., "text", "image", "audio").
    pub r#type: String,
    /// Text value or file path payload.
    pub text: String,
}

/// Chat message structure for Vision-Language Models.
#[derive(Debug, Clone, Default)]
pub struct VlmChatMessage {
    /// Message role ("user", "assistant", "system", "tool", …).
    pub role: String,
    /// Slice of content payloads (text and media items).
    pub contents: Vec<VlmContent>,
    /// "assistant": calls issued this turn (optional).
    pub tool_calls: Vec<ToolCall>,
    /// "tool": id of the call this responds to (optional).
    pub tool_call_id: Option<String>,
    /// "tool": name of the function that ran (optional).
    pub tool_name: Option<String>,
}

impl VlmChatMessage {
    /// Creates a new VLM chat message.
    pub fn new(role: impl Into<String>, contents: Vec<VlmContent>) -> Self {
        Self {
            role: role.into(),
            contents,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        }
    }
}

pub(crate) struct RawVlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_VlmChatMessage>,
    _roles: Vec<CString>,
    _content_structs: Vec<Vec<ffi::geniex_VlmContent>>,
    _type_cstrings: Vec<Vec<CString>>,
    _text_cstrings: Vec<Vec<CString>>,
    _tool_call_ids: Vec<Option<CString>>,
    _tool_names: Vec<Option<CString>>,
    _tool_calls_structs: Vec<Vec<ffi::geniex_ToolCall>>,
    _tool_calls_strings: Vec<Vec<(Option<CString>, CString, CString)>>,
}

impl VlmChatMessage {
    pub(crate) fn vec_to_raw(messages: &[VlmChatMessage]) -> RawVlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut content_structs = Vec::with_capacity(messages.len());
        let mut type_cstrings = Vec::with_capacity(messages.len());
        let mut text_cstrings = Vec::with_capacity(messages.len());
        let mut tool_call_ids = Vec::with_capacity(messages.len());
        let mut tool_names = Vec::with_capacity(messages.len());
        let mut tool_calls_structs = Vec::with_capacity(messages.len());
        let mut tool_calls_strings = Vec::with_capacity(messages.len());
        let mut raw_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role_c = safe_c_string(&msg.role);
            let mut sub_types = Vec::with_capacity(msg.contents.len());
            let mut sub_texts = Vec::with_capacity(msg.contents.len());
            let mut sub_structs = Vec::with_capacity(msg.contents.len());

            for c in &msg.contents {
                let t_c = safe_c_string(&c.r#type);
                let txt_c = safe_c_string(&c.text);
                sub_structs.push(ffi::geniex_VlmContent {
                    type_: t_c.as_ptr(),
                    text: txt_c.as_ptr(),
                });
                sub_types.push(t_c);
                sub_texts.push(txt_c);
            }

            let tool_call_id_c = msg.tool_call_id.as_ref().map(|s| safe_c_string(s));
            let tool_name_c = msg.tool_name.as_ref().map(|s| safe_c_string(s));

            let mut tc_strings = Vec::with_capacity(msg.tool_calls.len());
            let mut tc_structs = Vec::with_capacity(msg.tool_calls.len());

            for tc in &msg.tool_calls {
                let id_c = tc.id.as_ref().map(|s| safe_c_string(s));
                let name_c = safe_c_string(&tc.name);
                let args_c = safe_c_string(&tc.arguments);

                tc_structs.push(ffi::geniex_ToolCall {
                    id: id_c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                    name: name_c.as_ptr(),
                    arguments: args_c.as_ptr(),
                });

                tc_strings.push((id_c, name_c, args_c));
            }

            raw_messages.push(ffi::geniex_VlmChatMessage {
                role: role_c.as_ptr(),
                contents: if sub_structs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    sub_structs.as_ptr() as *mut _
                },
                content_count: sub_structs.len() as i64,
                tool_calls: if tc_structs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    tc_structs.as_ptr() as *mut _
                },
                tool_call_count: tc_structs.len() as i32,
                tool_call_id: tool_call_id_c
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
                tool_name: tool_name_c
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
            });

            roles.push(role_c);
            content_structs.push(sub_structs);
            type_cstrings.push(sub_types);
            text_cstrings.push(sub_texts);
            tool_call_ids.push(tool_call_id_c);
            tool_names.push(tool_name_c);
            tool_calls_structs.push(tc_structs);
            tool_calls_strings.push(tc_strings);
        }

        RawVlmChatMessages {
            raw_messages,
            _roles: roles,
            _content_structs: content_structs,
            _type_cstrings: type_cstrings,
            _text_cstrings: text_cstrings,
            _tool_call_ids: tool_call_ids,
            _tool_names: tool_names,
            _tool_calls_structs: tool_calls_structs,
            _tool_calls_strings: tool_calls_strings,
        }
    }
}

/// Hardware capabilities supported by a VLM plugin.
#[derive(Debug, Clone, Copy, Default)]
pub struct VlmCapabilities {
    /// True if vision/image inputs are supported.
    pub supports_vision: bool,
    /// True if audio inputs are supported.
    pub supports_audio: bool,
}

impl From<&ffi::geniex_VlmCapabilities> for VlmCapabilities {
    fn from(raw: &ffi::geniex_VlmCapabilities) -> Self {
        Self {
            supports_vision: raw.supports_vision,
            supports_audio: raw.supports_audio,
        }
    }
}

/// Metadata and token parameters for a loaded LLM model.
#[derive(Debug, Clone, Copy, Default)]
pub struct LlmModelInfo {
    /// Vocabulary size.
    pub vocab_size: i32,
    /// Beginning of sequence token ID.
    pub bos_token: i32,
    /// Flag indicating whether BOS token is prepended automatically.
    pub add_bos: i32,
}

impl From<&ffi::geniex_LlmModelInfo> for LlmModelInfo {
    fn from(raw: &ffi::geniex_LlmModelInfo) -> Self {
        Self {
            vocab_size: raw.vocab_size,
            bos_token: raw.bos_token,
            add_bos: raw.add_bos,
        }
    }
}

/// Input parameters for hardware device alias resolution.
#[derive(Debug, Clone)]
pub struct ResolveDeviceInput {
    /// Target plugin identifier (e.g., "llama_cpp", "qairt").
    pub plugin_id: String,
    /// Optional model file path or identifier.
    pub model_name: Option<String>,
    /// Optional execution mode ("cpu", "gpu", "npu").
    pub mode: Option<String>,
    /// Default number of GPU/NPU offload layers.
    pub ngl_default: i32,
}

/// Resolved compute device configuration.
#[derive(Debug, Clone, Default)]
pub struct ResolveDeviceOutput {
    /// Resolved native device ID string.
    pub device_id: Option<String>,
    /// Resolved offload layer count.
    pub ngl: i32,
    /// Optional hardware compatibility warning message.
    pub warning: Option<String>,
}

/// List of available hardware execution devices.
#[derive(Debug, Clone, Default)]
pub struct DeviceList {
    /// Unique target device identifiers.
    pub device_ids: Vec<String>,
    /// Human-readable device names.
    pub device_names: Vec<String>,
}

/// Strongly typed media content for Vision-Language Models (VLM).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VlmMedia {
    /// Plain text payload.
    Text(String),
    /// File path to an input image.
    Image(std::path::PathBuf),
    /// File path to an input audio recording.
    Audio(std::path::PathBuf),
}

impl From<VlmMedia> for VlmContent {
    fn from(media: VlmMedia) -> Self {
        match media {
            VlmMedia::Text(t) => VlmContent {
                r#type: "text".to_string(),
                text: t,
            },
            VlmMedia::Image(p) => VlmContent {
                r#type: "image".to_string(),
                text: p.to_string_lossy().into_owned(),
            },
            VlmMedia::Audio(p) => VlmContent {
                r#type: "audio".to_string(),
                text: p.to_string_lossy().into_owned(),
            },
        }
    }
}

/// Raw logits and corresponding token IDs returned by a single forward pass.
#[derive(Debug, Clone, Default)]
pub struct ForwardLogitsOutput {
    /// Row-major logits buffer. Length is `n_rows * row_width`.
    pub logits: Vec<f32>,
    /// Token IDs corresponding to the logits, populated only when `top_n > 0`.
    pub token_ids: Option<Vec<i32>>,
    /// Number of rows returned (depends on `all_positions`).
    pub n_rows: usize,
    /// Width of each row (depends on `top_n` and `vocab_size`).
    pub row_width: usize,
    /// Full vocabulary size of the model.
    pub vocab_size: usize,
}
