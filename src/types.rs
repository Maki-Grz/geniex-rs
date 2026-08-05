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
    /// Time to first token in milliseconds.
    pub ttft: i64,
    /// Total prompt processing time in milliseconds.
    pub prompt_time: i64,
    /// Total decoding time in milliseconds.
    pub decode_time: i64,
    /// Number of prompt tokens processed.
    pub prompt_tokens: i64,
    /// Number of generated tokens produced.
    pub generated_tokens: i64,
    /// Total audio processing duration in milliseconds.
    pub audio_duration: i64,
    /// Prefill speed in tokens per second.
    pub prefill_speed: f64,
    /// Decoding speed in tokens per second.
    pub decoding_speed: f64,
    /// Real-time factor for audio/multimodal execution.
    pub real_time_factor: f64,
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
            prompt_time: raw.prompt_time,
            decode_time: raw.decode_time,
            prompt_tokens: raw.prompt_tokens,
            generated_tokens: raw.generated_tokens,
            audio_duration: raw.audio_duration,
            prefill_speed: raw.prefill_speed,
            decoding_speed: raw.decoding_speed,
            real_time_factor: raw.real_time_factor,
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
    /// Enable strict JSON output schema constraint.
    pub enable_json: bool,
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
            enable_json: false,
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
            enable_json: self.enable_json,
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
    /// Number of past tokens stored in KV cache context.
    pub n_past: i32,
    /// Optional sampler settings.
    pub sampler_config: Option<SamplerConfig>,
    /// Paths to image inputs for multimodal generation.
    pub image_paths: Vec<String>,
    /// Maximum length dimension for image resizing.
    pub image_max_length: i32,
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
            n_past: self.n_past,
            sampler_config: sampler_ptr,
            image_paths: if image_path_ptrs.is_empty() {
                std::ptr::null_mut()
            } else {
                image_path_ptrs.as_ptr() as *mut *const c_char
            },
            image_count: image_path_ptrs.len() as i32,
            image_max_length: self.image_max_length,
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
    /// Default system prompt.
    pub system_prompt: Option<String>,
    /// Enable sampling during generation.
    pub enable_sampling: bool,
    /// Grammar format string.
    pub grammar_str: Option<String>,
    /// Maximum default tokens.
    pub max_tokens: i32,
    /// Enable model thinking/reasoning outputs if supported.
    pub enable_thinking: bool,
    /// Enable verbose SDK log output.
    pub verbose: bool,
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
    _system_prompt: Option<CString>,
    _grammar_str: Option<CString>,
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
        let system_prompt_c = self.system_prompt.as_ref().map(|s| safe_c_string(s));
        let grammar_str_c = self.grammar_str.as_ref().map(|s| safe_c_string(s));
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
            system_prompt: system_prompt_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            enable_sampling: self.enable_sampling,
            grammar_str: grammar_str_c
                .as_ref()
                .map_or(std::ptr::null(), |s| s.as_ptr()),
            max_tokens: self.max_tokens,
            enable_thinking: self.enable_thinking,
            verbose: self.verbose,
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
            _system_prompt: system_prompt_c,
            _grammar_str: grammar_str_c,
            _spec_type: spec_type_c,
            _spec_draft_model: spec_draft_model_c,
        }
    }
}

/// Representation of a single chat message (role and content).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Message sender role (e.g., "user", "assistant", "system").
    pub role: String,
    /// Message body content.
    pub content: String,
}

pub(crate) struct RawLlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_LlmChatMessage>,
    _roles: Vec<CString>,
    _contents: Vec<CString>,
}

impl ChatMessage {
    pub(crate) fn vec_to_raw(messages: &[ChatMessage]) -> RawLlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut contents = Vec::with_capacity(messages.len());
        let mut raw_messages = Vec::with_capacity(messages.len());

        for msg in messages {
            let role_c = safe_c_string(&msg.role);
            let content_c = safe_c_string(&msg.content);
            raw_messages.push(ffi::geniex_LlmChatMessage {
                role: role_c.as_ptr(),
                content: content_c.as_ptr(),
            });
            roles.push(role_c);
            contents.push(content_c);
        }

        RawLlmChatMessages {
            raw_messages,
            _roles: roles,
            _contents: contents,
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
#[derive(Debug, Clone)]
pub struct VlmChatMessage {
    /// Message role ("user", "assistant", "system").
    pub role: String,
    /// Slice of content payloads (text and media items).
    pub contents: Vec<VlmContent>,
}

pub(crate) struct RawVlmChatMessages {
    pub raw_messages: Vec<ffi::geniex_VlmChatMessage>,
    _roles: Vec<CString>,
    _content_structs: Vec<Vec<ffi::geniex_VlmContent>>,
    _type_cstrings: Vec<Vec<CString>>,
    _text_cstrings: Vec<Vec<CString>>,
}

impl VlmChatMessage {
    pub(crate) fn vec_to_raw(messages: &[VlmChatMessage]) -> RawVlmChatMessages {
        let mut roles = Vec::with_capacity(messages.len());
        let mut content_structs = Vec::with_capacity(messages.len());
        let mut type_cstrings = Vec::with_capacity(messages.len());
        let mut text_cstrings = Vec::with_capacity(messages.len());
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

            raw_messages.push(ffi::geniex_VlmChatMessage {
                role: role_c.as_ptr(),
                contents: if sub_structs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    sub_structs.as_ptr() as *mut _
                },
                content_count: sub_structs.len() as i64,
            });

            roles.push(role_c);
            content_structs.push(sub_structs);
            type_cstrings.push(sub_types);
            text_cstrings.push(sub_texts);
        }

        RawVlmChatMessages {
            raw_messages,
            _roles: roles,
            _content_structs: content_structs,
            _type_cstrings: type_cstrings,
            _text_cstrings: text_cstrings,
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
