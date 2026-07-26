use crate::ffi;
use std::fmt;

/// Error types returned by the GenieX C API and Rust wrapper functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeniexError {
    /// Common unknown error.
    CommonUnknown,
    /// Invalid input parameter supplied.
    CommonInvalidInput,
    /// Specified compute device is invalid or unavailable.
    CommonInvalidDevice,
    /// Memory allocation failed in native C runtime.
    CommonMemoryAllocation,
    /// Specified file or model binary was not found.
    CommonFileNotFound,
    /// Network communication failure.
    CommonNetwork,
    /// Operation was cancelled.
    CommonCancelled,
    /// GenieX SDK has not been initialized.
    CommonNotInitialized,
    /// GenieX SDK has already been initialized.
    CommonAlreadyInitialized,
    /// Authentication or authorization failure.
    CommonAuth,
    /// Model was not found on Qualcomm AI Hub.
    CommonHubModelNotFound,
    /// Request rate limit reached.
    CommonRateLimited,
    /// Qualcomm AI Hub server error.
    CommonHubServer,
    /// Operation or parameter combination is not supported.
    CommonNotSupported,
    /// Failed to parse model manifest.
    CommonManifestParse,
    /// Selected hardware chipset is unavailable.
    CommonChipsetUnavailable,
    /// Specified parameter is not supported by the plugin.
    CommonParamNotSupported,
    /// Failed to load model weights.
    CommonModelLoad,
    /// Model binary is invalid or corrupted.
    CommonModelInvalid,
    /// Failed to load dynamic plugin library.
    CommonPluginLoad,
    /// Plugin interface is invalid.
    CommonPluginInvalid,
    /// Tokenization failed during text encoding.
    LlmTokenizationFailed,
    /// Prompt exceeded maximum tokenization context length.
    LlmTokenizationContextLength,
    /// Text generation failed in LLM engine.
    LlmGenerationFailed,
    /// Prompt text is too long for generation.
    LlmGenerationPromptTooLong,
    /// Failed to load image input for VLM.
    VlmImageLoad,
    /// Image format is unsupported or corrupted.
    VlmImageFormat,
    /// Failed to load audio input for VLM.
    VlmAudioLoad,
    /// Audio format is unsupported or corrupted.
    VlmAudioFormat,
    /// Multimodal generation failed in VLM engine.
    VlmGenerationFailed,
    /// Unrecognized return code from native C SDK.
    Unknown(i32),
}

impl GeniexError {
    /// Checks a native status code and converts non-success status to a [`Result`].
    pub fn check(code: i32) -> Result<()> {
        if code == ffi::geniex_ErrorCode_GENIEX_SUCCESS {
            Ok(())
        } else {
            Err(Self::from_i32(code))
        }
    }

    /// Converts a raw C integer error code into a [`GeniexError`] enum variant.
    pub fn from_i32(code: i32) -> Self {
        match code {
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_UNKNOWN => Self::CommonUnknown,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_INPUT => Self::CommonInvalidInput,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_DEVICE => Self::CommonInvalidDevice,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MEMORY_ALLOCATION => {
                Self::CommonMemoryAllocation
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_FILE_NOT_FOUND => Self::CommonFileNotFound,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NETWORK => Self::CommonNetwork,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CANCELLED => Self::CommonCancelled,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_INITIALIZED => Self::CommonNotInitialized,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_ALREADY_INITIALIZED => {
                Self::CommonAlreadyInitialized
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_AUTH => Self::CommonAuth,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND => {
                Self::CommonHubModelNotFound
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_RATE_LIMITED => Self::CommonRateLimited,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_SERVER => Self::CommonHubServer,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_SUPPORTED => Self::CommonNotSupported,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MANIFEST_PARSE => Self::CommonManifestParse,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE => {
                Self::CommonChipsetUnavailable
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED => {
                Self::CommonParamNotSupported
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_LOAD => Self::CommonModelLoad,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_INVALID => Self::CommonModelInvalid,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_LOAD => Self::CommonPluginLoad,
            ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_INVALID => Self::CommonPluginInvalid,
            ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_FAILED => {
                Self::LlmTokenizationFailed
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH => {
                Self::LlmTokenizationContextLength
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_FAILED => Self::LlmGenerationFailed,
            ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG => {
                Self::LlmGenerationPromptTooLong
            }
            ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_LOAD => Self::VlmImageLoad,
            ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_FORMAT => Self::VlmImageFormat,
            ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_LOAD => Self::VlmAudioLoad,
            ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_FORMAT => Self::VlmAudioFormat,
            ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_GENERATION_FAILED => Self::VlmGenerationFailed,
            _ => Self::Unknown(code),
        }
    }

    /// Retrieves the detailed error string from the native GenieX runtime.
    pub fn message(&self) -> String {
        let code = match self {
            Self::CommonUnknown => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_UNKNOWN,
            Self::CommonInvalidInput => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_INPUT,
            Self::CommonInvalidDevice => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_INVALID_DEVICE,
            Self::CommonMemoryAllocation => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MEMORY_ALLOCATION
            }
            Self::CommonFileNotFound => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_FILE_NOT_FOUND,
            Self::CommonNetwork => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NETWORK,
            Self::CommonCancelled => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CANCELLED,
            Self::CommonNotInitialized => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_INITIALIZED,
            Self::CommonAlreadyInitialized => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_ALREADY_INITIALIZED
            }
            Self::CommonAuth => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_AUTH,
            Self::CommonHubModelNotFound => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND
            }
            Self::CommonRateLimited => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_RATE_LIMITED,
            Self::CommonHubServer => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_HUB_SERVER,
            Self::CommonNotSupported => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_NOT_SUPPORTED,
            Self::CommonManifestParse => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MANIFEST_PARSE,
            Self::CommonChipsetUnavailable => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE
            }
            Self::CommonParamNotSupported => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED
            }
            Self::CommonModelLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_LOAD,
            Self::CommonModelInvalid => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_MODEL_INVALID,
            Self::CommonPluginLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_LOAD,
            Self::CommonPluginInvalid => ffi::geniex_ErrorCode_GENIEX_ERROR_COMMON_PLUGIN_INVALID,
            Self::LlmTokenizationFailed => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_FAILED
            }
            Self::LlmTokenizationContextLength => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH
            }
            Self::LlmGenerationFailed => ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_FAILED,
            Self::LlmGenerationPromptTooLong => {
                ffi::geniex_ErrorCode_GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG
            }
            Self::VlmImageLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_LOAD,
            Self::VlmImageFormat => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_IMAGE_FORMAT,
            Self::VlmAudioLoad => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_LOAD,
            Self::VlmAudioFormat => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_AUDIO_FORMAT,
            Self::VlmGenerationFailed => ffi::geniex_ErrorCode_GENIEX_ERROR_VLM_GENERATION_FAILED,
            Self::Unknown(c) => *c,
        };
        // SAFETY: geniex_get_error_message takes an integer error code and returns either
        // a valid null-terminated C string pointer or null if unknown.
        unsafe {
            let ptr = ffi::geniex_get_error_message(code);
            if ptr.is_null() {
                format!("Unknown error code ({})", code)
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }
}

impl fmt::Display for GeniexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for GeniexError {}

/// Specialized Result type for GenieX operations.
pub type Result<T> = std::result::Result<T, GeniexError>;
