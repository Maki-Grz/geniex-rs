use crate::error::{GeniexError, Result};
use crate::ffi;
use crate::types::{ChatMessage, GenerationConfig, LlmModelInfo, ModelConfig, ProfileData};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// High-level, thread-safe Rust handle for a GenieX Large Language Model (LLM) session.
pub struct Llm {
    handle: *mut ffi::geniex_LLM,
}

// SAFETY: Llm handle is thread-safe for transfer between threads.
unsafe impl Send for Llm {}
// SAFETY: Llm handle operations are synchronized internally by native SDK.
unsafe impl Sync for Llm {}

impl Llm {
    /// Creates a new LLM instance from a model file (GGUF or Qualcomm AI Hub bundle).
    pub fn create(
        model_path: &str,
        plugin_id: &str,
        config: &ModelConfig,
        model_name: Option<&str>,
        tokenizer_path: Option<&str>,
        device_id: Option<&str>,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_model_name = model_name
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_tokenizer_path = tokenizer_path
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_device_id = device_id
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;

        let raw_config = config.to_raw();

        let raw_input = ffi::geniex_LlmCreateInput {
            model_name: c_model_name.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            model_path: c_model_path.as_ptr(),
            tokenizer_path: c_tokenizer_path
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
            config: raw_config.raw,
            plugin_id: c_plugin_id.as_ptr(),
            device_id: c_device_id.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        };

        let mut handle: *mut ffi::geniex_LLM = ptr::null_mut();
        // SAFETY: FFI call creating LLM model handle.
        let code = unsafe { ffi::geniex_llm_create(&raw_input, &mut handle) };
        GeniexError::check(code)?;

        if handle.is_null() {
            Err(GeniexError::CommonMemoryAllocation)
        } else {
            Ok(Self { handle })
        }
    }

    /// Resets the internal KV cache and state of the LLM.
    pub fn reset(&mut self) -> Result<()> {
        // SAFETY: FFI call resetting LLM handle state.
        let code = unsafe { ffi::geniex_llm_reset(self.handle) };
        GeniexError::check(code)
    }

    /// Saves the current KV cache state to a file.
    pub fn save_kv_cache(&mut self, path: &str) -> Result<()> {
        let c_path = CString::new(path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let input = ffi::geniex_KvCacheSaveInput {
            path: c_path.as_ptr(),
        };
        let mut output = ffi::geniex_KvCacheSaveOutput {
            reserved: ptr::null_mut(),
        };
        // SAFETY: FFI call writing KV cache to filesystem.
        let code = unsafe { ffi::geniex_llm_save_kv_cache(self.handle, &input, &mut output) };
        GeniexError::check(code)
    }

    /// Loads a previously saved KV cache state from a file.
    pub fn load_kv_cache(&mut self, path: &str) -> Result<()> {
        let c_path = CString::new(path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let input = ffi::geniex_KvCacheLoadInput {
            path: c_path.as_ptr(),
        };
        let mut output = ffi::geniex_KvCacheLoadOutput {
            reserved: ptr::null_mut(),
        };
        // SAFETY: FFI call reading KV cache state into runtime.
        let code = unsafe { ffi::geniex_llm_load_kv_cache(self.handle, &input, &mut output) };
        GeniexError::check(code)
    }

    /// Formats a list of chat messages using the model's native chat template.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        tools: Option<&str>,
        enable_thinking: bool,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let raw_messages = ChatMessage::vec_to_raw(messages);
        let c_tools = tools
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;

        let input = ffi::geniex_LlmApplyChatTemplateInput {
            messages: raw_messages.raw_messages.as_ptr() as *mut _,
            message_count: raw_messages.raw_messages.len() as i32,
            tools: c_tools.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            enable_thinking,
            add_generation_prompt,
        };

        let mut output = ffi::geniex_LlmApplyChatTemplateOutput {
            formatted_text: ptr::null_mut(),
        };

        // SAFETY: FFI call applying chat template formatting.
        let code = unsafe { ffi::geniex_llm_apply_chat_template(self.handle, &input, &mut output) };
        GeniexError::check(code)?;

        if output.formatted_text.is_null() {
            Ok(String::new())
        } else {
            // SAFETY: Converting non-null returned formatted string pointer.
            let result = unsafe {
                CStr::from_ptr(output.formatted_text)
                    .to_string_lossy()
                    .into_owned()
            };
            // SAFETY: Freeing memory allocated by C SDK for formatted_text string.
            unsafe { ffi::geniex_free(output.formatted_text as *mut _) };
            Ok(result)
        }
    }

    /// Generates text from a prompt or input token IDs, streaming tokens via callback.
    pub fn generate<F>(
        &mut self,
        prompt: Option<&str>,
        input_ids: Option<&[i32]>,
        config: Option<&GenerationConfig>,
        mut callback: Option<F>,
    ) -> Result<(String, ProfileData)>
    where
        F: FnMut(&str) -> bool,
    {
        let c_prompt = prompt
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let raw_config = config.map(|c| c.to_raw());

        extern "C" fn token_trampoline<F: FnMut(&str) -> bool>(
            token: *const c_char,
            user_data: *mut c_void,
        ) -> bool {
            if token.is_null() || user_data.is_null() {
                return true;
            }
            // SAFETY: Dereferencing user_data pointer back to Rust closure handle.
            unsafe {
                let cb = &mut *(user_data as *mut F);
                let s = CStr::from_ptr(token).to_string_lossy();
                cb(&s)
            }
        }

        let (cb_ptr, user_data_ptr): (ffi::geniex_token_callback, *mut c_void) =
            if let Some(ref mut cb) = callback {
                (Some(token_trampoline::<F>), cb as *mut F as *mut c_void)
            } else {
                (None, ptr::null_mut())
            };

        let (ids_ptr, ids_count) = if let Some(ids) = input_ids {
            (ids.as_ptr(), ids.len() as i32)
        } else {
            (ptr::null(), 0)
        };

        let input = ffi::geniex_LlmGenerateInput {
            prompt_utf8: c_prompt.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            config: raw_config.as_ref().map_or(ptr::null(), |c| &c.raw),
            on_token: cb_ptr,
            user_data: user_data_ptr,
            input_ids: ids_ptr,
            input_ids_count: ids_count,
        };

        // SAFETY: Zeroing stack output struct memory before FFI call.
        let mut output = ffi::geniex_LlmGenerateOutput {
            full_text: ptr::null_mut(),
            profile_data: unsafe { std::mem::zeroed() },
        };

        // SAFETY: FFI call triggering text generation loop.
        let code = unsafe { ffi::geniex_llm_generate(self.handle, &input, &mut output) };
        GeniexError::check(code)?;

        let full_text = if output.full_text.is_null() {
            String::new()
        } else {
            // SAFETY: Reading returned full generation string pointer.
            let s = unsafe {
                CStr::from_ptr(output.full_text)
                    .to_string_lossy()
                    .into_owned()
            };
            // SAFETY: Freeing memory allocated by C SDK for full_text string.
            unsafe { ffi::geniex_free(output.full_text as *mut _) };
            s
        };

        let profile_data = ProfileData::from(&output.profile_data);
        Ok((full_text, profile_data))
    }

    /// Retrieves metadata parameters for the loaded LLM model.
    pub fn get_model_info(&self) -> Result<LlmModelInfo> {
        // SAFETY: Zeroing raw struct memory before FFI call.
        let mut raw = unsafe { std::mem::zeroed::<ffi::geniex_LlmModelInfo>() };
        // SAFETY: FFI call querying model metadata.
        let code = unsafe { ffi::geniex_llm_get_model_info(self.handle, &mut raw) };
        GeniexError::check(code)?;
        Ok(LlmModelInfo::from(&raw))
    }

    /// Returns the raw underlying C SDK handle pointer.
    pub fn raw_handle(&self) -> *mut ffi::geniex_LLM {
        self.handle
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Destroying raw C SDK handle on drop.
            unsafe {
                ffi::geniex_llm_destroy(self.handle);
            }
        }
    }
}
