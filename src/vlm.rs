use crate::error::{GeniexError, Result};
use crate::ffi;
use crate::types::{GenerationConfig, ModelConfig, ProfileData, VlmCapabilities, VlmChatMessage};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

/// High-level, thread-safe Rust handle for a GenieX Vision-Language Model (VLM) session.
pub struct Vlm {
    handle: *mut ffi::geniex_VLM,
}

// SAFETY: Vlm handle is thread-safe for transfer between threads.
unsafe impl Send for Vlm {}
// SAFETY: Vlm handle operations are synchronized internally by native SDK.
unsafe impl Sync for Vlm {}

impl Vlm {
    /// Creates a new VLM instance from a multimodal model file and projector weights.
    pub fn create(
        model_path: &str,
        plugin_id: &str,
        config: &ModelConfig,
        model_name: Option<&str>,
        mmproj_path: Option<&str>,
        tokenizer_path: Option<&str>,
        device_id: Option<&str>,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_model_name = model_name
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_mmproj_path = mmproj_path
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_tokenizer_path = tokenizer_path
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_device_id = device_id
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;

        let raw_config = config.to_raw();

        let raw_input = ffi::geniex_VlmCreateInput {
            model_name: c_model_name.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            model_path: c_model_path.as_ptr(),
            mmproj_path: c_mmproj_path.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            config: raw_config.raw,
            plugin_id: c_plugin_id.as_ptr(),
            device_id: c_device_id.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            tokenizer_path: c_tokenizer_path
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
        };

        let mut handle: *mut ffi::geniex_VLM = ptr::null_mut();
        // SAFETY: FFI call creating VLM model handle.
        let code = unsafe { ffi::geniex_vlm_create(&raw_input, &mut handle) };
        GeniexError::check(code)?;

        if handle.is_null() {
            Err(GeniexError::CommonMemoryAllocation)
        } else {
            Ok(Self { handle })
        }
    }

    /// Resets the internal state of the VLM.
    pub fn reset(&mut self) -> Result<()> {
        // SAFETY: FFI call resetting VLM handle state.
        let code = unsafe { ffi::geniex_vlm_reset(self.handle) };
        GeniexError::check(code)
    }

    /// Formats a list of multimodal chat messages using the model's chat template.
    pub fn apply_chat_template(
        &self,
        messages: &[VlmChatMessage],
        tools: Option<&str>,
        enable_thinking: bool,
        grounding: bool,
    ) -> Result<String> {
        let raw_messages = VlmChatMessage::vec_to_raw(messages);
        let c_tools = tools
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;

        let input = ffi::geniex_VlmApplyChatTemplateInput {
            messages: raw_messages.raw_messages.as_ptr() as *mut _,
            message_count: raw_messages.raw_messages.len() as i32,
            tools: c_tools.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            enable_thinking,
            grounding,
        };

        let mut output = ffi::geniex_VlmApplyChatTemplateOutput {
            formatted_text: ptr::null_mut(),
        };

        // SAFETY: FFI call applying VLM chat template formatting.
        let code = unsafe { ffi::geniex_vlm_apply_chat_template(self.handle, &input, &mut output) };
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

    /// Queries the vision and audio capabilities supported by the loaded VLM engine.
    pub fn get_capabilities(&self) -> Result<VlmCapabilities> {
        // SAFETY: Zeroing raw struct memory before FFI call.
        let mut raw = unsafe { std::mem::zeroed::<ffi::geniex_VlmCapabilities>() };
        // SAFETY: FFI call querying VLM capabilities.
        let code = unsafe { ffi::geniex_vlm_get_capabilities(self.handle, &mut raw) };
        GeniexError::check(code)?;
        Ok(VlmCapabilities::from(&raw))
    }

    /// Generates multimodal responses given text and image/audio input payloads.
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        config: Option<&GenerationConfig>,
        mut callback: Option<F>,
    ) -> Result<(String, ProfileData)>
    where
        F: FnMut(&str) -> bool,
    {
        let c_prompt = CString::new(prompt).map_err(|_| GeniexError::CommonInvalidInput)?;
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

        let input = ffi::geniex_VlmGenerateInput {
            prompt_utf8: c_prompt.as_ptr(),
            config: raw_config.as_ref().map_or(ptr::null(), |c| &c.raw),
            on_token: cb_ptr,
            user_data: user_data_ptr,
        };

        // SAFETY: Zeroing stack output struct memory before FFI call.
        let mut output = ffi::geniex_VlmGenerateOutput {
            full_text: ptr::null_mut(),
            profile_data: unsafe { std::mem::zeroed() },
        };

        // SAFETY: FFI call triggering VLM generation.
        let code = unsafe { ffi::geniex_vlm_generate(self.handle, &input, &mut output) };
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

    /// Returns the raw underlying C SDK handle pointer.
    pub fn raw_handle(&self) -> *mut ffi::geniex_VLM {
        self.handle
    }
}

impl Drop for Vlm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Destroying raw C SDK handle on drop.
            unsafe {
                ffi::geniex_vlm_destroy(self.handle);
            }
        }
    }
}
