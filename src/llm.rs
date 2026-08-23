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
        tokenizer_path: Option<&str>,
        device_id: Option<&str>,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
        let c_tokenizer_path = tokenizer_path
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;
        let c_device_id = device_id
            .map(|s| CString::new(s).map_err(|_| GeniexError::CommonInvalidInput))
            .transpose()?;

        let raw_config = config.to_raw();

        let raw_input = ffi::geniex_LlmCreateInput {
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

    /// Generates response tokens asynchronously on a background thread, yielding strings via an Iterator.
    ///
    /// Dropping the returned Iterator cancels token generation early inside the C engine.
    pub fn generate_iter(
        &mut self,
        prompt: Option<&str>,
        input_ids: Option<&[i32]>,
        config: Option<&GenerationConfig>,
    ) -> LlmIterator<'_> {
        let (tx, rx) = std::sync::mpsc::channel();
        let c_prompt = prompt.map(|s| s.to_string());
        let owned_ids = input_ids.map(|ids| ids.to_vec());
        let owned_config = config.cloned();
        let raw_handle = self.handle as usize;

        std::thread::spawn(move || {
            let c_prompt_str = c_prompt
                .as_ref()
                .map(|s| CString::new(s.as_str()).map_err(|_| GeniexError::CommonInvalidInput));

            let c_prompt_c = match c_prompt_str {
                Some(Ok(cs)) => Some(cs),
                Some(Err(e)) => {
                    let _ = tx.send(Err(e));
                    return;
                }
                None => None,
            };

            let raw_config = owned_config.as_ref().map(|c| c.to_raw());
            let user_data = Box::into_raw(Box::new(tx));

            extern "C" fn token_callback(token: *const c_char, user_data: *mut c_void) -> bool {
                if token.is_null() || user_data.is_null() {
                    return true;
                }
                let tx = unsafe { &*(user_data as *const std::sync::mpsc::Sender<Result<String>>) };
                let s = unsafe { CStr::from_ptr(token).to_string_lossy().into_owned() };
                tx.send(Ok(s)).is_ok()
            }

            let (ids_ptr, ids_count) = if let Some(ref ids) = owned_ids {
                (ids.as_ptr(), ids.len() as i32)
            } else {
                (ptr::null(), 0)
            };

            let input = ffi::geniex_LlmGenerateInput {
                prompt_utf8: c_prompt_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                config: raw_config.as_ref().map_or(ptr::null(), |c| &c.raw),
                on_token: Some(token_callback),
                user_data: user_data as *mut c_void,
                input_ids: ids_ptr,
                input_ids_count: ids_count,
            };

            let mut output = ffi::geniex_LlmGenerateOutput {
                full_text: ptr::null_mut(),
                profile_data: unsafe { std::mem::zeroed() },
            };

            let code = unsafe {
                ffi::geniex_llm_generate(raw_handle as *mut ffi::geniex_LLM, &input, &mut output)
            };
            let tx = unsafe { Box::from_raw(user_data) };

            if let Err(e) = GeniexError::check(code) {
                let _ = tx.send(Err(e));
            }

            if !output.full_text.is_null() {
                unsafe { ffi::geniex_free(output.full_text as *mut _) };
            }
        });

        LlmIterator { _llm: self, rx }
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

/// Iterator yielding generated text tokens from the LLM.
pub struct LlmIterator<'a> {
    _llm: &'a mut Llm,
    rx: std::sync::mpsc::Receiver<Result<String>>,
}

impl<'a> Iterator for LlmIterator<'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

/// High-level session manager that handles chat history and automatically saves/loads KV cache context.
pub struct ChatSession<'a> {
    llm: &'a mut Llm,
    history: Vec<ChatMessage>,
}

impl<'a> ChatSession<'a> {
    /// Creates a new chat session using a loaded LLM.
    pub fn new(llm: &'a mut Llm) -> Self {
        Self {
            llm,
            history: Vec::new(),
        }
    }

    /// Appends a new chat message to the session history.
    pub fn push_message(&mut self, role: &str, content: &str) {
        self.history.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    /// Accesses the complete conversation history.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clears the conversation history.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Sends a prompt to the LLM, returning the full response text and appending both prompt and response to history.
    pub fn send_message(
        &mut self,
        prompt: &str,
        enable_thinking: bool,
        config: Option<&GenerationConfig>,
    ) -> Result<String> {
        self.push_message("user", prompt);
        let formatted = self
            .llm
            .apply_chat_template(&self.history, None, enable_thinking, true)?;

        let (response, _) =
            self.llm
                .generate::<fn(&str) -> bool>(Some(&formatted), None, config, None)?;

        self.push_message("assistant", &response);
        Ok(response)
    }

    /// Sends a prompt to the LLM, yielding tokens on a background thread via an iterator, and appending the final response to history.
    pub fn send_message_iter(
        &mut self,
        prompt: &str,
        enable_thinking: bool,
        config: Option<&GenerationConfig>,
    ) -> Result<ChatIterator<'a, '_>> {
        self.push_message("user", prompt);
        let formatted = self
            .llm
            .apply_chat_template(&self.history, None, enable_thinking, true)?;

        let (tx, rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();

        let raw_handle = self.llm.handle as usize;
        let c_prompt = CString::new(formatted).map_err(|_| GeniexError::CommonInvalidInput)?;
        let owned_config = config.cloned();

        std::thread::spawn(move || {
            let raw_config = owned_config.as_ref().map(|c| c.to_raw());
            let user_data = Box::into_raw(Box::new(tx));

            extern "C" fn token_callback(token: *const c_char, user_data: *mut c_void) -> bool {
                if token.is_null() || user_data.is_null() {
                    return true;
                }
                let tx = unsafe { &*(user_data as *const std::sync::mpsc::Sender<Result<String>>) };
                let s = unsafe { CStr::from_ptr(token).to_string_lossy().into_owned() };
                tx.send(Ok(s)).is_ok()
            }

            let input = ffi::geniex_LlmGenerateInput {
                prompt_utf8: c_prompt.as_ptr(),
                config: raw_config.as_ref().map_or(ptr::null(), |c| &c.raw),
                on_token: Some(token_callback),
                user_data: user_data as *mut c_void,
                input_ids: ptr::null(),
                input_ids_count: 0,
            };

            let mut output = ffi::geniex_LlmGenerateOutput {
                full_text: ptr::null_mut(),
                profile_data: unsafe { std::mem::zeroed() },
            };

            let code = unsafe {
                ffi::geniex_llm_generate(raw_handle as *mut ffi::geniex_LLM, &input, &mut output)
            };
            let tx = unsafe { Box::from_raw(user_data) };

            let full_text = if !output.full_text.is_null() {
                let s = unsafe {
                    CStr::from_ptr(output.full_text)
                        .to_string_lossy()
                        .into_owned()
                };
                unsafe { ffi::geniex_free(output.full_text as *mut _) };
                s
            } else {
                String::new()
            };

            if let Err(e) = GeniexError::check(code) {
                let _ = tx.send(Err(e));
            }

            let _ = done_tx.send(full_text);
        });

        Ok(ChatIterator {
            session: self,
            rx,
            done_rx: Some(done_rx),
            acc: String::new(),
        })
    }
}

/// Iterator yielding tokens from a chat session. Appends the full message to history when finished or dropped.
pub struct ChatIterator<'a, 'b> {
    session: &'b mut ChatSession<'a>,
    rx: std::sync::mpsc::Receiver<Result<String>>,
    done_rx: Option<std::sync::mpsc::Receiver<String>>,
    acc: String,
}

impl<'a, 'b> Iterator for ChatIterator<'a, 'b> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.recv() {
            Ok(Ok(s)) => {
                self.acc.push_str(&s);
                Some(Ok(s))
            }
            Ok(Err(e)) => Some(Err(e)),
            Err(_) => {
                if let Some(done_rx) = self.done_rx.take() {
                    if let Ok(full_text) = done_rx.try_recv() {
                        self.session.push_message("assistant", &full_text);
                    } else {
                        let acc = std::mem::take(&mut self.acc);
                        self.session.push_message("assistant", &acc);
                    }
                }
                None
            }
        }
    }
}

impl<'a, 'b> Drop for ChatIterator<'a, 'b> {
    fn drop(&mut self) {
        if let Some(done_rx) = self.done_rx.take() {
            let final_text = done_rx
                .try_recv()
                .unwrap_or_else(|_| std::mem::take(&mut self.acc));
            if !final_text.is_empty() {
                self.session.push_message("assistant", &final_text);
            }
        }
    }
}
