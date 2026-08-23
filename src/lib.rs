//! # `geniex` - Rust Bindings for Qualcomm GenieX C API
//!
//! Safe, high-level, idiomatic Rust wrappers for Qualcomm GenieX Large Language Model (LLM)
//! and Vision-Language Model (VLM) inference engines.
//!
//! ## Overview
//!
//! The `geniex` crate provides RAII handles ([`Llm`], [`Vlm`]), automated device alias resolution,
//! Jinja chat template application, KV cache persistence, and streaming token generation callbacks.

pub mod core;
pub mod error;
pub mod ffi;
pub mod llm;
pub mod types;
pub mod vlm;

pub use core::*;
pub use error::{GeniexError, Result};
pub use llm::{Llm, LlmIterator, ChatSession, ChatIterator};
pub use types::*;
pub use vlm::{Vlm, VlmIterator};
