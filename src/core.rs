use crate::error::{GeniexError, Result};
use crate::ffi;
use crate::types::{DeviceList, ResolveDeviceInput, ResolveDeviceOutput};
use std::ffi::{CStr, CString};

/// Initializes the native GenieX C SDK runtime environment.
pub fn init() -> Result<()> {
    // SAFETY: Call to C SDK runtime initialization function.
    let code = unsafe { ffi::geniex_init() };
    GeniexError::check(code)
}

/// De-initializes and cleans up the native GenieX C SDK runtime environment.
pub fn deinit() -> Result<()> {
    // SAFETY: Call to C SDK runtime teardown function.
    let code = unsafe { ffi::geniex_deinit() };
    GeniexError::check(code)
}

/// Retrieves the version string of the underlying GenieX native SDK.
pub fn version() -> String {
    // SAFETY: geniex_version returns a pointer to a static C string string or null.
    unsafe {
        let ptr = ffi::geniex_version();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Retrieves the version string of a specific loaded plugin.
pub fn get_plugin_version(plugin_id: &str) -> Option<String> {
    let c_id = CString::new(plugin_id).ok()?;
    // SAFETY: FFI call passing a null-terminated plugin ID string pointer.
    unsafe {
        let ptr = ffi::geniex_get_plugin_version(c_id.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

/// Scans and returns a list of installed GenieX execution plugin identifiers (e.g. `llama_cpp`, `qairt`).
pub fn get_plugin_list() -> Result<Vec<String>> {
    let mut output = ffi::geniex_GetPluginListOutput {
        plugin_ids: std::ptr::null_mut(),
        plugin_count: 0,
    };
    // SAFETY: ffi call populating plugin output struct.
    let code = unsafe { ffi::geniex_get_plugin_list(&mut output) };
    GeniexError::check(code)?;

    let mut result = Vec::new();
    if !output.plugin_ids.is_null() && output.plugin_count > 0 {
        // SAFETY: Constructing slice from native pointer array up to plugin_count items.
        let slice =
            unsafe { std::slice::from_raw_parts(output.plugin_ids, output.plugin_count as usize) };
        for &ptr in slice {
            if !ptr.is_null() {
                // SAFETY: Reading C string pointer from plugin array.
                let s = unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() };
                result.push(s);
            }
        }
        // SAFETY: Freeing memory allocated by the C SDK for the plugin list array.
        unsafe {
            ffi::geniex_free(output.plugin_ids as *mut _);
        }
    }
    Ok(result)
}

/// Retrieves the available hardware acceleration device list for a given plugin.
pub fn get_device_list(plugin_id: &str) -> Result<DeviceList> {
    let c_plugin_id = CString::new(plugin_id).map_err(|_| GeniexError::CommonInvalidInput)?;
    let input = ffi::geniex_GetDeviceListInput {
        plugin_id: c_plugin_id.as_ptr(),
    };
    let mut output = ffi::geniex_GetDeviceListOutput {
        device_ids: std::ptr::null_mut(),
        device_names: std::ptr::null_mut(),
        device_count: 0,
    };

    // SAFETY: FFI call with input/output struct pointers.
    let code = unsafe { ffi::geniex_get_device_list(&input, &mut output) };
    GeniexError::check(code)?;

    let mut device_ids = Vec::new();
    let mut device_names = Vec::new();

    if output.device_count > 0 {
        if !output.device_ids.is_null() {
            // SAFETY: Constructing slice from device_ids pointer array.
            let slice = unsafe {
                std::slice::from_raw_parts(output.device_ids, output.device_count as usize)
            };
            for &ptr in slice {
                if !ptr.is_null() {
                    // SAFETY: Converting non-null device ID string.
                    device_ids.push(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() });
                }
            }
            // SAFETY: Freeing memory allocated by C SDK for device_ids array.
            unsafe {
                ffi::geniex_free(output.device_ids as *mut _);
            }
        }
        if !output.device_names.is_null() {
            // SAFETY: Constructing slice from device_names pointer array.
            let slice = unsafe {
                std::slice::from_raw_parts(output.device_names, output.device_count as usize)
            };
            for &ptr in slice {
                if !ptr.is_null() {
                    // SAFETY: Converting non-null device name string.
                    device_names
                        .push(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() });
                }
            }
            // SAFETY: Freeing memory allocated by C SDK for device_names array.
            unsafe {
                ffi::geniex_free(output.device_names as *mut _);
            }
        }
    }

    Ok(DeviceList {
        device_ids,
        device_names,
    })
}

/// Resolves device alias and offload layer configurations for model execution.
pub fn resolve_device(input: &ResolveDeviceInput) -> Result<ResolveDeviceOutput> {
    let c_plugin_id =
        CString::new(input.plugin_id.as_str()).map_err(|_| GeniexError::CommonInvalidInput)?;
    let c_model_name = input
        .model_name
        .as_ref()
        .map(|s| CString::new(s.as_str()).map_err(|_| GeniexError::CommonInvalidInput))
        .transpose()?;
    let c_mode = input
        .mode
        .as_ref()
        .map(|s| CString::new(s.as_str()).map_err(|_| GeniexError::CommonInvalidInput))
        .transpose()?;

    let raw_input = ffi::geniex_ResolveDeviceInput {
        plugin_id: c_plugin_id.as_ptr(),
        model_name: c_model_name
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr()),
        mode: c_mode.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
        ngl_default: input.ngl_default,
    };

    let mut raw_output = ffi::geniex_ResolveDeviceOutput {
        device_id: std::ptr::null_mut(),
        ngl: 0,
        warning: std::ptr::null_mut(),
    };

    // SAFETY: FFI call resolving device selection.
    let code = unsafe { ffi::geniex_resolve_device(&raw_input, &mut raw_output) };
    GeniexError::check(code)?;

    let device_id = if !raw_output.device_id.is_null() {
        // SAFETY: Converting returned C string pointer and freeing memory.
        let s = unsafe {
            CStr::from_ptr(raw_output.device_id)
                .to_string_lossy()
                .into_owned()
        };
        // SAFETY: Freeing allocated device_id memory.
        unsafe { ffi::geniex_free(raw_output.device_id as *mut _) };
        Some(s)
    } else {
        None
    };

    let warning = if !raw_output.warning.is_null() {
        // SAFETY: Converting returned warning C string pointer.
        let s = unsafe {
            CStr::from_ptr(raw_output.warning)
                .to_string_lossy()
                .into_owned()
        };
        // SAFETY: Freeing allocated warning string memory.
        unsafe { ffi::geniex_free(raw_output.warning as *mut _) };
        Some(s)
    } else {
        None
    };

    Ok(ResolveDeviceOutput {
        device_id,
        ngl: raw_output.ngl,
        warning,
    })
}

/// Registers a log callback handler to receive SDK log messages.
pub fn set_log_callback(callback: ffi::geniex_log_callback) -> Result<()> {
    // SAFETY: FFI call passing valid C function pointer for logging.
    let code = unsafe { ffi::geniex_set_log(callback) };
    GeniexError::check(code)
}

/// Registers a custom plugin creation handler with the GenieX runtime.
pub fn register_plugin(
    plugin_id_func: ffi::geniex_plugin_id_func,
    create_func: ffi::geniex_create_plugin_func,
) -> Result<()> {
    // SAFETY: FFI call registering plugin callback entry points.
    let code = unsafe { ffi::geniex_register_plugin(plugin_id_func, create_func) };
    GeniexError::check(code)
}

/// Deallocates memory allocated by the GenieX C SDK runtime.
///
/// # Safety
///
/// `ptr` must be a valid pointer allocated by the GenieX C SDK or null.
pub unsafe fn free_ptr(ptr: *mut std::os::raw::c_void) {
    if !ptr.is_null() {
        // SAFETY: Freeing non-null raw C pointer using SDK free handler.
        ffi::geniex_free(ptr);
    }
}
