//! Local narrow C ABI over pinned Apache-2.0 Hugging Face Tokenizers.
//! No Python/runtime downloads/filesystem calls on encode/decode requests.
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use tokenizers::Tokenizer;
mod renderer;

#[repr(C)]
pub struct IdBuffer { pub data: *mut u32, pub len: usize, pub capacity: usize }
#[repr(C)]
pub struct ByteBuffer { pub data: *mut u8, pub len: usize, pub capacity: usize }

fn guarded<F: FnOnce() -> Result<(), String>>(error: *mut u8, size: usize, operation: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => 0,
        result => {
            let message = match result { Ok(Err(message)) => message, _ => "native text panic; operation rejected".to_owned() };
            if !error.is_null() && size > 0 { unsafe {
                let count = message.len().min(size - 1);
                ptr::copy_nonoverlapping(message.as_ptr(), error, count);
                *error.add(count) = 0;
            }}
            1
        }
    }
}
unsafe fn bytes<'a>(data: *const u8, len: usize) -> Result<&'a [u8], String> {
    if len > 32 * 1024 * 1024 { return Err("invalid/bounded text byte input".into()); }
    if data.is_null() { return if len==0 {Ok(&[])} else {Err("null text byte input".into())}; }
    Ok(slice::from_raw_parts(data, len))
}
unsafe fn instance<'a>(handle: *const c_void) -> Result<&'a Tokenizer, String> {
    if handle.is_null() { return Err("null tokenizer handle".into()); }
    Ok(&*(handle as *const Tokenizer))
}
#[no_mangle]
pub unsafe extern "C" fn mellum_tokenizer_create(data: *const u8, len: usize, output: *mut *mut c_void, error: *mut u8, size: usize) -> i32 {
    guarded(error, size, || {
        if output.is_null() { return Err("null tokenizer output".into()); }
        *output = ptr::null_mut();
        // Single-input calls only. Prevent Rayon thread-pool creation and its
        // lifetime growth; no per-request environment mutation.
        tokenizers::utils::parallelism::set_parallelism(false);
        let mut tokenizer = Tokenizer::from_bytes(bytes(data, len)?).map_err(|e| e.to_string())?;
        let mut model = tokenizer.get_model().clone();
        if let tokenizers::models::ModelWrapper::BPE(ref mut bpe) = model {bpe.resize_cache(0);}
        else {return Err("native Mellum text requires BPE".into());}
        tokenizer.with_model(model);
        if tokenizer.get_vocab_size(true) != 98304 { return Err("unexpected Mellum vocabulary".into()); }
        *output = Box::into_raw(Box::new(tokenizer)) as *mut c_void;
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn mellum_tokenizer_destroy(handle: *mut c_void) {
    if !handle.is_null() { drop(Box::from_raw(handle as *mut Tokenizer)); }
}
#[no_mangle]
pub unsafe extern "C" fn mellum_tokenizer_encode(handle: *const c_void, data: *const u8, len: usize, output: *mut IdBuffer, error: *mut u8, size: usize) -> i32 {
    guarded(error, size, || {
        if output.is_null() { return Err("null ID output".into()); }
        *output = IdBuffer { data: ptr::null_mut(), len: 0, capacity: 0 };
        let text = std::str::from_utf8(bytes(data, len)?).map_err(|e| e.to_string())?;
        let encoded = instance(handle)?.encode(text, false).map_err(|e| e.to_string())?;
        let mut ids = encoded.get_ids().to_vec();
        *output = IdBuffer { data: ids.as_mut_ptr(), len: ids.len(), capacity: ids.capacity() };
        std::mem::forget(ids);
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn mellum_tokenizer_decode(handle: *const c_void, ids: *const u32, len: usize, skip_special: bool, output: *mut ByteBuffer, error: *mut u8, size: usize) -> i32 {
    guarded(error, size, || {
        if output.is_null() || ids.is_null() || len > 131072 { return Err("invalid/bounded ID input".into()); }
        *output = ByteBuffer { data: ptr::null_mut(), len: 0, capacity: 0 };
        let ids = slice::from_raw_parts(ids, len);
        if ids.iter().any(|id| *id >= 98304) { return Err("token ID outside Mellum vocabulary".into()); }
        let text = instance(handle)?.decode(ids, skip_special).map_err(|e| e.to_string())?;
        return_bytes(text.into_bytes(), output);
        Ok(())
    })
}
unsafe fn return_bytes(mut bytes: Vec<u8>, output: *mut ByteBuffer) {
    *output = ByteBuffer { data: bytes.as_mut_ptr(), len: bytes.len(), capacity: bytes.capacity() };
    std::mem::forget(bytes);
}
#[no_mangle]
pub unsafe extern "C" fn mellum_tokenizer_free_ids(buffer: IdBuffer) {
    if !buffer.data.is_null() { drop(Vec::from_raw_parts(buffer.data, buffer.len, buffer.capacity)); }
}
#[no_mangle]
pub unsafe extern "C" fn mellum_text_free_bytes(buffer: ByteBuffer) {
    if !buffer.data.is_null() { drop(Vec::from_raw_parts(buffer.data, buffer.len, buffer.capacity)); }
}

#[no_mangle]
pub unsafe extern "C" fn mellum_render_chat(data: *const u8, len: usize, output: *mut ByteBuffer, error: *mut u8, size: usize) -> i32 {
    guarded(error, size, || {
        if output.is_null() {return Err("null rendered output".into());}
        *output=ByteBuffer {data:ptr::null_mut(),len:0,capacity:0};
        let request:serde_json::Value=serde_json::from_slice(bytes(data,len)?).map_err(|e|e.to_string())?;
        let rendered=renderer::render(&request)?;
        return_bytes(rendered.into_bytes(),output);Ok(())
    })
}

#[cfg(test)]
mod tests;
