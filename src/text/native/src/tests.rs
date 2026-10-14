use super::*;
use serde_json::json;

#[test]
fn bounded_abi_inputs_and_errors() {
    unsafe {
        assert!(bytes(ptr::null(), 0).unwrap().is_empty());
        assert!(bytes(ptr::null(), 1).is_err());
        assert!(bytes(ptr::null(), 32 * 1024 * 1024 + 1).is_err());
        let mut error = [0xaa; 8];
        assert_eq!(guarded(error.as_mut_ptr(), error.len(), || Err("long rejection".into())), 1);
        assert_eq!(&error, b"long re\0");
        let mut output = IdBuffer { data: ptr::null_mut(), len: 99, capacity: 99 };
        assert_eq!(mellum_tokenizer_encode(ptr::null(), b"x".as_ptr(), 1,
                   &mut output, error.as_mut_ptr(), error.len()), 1);
        assert!(output.data.is_null());
        assert_eq!(output.len, 0);
    }
}

#[test]
fn native_bpe_abi_roundtrip_and_vocabulary_validation() {
    let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let mut config: serde_json::Value = serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
    let small = serde_json::to_vec(&config).unwrap();
    let mut handle = ptr::null_mut();
    let mut error = [0; 256];
    unsafe {
        assert_eq!(mellum_tokenizer_create(small.as_ptr(), small.len(), &mut handle,
                   error.as_mut_ptr(), error.len()), 1);
        assert!(handle.is_null());
    }
    let mut vocab = serde_json::Map::new();
    vocab.insert("a".into(), json!(0));
    vocab.insert("b".into(), json!(1));
    for id in 2..98304 {
        vocab.insert(format!("v{id}"), json!(id));
    }
    config["model"]["vocab"] = serde_json::Value::Object(vocab);
    let encoded_config = serde_json::to_vec(&config).unwrap();
    struct Handle(*mut c_void);
    impl Drop for Handle {
        fn drop(&mut self) { unsafe { mellum_tokenizer_destroy(self.0) }; }
    }
    unsafe {
        assert_eq!(mellum_tokenizer_create(encoded_config.as_ptr(), encoded_config.len(),
                   &mut handle, error.as_mut_ptr(), error.len()), 0);
        let handle = Handle(handle);
        let mut ids = IdBuffer { data: ptr::null_mut(), len: 0, capacity: 0 };
        assert_eq!(mellum_tokenizer_encode(handle.0, b"ab".as_ptr(), 2, &mut ids,
                   error.as_mut_ptr(), error.len()), 0);
        assert_eq!(slice::from_raw_parts(ids.data, ids.len), &[0, 1]);
        let mut decoded = ByteBuffer { data: ptr::null_mut(), len: 0, capacity: 0 };
        assert_eq!(mellum_tokenizer_decode(handle.0, ids.data, ids.len, true, &mut decoded,
                   error.as_mut_ptr(), error.len()), 0);
        assert_eq!(slice::from_raw_parts(decoded.data, decoded.len), b"a b");
        mellum_tokenizer_free_ids(ids);
        mellum_text_free_bytes(decoded);
        assert_eq!(mellum_tokenizer_encode(handle.0, [0xff].as_ptr(), 1, &mut
                   IdBuffer { data: ptr::null_mut(), len: 0, capacity: 0 },
                   error.as_mut_ptr(), error.len()), 1);
    }
}

#[test]
fn render_plain_text_and_thinking_control() {
    assert_eq!(renderer::render(&json!({"messages":[{"role":"user","content":"€ hello"}],
                                      "enable_thinking":false})).unwrap(),
               "<|im_start|>user\n€ hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

#[test]
fn renderer_rejects_nontext_and_unknown_roles() {
    for message in [json!({"role":"developer","content":"x"}),
                    json!({"role":"user","content":[{"type":"image_url","image_url":"x"}]})] {
        assert!(renderer::render(&json!({"messages":[message]})).is_err());
    }
    assert!(renderer::render(&json!({"messages":[]})).is_err());
}

#[test]
fn renderer_retains_raw_tool_arguments_and_groups_results() {
    let raw = "{\"z\": 1, \"a\": [2,3]}";
    let result = renderer::render(&json!({"messages":[
        {"role":"user","content":"lookup"},
        {"role":"assistant","content":null,"tool_calls":[{"function":{"name":"lookup","arguments":raw}}]},
        {"role":"tool","content":"first"}, {"role":"tool","content":"second"}],
        "add_generation_prompt":false})).unwrap();
    assert!(result.contains(&format!("\"arguments\": {raw}")));
    assert!(result.ends_with("<|im_start|>user\n<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>\n"));
}

#[test]
fn native_renderer_abi_owns_and_frees_output() {
    let request = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
    unsafe {
        let mut out = ByteBuffer { data: ptr::null_mut(), len: 0, capacity: 0 };
        let mut error = [0; 128];
        assert_eq!(mellum_render_chat(request.as_ptr(), request.len(), &mut out,
                   error.as_mut_ptr(), error.len()), 0);
        assert!(std::str::from_utf8(slice::from_raw_parts(out.data, out.len)).unwrap().contains("hello"));
        mellum_text_free_bytes(out);
    }
}
