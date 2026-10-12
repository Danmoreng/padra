// Model-specific translation of the pinned Mellum chat_template.jinja.
// HF/Transformers JSON formatting, not Jinja's default HTML-escaped tojson.
use serde_json::Value;

pub fn python_json(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        Value::Number(n) => {
            if n.is_f64() {
                let repr = format!("{:?}", n.as_f64().unwrap());
                if let Some((mantissa, exponent)) = repr.split_once('e') {
                    let exponent: i32 = exponent.parse().unwrap();
                    format!("{}e{}{:02}", mantissa, if exponent < 0 { '-' } else { '+' }, exponent.abs())
                } else { repr }
            } else { n.to_string() }
        }
        Value::Array(a) => format!("[{}]", a.iter().map(python_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k,v)| format!("{}: {}", serde_json::to_string(k).unwrap(), python_json(v))).collect::<Vec<_>>().join(", ")),
    }
}
fn normalized(content: Option<&Value>) -> Result<String, String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut result = String::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).ok_or("invalid content part")?;
                let text = match kind {
                    "text" => part.get("text"),
                    "tool-result" if part.get("output").and_then(|v|v.get("type")).and_then(Value::as_str)==Some("text") => part.get("output").and_then(|v|v.get("value")),
                    _ => return Err("unsupported nontext content; nothing may be silently dropped".into()),
                };
                match text { None | Some(Value::Null) => {}, Some(Value::String(text))=>result.push_str(text), _=>return Err("content text must be a string".into()) }
            }
            Ok(result)
        }
        _ => Err("content must be text or supported text parts".into()),
    }
}
fn array<'a>(value: Option<&'a Value>, name: &str) -> Result<&'a [Value], String> {
    match value { None | Some(Value::Null)=>Ok(&[]),Some(Value::Array(a))=>Ok(a),_=>Err(format!("{name} must be an array")) }
}
pub fn render(request: &Value) -> Result<String, String> {
    let messages = array(request.get("messages"), "messages")?;
    if messages.is_empty() { return Err("at least one message required".into()); }
    let tools = array(request.get("tools"), "tools")?;
    let thinking = request.get("enable_thinking").map(|v|v.as_bool().ok_or("enable_thinking must be boolean")).transpose()?.unwrap_or(true);
    let generation = request.get("add_generation_prompt").map(|v|v.as_bool().ok_or("add_generation_prompt must be boolean")).transpose()?.unwrap_or(true);
    let mut roles = Vec::with_capacity(messages.len());
    let mut contents = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("missing/invalid role")?;
        if !["system","user","assistant","tool"].contains(&role) { return Err("unsupported role; no silent template omission".into()); }
        roles.push(role);contents.push(normalized(message.get("content"))?);
    }
    let mut out=String::new();
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if roles[0]=="system" {out.push_str(&contents[0]);out.push_str("\n\n");}
        out.push_str("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for tool in tools {out.push('\n');out.push_str(&python_json(tool));}
        out.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n");
    } else if roles[0]=="system" {out.push_str("<|im_start|>system\n");out.push_str(&contents[0]);out.push_str("<|im_end|>\n");}
    let mut last_query=messages.len()-1;
    for i in (0..messages.len()).rev() {
        if roles[i]=="user" && !(contents[i].starts_with("<tool_response>") && contents[i].ends_with("</tool_response>")) {last_query=i;break;}
    }
    for (i,message) in messages.iter().enumerate() {
        let role=roles[i];let mut content=contents[i].clone();
        if role=="user" || (role=="system" && i!=0) {
            out.push_str("<|im_start|>");out.push_str(role);out.push('\n');out.push_str(&content);out.push_str("<|im_end|>\n");
        } else if role=="assistant" {
            let reasoning=if let Some(text)=message.get("reasoning_content").and_then(Value::as_str) {text.to_owned()}
            else if content.contains("</think>") {
                let prefix=content.split("</think>").next().unwrap().trim_end_matches('\n');
                let reason=prefix.rsplit("<think>").next().unwrap().trim_start_matches('\n').to_owned();
                content=content.rsplit("</think>").next().unwrap().trim_start_matches('\n').to_owned();reason
            } else {String::new()};
            out.push_str("<|im_start|>assistant\n");
            if i>last_query && !reasoning.is_empty() {out.push_str("<think>\n");out.push_str(reasoning.trim_matches('\n'));out.push_str("\n</think>\n\n");out.push_str(content.trim_start_matches('\n'));}
            else {out.push_str(&content);}
            for (index,call) in array(message.get("tool_calls"), "tool_calls")?.iter().enumerate() {
                if index!=0 || !content.is_empty() {out.push('\n');}
                let function=call.get("function").filter(|f|!f.is_null()).unwrap_or(call);
                let name=function.get("name").and_then(Value::as_str).ok_or("tool name must be string")?;
                if name.is_empty() || name.len()>64 || !name.bytes().all(|c|c.is_ascii_alphanumeric() || c==b'_' || c==b'-') {return Err("invalid function name".into());}
                out.push_str("<tool_call>\n{\"name\": \"");out.push_str(name);out.push_str("\", \"arguments\": ");
                match function.get("arguments") { Some(Value::String(args))=>out.push_str(args),Some(args)=>out.push_str(&python_json(args)),None=>return Err("tool arguments required".into()) }
                out.push_str("}\n</tool_call>");
            }
            out.push_str("<|im_end|>\n");
        } else if role=="tool" {
            if i==0 || roles[i-1]!="tool" {out.push_str("<|im_start|>user");}
            out.push_str("\n<tool_response>\n");out.push_str(&content);out.push_str("\n</tool_response>");
            if i+1==messages.len() || roles[i+1]!="tool" {out.push_str("<|im_end|>\n");}
        }
    }
    if generation {out.push_str("<|im_start|>assistant\n");if !thinking {out.push_str("<think>\n\n</think>\n\n");}}
    Ok(out)
}
