//! Chat prompt formatting with the model's own Jinja chat template (as stored in GGUF
//! metadata or `tokenizer_config.json`), rendered with MiniJinja configured like Hugging Face
//! transformers (`trim_blocks`, `lstrip_blocks`, Python string methods, `raise_exception`).

use anyhow::{Result, anyhow};
use minijinja::{Environment, ErrorKind, UndefinedBehavior, Value, context};

#[derive(Clone, Debug, PartialEq)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

impl TemplateMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Jinja(String),
    ChatMl,
    Plain,
}

#[derive(Clone, Debug)]
pub struct ChatTemplate {
    kind: Kind,
    pub bos: String,
    pub eos: String,
}

impl ChatTemplate {
    /// `source`: Jinja template text, if the model ships one. `has_chatml_tokens`: whether the
    /// vocabulary contains `<|im_start|>` (used to pick a fallback format).
    pub fn new(source: Option<String>, bos: &str, eos: &str, has_chatml_tokens: bool) -> Self {
        let kind = match source {
            Some(s) if !s.trim().is_empty() => Kind::Jinja(s),
            _ if has_chatml_tokens => Kind::ChatMl,
            _ => Kind::Plain,
        };
        Self { kind, bos: bos.into(), eos: eos.into() }
    }

    pub fn is_builtin_fallback(&self) -> bool {
        !matches!(self.kind, Kind::Jinja(_))
    }

    /// Text that marks the end of an assistant turn for fallback formats.
    pub fn fallback_stop(&self) -> Option<&'static str> {
        match self.kind {
            Kind::ChatMl => Some("<|im_end|>"),
            Kind::Plain => Some("\n### User:"),
            Kind::Jinja(_) => None,
        }
    }

    /// Renders the conversation into a prompt string.
    pub fn render(&self, messages: &[TemplateMessage], add_generation_prompt: bool) -> Result<String> {
        match &self.kind {
            Kind::ChatMl => {
                let mut s = String::new();
                for m in messages {
                    s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
                }
                if add_generation_prompt {
                    s.push_str("<|im_start|>assistant\n");
                }
                Ok(s)
            }
            Kind::Plain => {
                let mut s = String::new();
                for m in messages {
                    let who = match m.role.as_str() {
                        "system" => "System",
                        "assistant" => "Assistant",
                        _ => "User",
                    };
                    s.push_str(&format!("### {who}:\n{}\n\n", m.content));
                }
                if add_generation_prompt {
                    s.push_str("### Assistant:\n");
                }
                Ok(s)
            }
            Kind::Jinja(src) => match self.render_jinja(src, messages, add_generation_prompt) {
                Ok(s) => Ok(s),
                Err(e) if messages.first().map(|m| m.role == "system").unwrap_or(false) => {
                    // Some templates (e.g. Gemma 2) reject a system role: fold it into the
                    // first user message instead.
                    let merged = merge_system(messages);
                    self.render_jinja(src, &merged, add_generation_prompt)
                        .map_err(|e2| anyhow!("chat template failed: {e}; retry without system role failed: {e2}"))
                }
                Err(e) => Err(e),
            },
        }
    }

    fn render_jinja(&self, src: &str, messages: &[TemplateMessage], add_generation_prompt: bool) -> Result<String> {
        let mut env = Environment::new();
        let mut syntax = minijinja::syntax::SyntaxConfig::builder();
        syntax.trim_blocks(true).lstrip_blocks(true);
        env.set_syntax(syntax.build().map_err(|e| anyhow!("{e}"))?);
        env.set_undefined_behavior(UndefinedBehavior::Chainable);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String { chrono::Local::now().format(&fmt).to_string() });
        env.add_template_owned("chat".to_string(), src.to_string())
            .map_err(|e| anyhow!("invalid chat template: {e}"))?;
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                Value::from_pairs([("role", Value::from(m.role.clone())), ("content", Value::from(m.content.clone()))])
            })
            .collect();
        let tmpl = env.get_template("chat").map_err(|e| anyhow!("{e}"))?;
        let out = tmpl
            .render(context! {
                messages => msgs,
                add_generation_prompt => add_generation_prompt,
                bos_token => self.bos.clone(),
                eos_token => self.eos.clone(),
                date_string => chrono::Local::now().format("%d %b %Y").to_string(),
            })
            .map_err(|e| anyhow!("chat template error: {e}"))?;
        Ok(out)
    }
}

fn merge_system(messages: &[TemplateMessage]) -> Vec<TemplateMessage> {
    let mut out: Vec<TemplateMessage> = Vec::new();
    let mut system = String::new();
    for m in messages {
        if m.role == "system" {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&m.content);
            continue;
        }
        if !system.is_empty() && m.role == "user" {
            out.push(TemplateMessage::new("user", &format!("{system}\n\n{}", m.content)));
            system.clear();
        } else {
            out.push(m.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    const LLAMA3: &str = "{{- bos_token }}\n{%- for message in messages %}\n    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n";

    const GEMMA2: &str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}";

    #[test]
    fn chatml_template_renders() {
        let t = ChatTemplate::new(Some(QWEN.into()), "", "<|im_end|>", true);
        let s = t.render(&[TemplateMessage::new("user", "Hi")], true).unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn llama3_template_trims_and_uses_bos() {
        let t = ChatTemplate::new(Some(LLAMA3.into()), "<|begin_of_text|>", "<|eot_id|>", false);
        let s = t
            .render(&[TemplateMessage::new("system", "Be brief."), TemplateMessage::new("user", "  Hello  ")], true)
            .unwrap();
        assert_eq!(
            s,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn system_role_is_folded_when_rejected() {
        let t = ChatTemplate::new(Some(GEMMA2.into()), "<bos>", "<eos>", false);
        let s = t
            .render(
                &[
                    TemplateMessage::new("system", "Be brief."),
                    TemplateMessage::new("user", "Hello"),
                    TemplateMessage::new("assistant", "Hi!"),
                    TemplateMessage::new("user", "Bye"),
                ],
                true,
            )
            .unwrap();
        assert!(s.starts_with("<bos><start_of_turn>user\nBe brief.\n\nHello<end_of_turn>\n"));
        assert!(s.contains("<start_of_turn>model\nHi!<end_of_turn>"));
        assert!(s.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn python_string_methods_work() {
        let src = "{% for m in messages %}{{ m.content.strip().upper() }}{% endfor %}";
        let t = ChatTemplate::new(Some(src.into()), "", "", false);
        assert_eq!(t.render(&[TemplateMessage::new("user", " hi ")], false).unwrap(), "HI");
    }

    #[test]
    fn fallbacks() {
        let t = ChatTemplate::new(None, "", "", true);
        assert!(t.is_builtin_fallback());
        assert!(t.render(&[TemplateMessage::new("user", "x")], true).unwrap().ends_with("<|im_start|>assistant\n"));
        let p = ChatTemplate::new(None, "", "", false);
        assert!(p.render(&[TemplateMessage::new("user", "x")], true).unwrap().ends_with("### Assistant:\n"));
    }
}
