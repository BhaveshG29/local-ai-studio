//! Generation parameters and which of them each engine supports.

use serde::{Deserialize, Serialize};

use crate::config::GenerationDefaults;

/// Sampling and length settings for one conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub repeat_penalty: f32,
    pub context_length: u32,
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self::from_defaults(&GenerationDefaults::default())
    }
}

impl GenParams {
    pub fn from_defaults(d: &GenerationDefaults) -> Self {
        Self {
            temperature: d.temperature,
            top_p: d.top_p,
            top_k: d.top_k,
            max_tokens: d.max_tokens,
            repeat_penalty: d.repeat_penalty,
            context_length: d.context_length,
            seed: d.seed,
        }
    }

    /// Clamps every value into a safe range.
    pub fn sanitized(mut self) -> Self {
        use crate::util::clamp_f32;
        self.temperature = clamp_f32(self.temperature, 0.0, 2.0);
        self.top_p = clamp_f32(self.top_p, 0.0, 1.0);
        self.top_k = self.top_k.min(1000);
        // 0 = no limit (until the model stops or the context window is full).
        self.max_tokens = self.max_tokens.min(262_144);
        self.repeat_penalty = clamp_f32(self.repeat_penalty, 0.5, 2.0);
        self.context_length = self.context_length.clamp(256, 262_144);
        self
    }
}

/// Which generation settings an engine honours (shown in the chat settings panel).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ParamSupport {
    pub temperature: bool,
    pub top_p: bool,
    pub top_k: bool,
    pub max_tokens: bool,
    pub repeat_penalty: bool,
    pub context_length: bool,
    pub seed: bool,
    pub images: bool,
    pub token_usage: bool,
}

impl ParamSupport {
    pub const ALL_TEXT: ParamSupport = ParamSupport {
        temperature: true,
        top_p: true,
        top_k: true,
        max_tokens: true,
        repeat_penalty: true,
        context_length: true,
        seed: true,
        images: false,
        token_usage: true,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_are_sanitized() {
        let p = GenParams {
            temperature: f32::NAN,
            top_p: 3.0,
            top_k: 5000,
            max_tokens: 999_999,
            repeat_penalty: 9.0,
            context_length: 1,
            seed: Some(1),
        }
        .sanitized();
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 1000);
        assert_eq!(p.max_tokens, 262_144);
        assert_eq!(GenParams { max_tokens: 0, ..GenParams::default() }.sanitized().max_tokens, 0, "no limit");
        assert_eq!(p.repeat_penalty, 2.0);
        assert_eq!(p.context_length, 256);
    }

    #[test]
    fn partial_json_uses_defaults() {
        let p: GenParams = serde_json::from_str(r#"{"temperature":0.1}"#).unwrap();
        assert_eq!(p.temperature, 0.1);
        assert_eq!(p.max_tokens, GenParams::default().max_tokens);
    }
}
