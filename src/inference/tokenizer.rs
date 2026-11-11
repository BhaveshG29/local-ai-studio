//! Tokenizers for the built-in engine.
//!
//! * GGUF files with a `gpt2` (byte-level BPE) vocabulary use Candle's GGUF → `tokenizers`
//!   conversion.
//! * GGUF files with a `llama` (SentencePiece) vocabulary use [`SpmTokenizer`], a faithful
//!   port of llama.cpp's SPM algorithm: special tokens are split out first, then characters
//!   are merged greedily by vocabulary score, with `<0xXX>` byte fallback.
//! * Safetensors models use their `tokenizer.json`.

use std::collections::{BinaryHeap, HashMap};

use anyhow::{Context, Result, anyhow};
use candle_core::quantized::gguf_file;

const SPACE: char = '\u{2581}';

/// llama.cpp token types.
const TOKEN_TYPE_CONTROL: u32 = 3;
const TOKEN_TYPE_USER_DEFINED: u32 = 4;
const TOKEN_TYPE_BYTE: u32 = 6;

pub struct SpmTokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    types: Vec<u32>,
    lookup: HashMap<String, u32>,
    /// Special tokens sorted longest first (matched verbatim in prompts).
    specials: Vec<(String, u32)>,
    byte_tokens: [Option<u32>; 256],
    unk_id: Option<u32>,
    add_space_prefix: bool,
}

#[derive(PartialEq)]
struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Highest score first; on ties, the leftmost pair first.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct Symbol {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl SpmTokenizer {
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        types: Vec<u32>,
        unk_id: Option<u32>,
        add_space_prefix: bool,
    ) -> Self {
        let mut lookup = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            lookup.entry(t.clone()).or_insert(i as u32);
        }
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                !t.is_empty() && matches!(types.get(*i), Some(&TOKEN_TYPE_CONTROL) | Some(&TOKEN_TYPE_USER_DEFINED))
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let mut byte_tokens = [None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = lookup.get(&format!("<0x{b:02X}>")).copied();
        }
        Self { tokens, scores, types, lookup, specials, byte_tokens, unk_id, add_space_prefix }
    }

    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let md = &ct.metadata;
        let tokens: Vec<String> = md
            .get("tokenizer.ggml.tokens")
            .context("missing tokenizer.ggml.tokens")?
            .to_vec()?
            .iter()
            .map(|v| v.to_string().cloned().map_err(|e| anyhow!("{e}")))
            .collect::<Result<_>>()?;
        let scores: Vec<f32> = match md.get("tokenizer.ggml.scores") {
            Some(v) => v.to_vec()?.iter().map(|x| x.to_f32().unwrap_or(0.0)).collect(),
            None => vec![0.0; tokens.len()],
        };
        let types: Vec<u32> = match md.get("tokenizer.ggml.token_type") {
            Some(v) => {
                v.to_vec()?.iter().map(|x| x.to_i32().map(|i| i as u32).or_else(|_| x.to_u32()).unwrap_or(1)).collect()
            }
            None => vec![1; tokens.len()],
        };
        let unk = md.get("tokenizer.ggml.unknown_token_id").and_then(|v| v.to_u32().ok());
        let add_space_prefix = md.get("tokenizer.ggml.add_space_prefix").and_then(|v| v.to_bool().ok()).unwrap_or(true);
        Ok(Self::new(tokens, scores, types, unk, add_space_prefix))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.lookup.get(s).copied()
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    fn is_special(&self, id: u32) -> bool {
        matches!(self.types.get(id as usize), Some(&TOKEN_TYPE_CONTROL) | Some(&TOKEN_TYPE_USER_DEFINED))
    }

    /// Tokenizes text; special token strings in the text map to their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut is_prev_special = true;
        let mut raw_start = 0;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let mut matched = None;
            if !self.specials.is_empty() {
                for (s, id) in &self.specials {
                    if bytes[i..].starts_with(s.as_bytes()) {
                        matched = Some((s.len(), *id));
                        break;
                    }
                }
            }
            if let Some((len, id)) = matched {
                if raw_start < i {
                    self.encode_raw(&text[raw_start..i], is_prev_special, &mut out);
                }
                out.push(id);
                is_prev_special = true;
                i += len;
                raw_start = i;
            } else {
                // advance one UTF-8 character
                let ch_len = text[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                i += ch_len;
            }
        }
        if raw_start < text.len() {
            self.encode_raw(&text[raw_start..], is_prev_special, &mut out);
        }
        out
    }

    fn encode_raw(&self, raw: &str, is_prev_special: bool, out: &mut Vec<u32>) {
        let mut text = String::with_capacity(raw.len() + 3);
        if self.add_space_prefix && is_prev_special {
            text.push(SPACE);
        }
        for c in raw.chars() {
            text.push(if c == ' ' { SPACE } else { c });
        }
        // Initial symbols: one per UTF-8 character.
        let mut symbols: Vec<Symbol> = Vec::new();
        for (idx, (start, c)) in text.char_indices().enumerate() {
            symbols.push(Symbol { start, len: c.len_utf8(), prev: idx.checked_sub(1), next: None });
        }
        let n = symbols.len();
        for (i, s) in symbols.iter_mut().enumerate() {
            s.next = (i + 1 < n).then_some(i + 1);
        }
        let mut heap = BinaryHeap::new();
        let try_add = |heap: &mut BinaryHeap<Bigram>, symbols: &[Symbol], l: Option<usize>, r: Option<usize>| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let s = &text[symbols[l].start..symbols[r].start + symbols[r].len];
            if let Some(&id) = self.lookup.get(s) {
                heap.push(Bigram {
                    score: self.scores.get(id as usize).copied().unwrap_or(0.0),
                    left: l,
                    right: r,
                    size: s.len(),
                });
            }
        };
        for i in 1..n {
            try_add(&mut heap, &symbols, Some(i - 1), Some(i));
        }
        while let Some(b) = heap.pop() {
            let (l, r) = (b.left, b.right);
            if symbols[l].len == 0 || symbols[r].len == 0 || symbols[l].len + symbols[r].len != b.size {
                continue;
            }
            symbols[l].len += symbols[r].len;
            symbols[r].len = 0;
            symbols[l].next = symbols[r].next;
            if let Some(nn) = symbols[r].next {
                symbols[nn].prev = Some(l);
            }
            let prev = symbols[l].prev;
            let next = symbols[l].next;
            try_add(&mut heap, &symbols, prev, Some(l));
            try_add(&mut heap, &symbols, Some(l), next);
        }
        let mut idx = if n > 0 { Some(0) } else { None };
        while let Some(i) = idx {
            let sym = &symbols[i];
            if sym.len > 0 {
                let s = &text[sym.start..sym.start + sym.len];
                match self.lookup.get(s) {
                    Some(&id) => out.push(id),
                    None => {
                        for b in s.bytes() {
                            if let Some(id) = self.byte_tokens[b as usize].or(self.unk_id) {
                                out.push(id)
                            }
                        }
                    }
                }
            }
            idx = sym.next;
        }
    }

    /// Bytes represented by one token (`<0xXX>` byte tokens become raw bytes; `▁` → space).
    pub fn token_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else {
            return Vec::new();
        };
        if self.types.get(id as usize) == Some(&TOKEN_TYPE_BYTE)
            && let Some(hex) = t.strip_prefix("<0x").and_then(|h| h.strip_suffix('>'))
            && let Ok(b) = u8::from_str_radix(hex, 16)
        {
            return vec![b];
        }
        if skip_special && self.is_special(id) {
            return Vec::new();
        }
        t.replace(SPACE, " ").into_bytes()
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id, skip_special));
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

/// Unified tokenizer interface used by the generation loop.
pub enum TextTokenizer {
    Hf(Box<tokenizers::Tokenizer>),
    Spm(Box<SpmTokenizer>),
}

impl TextTokenizer {
    /// Builds the tokenizer embedded in a GGUF file.
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        use candle_core::quantized::tokenizer::TokenizerFromGguf;
        let kind =
            ct.metadata.get("tokenizer.ggml.model").and_then(|v| v.to_string().ok()).cloned().unwrap_or_default();
        match kind.as_str() {
            "gpt2" => {
                Ok(TextTokenizer::Hf(Box::new(tokenizers::Tokenizer::from_gguf(ct).map_err(|e| anyhow!("{e}"))?)))
            }
            "llama" => Ok(TextTokenizer::Spm(Box::new(SpmTokenizer::from_gguf(ct)?))),
            other => Err(anyhow!("unsupported tokenizer type \"{other}\"")),
        }
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let t =
            tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("could not load {}: {e}", path.display()))?;
        Ok(TextTokenizer::Hf(Box::new(t)))
    }

    /// Encodes a fully formatted prompt. Special-token strings are recognised; no BOS/EOS is
    /// added automatically (chat templates insert them).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            TextTokenizer::Hf(t) => {
                Ok(t.encode(text, false).map_err(|e| anyhow!("tokenization failed: {e}"))?.get_ids().to_vec())
            }
            TextTokenizer::Spm(t) => Ok(t.encode(text)),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            TextTokenizer::Hf(t) => t.decode(ids, true).map_err(|e| anyhow!("decoding failed: {e}")),
            TextTokenizer::Spm(t) => Ok(t.decode(ids, true)),
        }
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        match self {
            TextTokenizer::Hf(t) => t.token_to_id(s),
            TextTokenizer::Spm(t) => t.token_id(s),
        }
    }

    pub fn token_str(&self, id: u32) -> Option<String> {
        match self {
            TextTokenizer::Hf(t) => t.id_to_token(id),
            TextTokenizer::Spm(t) => t.token_str(id).map(String::from),
        }
    }
}

/// Incremental decoder that only emits complete UTF-8 text.
pub struct StreamDecoder {
    ids: Vec<u32>,
    emitted: usize,
    window_start: usize,
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self { ids: Vec::new(), emitted: 0, window_start: 0 }
    }

    /// Adds a token and returns newly completed text, if any.
    pub fn push(&mut self, tok: &TextTokenizer, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let text = tok.decode(&self.ids[self.window_start..])?;
        if text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        if text.len() <= self.emitted {
            return Ok(None);
        }
        let new = text.get(self.emitted..).map(String::from).unwrap_or_default();
        self.emitted = text.len();
        // Keep the decoding window short so each step stays cheap; tokenizers that strip a
        // leading space need a little context, so restart the window a few tokens back.
        if self.ids.len() - self.window_start > 64 {
            let keep = 4;
            self.window_start = self.ids.len() - keep;
            self.emitted = tok.decode(&self.ids[self.window_start..])?.len();
        }
        Ok(Some(new))
    }

    /// Any remaining text (e.g. a trailing incomplete character rendered as U+FFFD).
    pub fn finish(&mut self, tok: &TextTokenizer) -> Result<Option<String>> {
        let text = tok.decode(&self.ids[self.window_start..])?;
        if text.len() > self.emitted {
            let rest = text.get(self.emitted..).map(String::from);
            self.emitted = text.len();
            Ok(rest)
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny SentencePiece-style vocabulary.
    fn toy() -> SpmTokenizer {
        let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let mut types = vec![2u32, 3, 3];
        for b in 0..=255u8 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(TOKEN_TYPE_BYTE);
        }
        let words = [
            ("▁", -1.0),
            ("h", -2.0),
            ("e", -2.0),
            ("l", -2.0),
            ("o", -2.0),
            ("w", -2.0),
            ("r", -2.0),
            ("d", -2.0),
            ("he", -3.0),
            ("ll", -3.0),
            ("hell", -4.0),
            ("hello", -5.0),
            ("▁hello", -6.0),
            ("wo", -3.0),
            ("rl", -3.0),
            ("worl", -5.0),
            ("world", -5.5),
            ("▁world", -7.0),
            ("<start_of_turn>", 0.0),
        ];
        let mut scores = vec![0.0f32; tokens.len()];
        for (w, s) in words {
            tokens.push(w.to_string());
            scores.push(s);
            types.push(if w.starts_with('<') { TOKEN_TYPE_CONTROL } else { 1 });
        }
        SpmTokenizer::new(tokens, scores, types, Some(0), true)
    }

    #[test]
    fn spm_merges_by_score_and_round_trips() {
        let t = toy();
        let ids = t.encode("hello world");
        let pieces: Vec<&str> = ids.iter().map(|i| t.token_str(*i).unwrap()).collect();
        assert_eq!(pieces, vec!["▁hello", "▁world"]);
        assert_eq!(t.decode(&ids, true), " hello world");
    }

    #[test]
    fn spm_specials_and_byte_fallback() {
        let t = toy();
        let ids = t.encode("<start_of_turn>hello é");
        assert_eq!(t.token_str(ids[0]), Some("<start_of_turn>"));
        assert_eq!(t.token_str(ids[1]), Some("▁hello"));
        // "é" is not in the vocabulary: two UTF-8 byte tokens.
        let tail: Vec<&str> = ids[ids.len() - 2..].iter().map(|i| t.token_str(*i).unwrap()).collect();
        assert_eq!(tail, vec!["<0xC3>", "<0xA9>"]);
        assert_eq!(t.decode(&ids, true), " hello é");
        // special tokens are skipped when decoding
        assert!(!t.decode(&ids, true).contains("start_of_turn"));
    }

    #[test]
    fn stream_decoder_waits_for_complete_characters() {
        let t = TextTokenizer::Spm(Box::new(toy()));
        let TextTokenizer::Spm(inner) = &t else { unreachable!() };
        let c3 = inner.token_id("<0xC3>").unwrap();
        let a9 = inner.token_id("<0xA9>").unwrap();
        let hello = inner.token_id("▁hello").unwrap();
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, hello).unwrap().as_deref(), Some(" hello"));
        assert_eq!(d.push(&t, c3).unwrap(), None);
        assert_eq!(d.push(&t, a9).unwrap().as_deref(), Some("é"));
        assert_eq!(d.finish(&t).unwrap(), None);
    }
}
