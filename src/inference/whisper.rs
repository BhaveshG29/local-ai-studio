//! Speech-to-text with OpenAI Whisper models running on Candle.
//!
//! Audio is decoded and resampled to 16 kHz mono ([`crate::media`]), converted to a log-mel
//! spectrogram (mel filter bank computed here, identical to `librosa.filters.mel` with Slaney
//! normalisation as used by Whisper), and decoded greedily in 30-second windows with
//! temperature fallback for low-confidence windows. Timestamps are per 30-second window.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::ops::{softmax, softmax_last_dim};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::whisper::model::{AudioEncoder, Whisper};
use candle_transformers::models::whisper::{self as w, audio};
use tokio_util::sync::CancellationToken;

use crate::db::jobs::TranscriptSegment;
use crate::db::models::ModelRow;
use crate::error::{AppError, AppResult};

/// Whisper language codes in token order.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("ca", "Catalan"),
    ("nl", "Dutch"),
    ("ar", "Arabic"),
    ("sv", "Swedish"),
    ("it", "Italian"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("fi", "Finnish"),
    ("vi", "Vietnamese"),
    ("he", "Hebrew"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
    ("ms", "Malay"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("da", "Danish"),
    ("hu", "Hungarian"),
    ("ta", "Tamil"),
    ("no", "Norwegian"),
    ("th", "Thai"),
    ("ur", "Urdu"),
    ("hr", "Croatian"),
    ("bg", "Bulgarian"),
    ("lt", "Lithuanian"),
    ("la", "Latin"),
    ("mi", "Maori"),
    ("ml", "Malayalam"),
    ("cy", "Welsh"),
    ("sk", "Slovak"),
    ("te", "Telugu"),
    ("fa", "Persian"),
    ("lv", "Latvian"),
    ("bn", "Bengali"),
    ("sr", "Serbian"),
    ("az", "Azerbaijani"),
    ("sl", "Slovenian"),
    ("kn", "Kannada"),
    ("et", "Estonian"),
    ("mk", "Macedonian"),
    ("br", "Breton"),
    ("eu", "Basque"),
    ("is", "Icelandic"),
    ("hy", "Armenian"),
    ("ne", "Nepali"),
    ("mn", "Mongolian"),
    ("bs", "Bosnian"),
    ("kk", "Kazakh"),
    ("sq", "Albanian"),
    ("sw", "Swahili"),
    ("gl", "Galician"),
    ("mr", "Marathi"),
    ("pa", "Punjabi"),
    ("si", "Sinhala"),
    ("km", "Khmer"),
    ("sn", "Shona"),
    ("yo", "Yoruba"),
    ("so", "Somali"),
    ("af", "Afrikaans"),
    ("oc", "Occitan"),
    ("ka", "Georgian"),
    ("be", "Belarusian"),
    ("tg", "Tajik"),
    ("sd", "Sindhi"),
    ("gu", "Gujarati"),
    ("am", "Amharic"),
    ("yi", "Yiddish"),
    ("lo", "Lao"),
    ("uz", "Uzbek"),
    ("fo", "Faroese"),
    ("ht", "Haitian Creole"),
    ("ps", "Pashto"),
    ("tk", "Turkmen"),
    ("nn", "Nynorsk"),
    ("mt", "Maltese"),
    ("sa", "Sanskrit"),
    ("lb", "Luxembourgish"),
    ("my", "Myanmar"),
    ("bo", "Tibetan"),
    ("tl", "Tagalog"),
    ("mg", "Malagasy"),
    ("as", "Assamese"),
    ("tt", "Tatar"),
    ("haw", "Hawaiian"),
    ("ln", "Lingala"),
    ("ha", "Hausa"),
    ("ba", "Bashkir"),
    ("jw", "Javanese"),
    ("su", "Sundanese"),
    ("yue", "Cantonese"),
];

pub fn language_name(code: &str) -> &'static str {
    LANGUAGES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n).unwrap_or("Unknown")
}

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m }
}

/// Mel filter bank `[n_mels × (n_fft/2 + 1)]`, row-major, matching
/// `librosa.filters.mel(sr, n_fft, n_mels=n_mels)` (Slaney scale and normalisation).
pub fn mel_filters(sr: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let fmax = sr as f64 / 2.0;
    let fft_freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * fmax / (n_freqs - 1) as f64).collect();
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(fmax);
    let mel_f: Vec<f64> =
        (0..n_mels + 2).map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64)).collect();
    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();
    let mut weights = vec![0f32; n_mels * n_freqs];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (j, &f) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[i] - f) / fdiff[i];
            let upper = (mel_f[i + 2] - f) / fdiff[i + 1];
            let v = lower.min(upper).max(0.0);
            weights[i * n_freqs + j] = (v * enorm) as f32;
        }
    }
    weights
}

fn layer_norm(d: usize, vb: VarBuilder) -> candle_core::Result<LayerNorm> {
    Ok(LayerNorm::new(vb.get(d, "weight")?, vb.get(d, "bias")?, 1e-5))
}

/// One attention layer of the text decoder (self-attention or attention to the audio).
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}

impl Attention {
    fn load(d: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            query: candle_nn::linear(d, d, vb.pp("q_proj"))?,
            key: candle_nn::linear_no_bias(d, d, vb.pp("k_proj"))?,
            value: candle_nn::linear(d, d, vb.pp("v_proj"))?,
            out: candle_nn::linear(d, d, vb.pp("out_proj"))?,
        })
    }
}

struct DecoderBlock {
    attn: Attention,
    attn_ln: LayerNorm,
    cross: Attention,
    cross_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    mlp_ln: LayerNorm,
}

/// The audio's keys (transposed and pre-scaled) and values for every decoder block, worked
/// out once per 30-second window.
struct AudioKv(Vec<(Tensor, Tensor)>);

/// Keys and values of the tokens decoded so far, so each new token only computes itself.
#[derive(Clone)]
struct TokenCache {
    kv: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

/// Whisper's text decoder (same weights and arithmetic as Candle's `TextDecoder`), with the
/// caches Candle's version doesn't have: it recomputes every earlier token and copies the
/// audio's keys and values for each new token, which on a laptop processor took about 0.16 s
/// per token (a sentence took longer to transcribe than to say). With the caches a token
/// takes 10 to 25 ms, half of it for the scores over the 51,865-token vocabulary.
struct TextDecoder {
    token_embedding: Tensor,
    positional_embedding: Tensor,
    blocks: Vec<DecoderBlock>,
    ln: LayerNorm,
    n_head: usize,
}

impl TextDecoder {
    fn load(vb: VarBuilder, cfg: &w::Config) -> candle_core::Result<Self> {
        let d = cfg.d_model;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                let vb = vb.pp(format!("layers.{i}"));
                Ok(DecoderBlock {
                    attn: Attention::load(d, vb.pp("self_attn"))?,
                    attn_ln: layer_norm(d, vb.pp("self_attn_layer_norm"))?,
                    cross: Attention::load(d, vb.pp("encoder_attn"))?,
                    cross_ln: layer_norm(d, vb.pp("encoder_attn_layer_norm"))?,
                    fc1: candle_nn::linear(d, 4 * d, vb.pp("fc1"))?,
                    fc2: candle_nn::linear(4 * d, d, vb.pp("fc2"))?,
                    mlp_ln: layer_norm(d, vb.pp("final_layer_norm"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: vb.get((cfg.vocab_size, d), "embed_tokens.weight")?,
            positional_embedding: vb.get((cfg.max_target_positions, d), "embed_positions.weight")?,
            blocks,
            ln: layer_norm(d, vb.pp("layer_norm"))?,
            n_head: cfg.decoder_attention_heads,
        })
    }

    /// `(1, n, d)` → `(1, heads, n, head size)`.
    fn heads(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        x.reshape((b, n, self.n_head, d / self.n_head))?.transpose(1, 2)
    }

    fn audio_kv(&self, features: &Tensor) -> candle_core::Result<AudioKv> {
        let (_, _, d) = features.dims3()?;
        let scale = ((d / self.n_head) as f64).powf(-0.5);
        let kv = self
            .blocks
            .iter()
            .map(|b| {
                let k = (self.heads(&b.cross.key.forward(features)?)?.transpose(2, 3)?.contiguous()? * scale)?;
                let v = self.heads(&b.cross.value.forward(features)?)?.contiguous()?;
                Ok((k, v))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(AudioKv(kv))
    }

    fn empty_cache(&self) -> TokenCache {
        TokenCache { kv: vec![None; self.blocks.len()], len: 0 }
    }

    /// Runs new tokens through the decoder; returns the final hidden states `(1, n, d)`.
    fn forward(&self, tokens: &[u32], audio: &AudioKv, cache: &mut TokenCache) -> candle_core::Result<Tensor> {
        let device = self.token_embedding.device();
        let (n, pos) = (tokens.len(), cache.len);
        let ids = Tensor::new(tokens, device)?;
        let mut x = (self.token_embedding.index_select(&ids, 0)? + self.positional_embedding.narrow(0, pos, n)?)?
            .unsqueeze(0)?;
        let d = x.dim(2)?;
        let scale = ((d / self.n_head) as f64).powf(-0.5);
        // New tokens see every earlier token and themselves, never later ones.
        let mask = if n > 1 {
            let m: Vec<f32> = (0..n)
                .flat_map(|i| (0..pos + n).map(move |j| if j > pos + i { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            Some(Tensor::from_vec(m, (n, pos + n), device)?)
        } else {
            None
        };
        for ((b, kv), (ak, av)) in self.blocks.iter().zip(cache.kv.iter_mut()).zip(&audio.0) {
            let h = b.attn_ln.forward(&x)?;
            let q = self.heads(&b.attn.query.forward(&h)?)?.contiguous()?;
            // Contiguous before and after joining the cache: the GPU matrix multiply (CUDA,
            // Metal) refuses the strided layout a join of split heads produces; the CPU one
            // doesn't care.
            let k = self.heads(&b.attn.key.forward(&h)?)?.contiguous()?;
            let v = self.heads(&b.attn.value.forward(&h)?)?.contiguous()?;
            let (k, v) = match kv.take() {
                Some((pk, pv)) => {
                    (Tensor::cat(&[&pk, &k], 2)?.contiguous()?, Tensor::cat(&[&pv, &v], 2)?.contiguous()?)
                }
                None => (k, v),
            };
            let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let att = match &mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            let y = softmax_last_dim(&att)?.matmul(&v)?.transpose(1, 2)?.reshape((1, n, d))?;
            x = (x + b.attn.out.forward(&y)?)?;
            *kv = Some((k, v));

            let h = b.cross_ln.forward(&x)?;
            let q = self.heads(&b.cross.query.forward(&h)?)?.contiguous()?;
            let y = softmax_last_dim(&q.matmul(ak)?)?.matmul(av)?.transpose(1, 2)?.reshape((1, n, d))?;
            x = (x + b.cross.out.forward(&y)?)?;

            let h = b.mlp_ln.forward(&x)?;
            x = (x + b.fc2.forward(&b.fc1.forward(&h)?.gelu()?)?)?;
        }
        cache.len += n;
        self.ln.forward(&x)
    }

    /// Scores for the token after the last position.
    fn logits(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let (_, n, _) = hidden.dims3()?;
        hidden.i((0, n - 1..n))?.matmul(&self.token_embedding.t()?)?.i(0)
    }
}

pub struct LoadedWhisper {
    pub model_id: String,
    config: w::Config,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    tokenizer: tokenizers::Tokenizer,
    filters: Vec<f32>,
    device: Device,
    suppress: Tensor,
    /// For live passes: `suppress` plus every special token (timestamps, languages, tasks),
    /// so a live pass can only write words or stop.
    suppress_specials: Tensor,
    sot: u32,
    eot: u32,
    transcribe: u32,
    translate: Option<u32>,
    no_timestamps: u32,
    no_speech: Option<u32>,
    multilingual: bool,
}

/// What one live pass heard (see [`LoadedWhisper::transcribe_live`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LivePass {
    pub text: String,
    /// Average log-probability of the chosen tokens: closer to 0 means more confident.
    pub avg_logprob: f64,
    /// How sure the model is that the audio holds no speech at all (0 to 1).
    pub no_speech_prob: f64,
    /// The language used: the one asked for, or the one detected.
    pub language: Option<String>,
    /// Number of text tokens written.
    pub tokens: usize,
}

struct DecodeOptions {
    language: Option<u32>,
    translate: bool,
    temperature: f64,
    seed: u64,
    max_tokens: usize,
    /// A live pass: special tokens are never chosen and repetition loops end sooner.
    live: bool,
}

struct Decoded {
    text: String,
    avg_logprob: f64,
    tokens: usize,
    /// Decoding was cut short because it kept repeating itself.
    looped: bool,
}

/// One encoded window, after <|startoftranscript|>: the scores there tell the language and
/// how likely the window holds no speech at all, and every decoding attempt continues from
/// this point.
struct Window {
    audio: AudioKv,
    start: TokenCache,
    first_logits: Tensor,
}

/// Whether greedy decoding is stuck repeating itself. File transcription stops after twelve
/// identical tokens in a row; live passes also stop when a short run of tokens (two to ten)
/// comes back four times in a row, or one token eight times, which is how Whisper loops on
/// noise.
fn stuck_in_loop(generated: &[u32], live: bool) -> bool {
    let n = generated.len();
    if n > 24 && generated[n - 12..].iter().all(|t| *t == generated[n - 1]) {
        return true;
    }
    live && (1..=10).any(|p| {
        let repeats = if p == 1 { 8 } else { 4 };
        n >= repeats * p && (1..repeats).all(|k| generated[n - p..] == generated[n - (k + 1) * p..n - k * p])
    })
}

impl LoadedWhisper {
    pub fn load(dir: &Path, model_id: &str, device: Device) -> Result<Self> {
        let config: w::Config =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json is missing")?)
                .context("config.json is not a valid Whisper config")?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        // SAFETY: memory-mapped read-only weights; never modified while loaded.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], w::DTYPE, &device)?
        };
        let filters = mel_filters(w::SAMPLE_RATE, w::N_FFT, config.num_mel_bins);
        let tok = |s: &str| tokenizer.token_to_id(s).ok_or_else(|| anyhow!("token {s} not in vocabulary"));
        let sot = tok(w::SOT_TOKEN)?;
        let eot = tok(w::EOT_TOKEN)?;
        let transcribe = tok(w::TRANSCRIBE_TOKEN)?;
        let translate = tokenizer.token_to_id(w::TRANSLATE_TOKEN);
        let no_timestamps = tok(w::NO_TIMESTAMPS_TOKEN)?;
        let no_speech = w::NO_SPEECH_TOKENS.iter().find_map(|t| tokenizer.token_to_id(t));
        let multilingual = tokenizer.token_to_id("<|en|>").is_some() && config.vocab_size >= 51865;
        let mask = |extra: &dyn Fn(u32) -> bool| -> Result<Tensor> {
            let v: Vec<f32> = (0..config.vocab_size as u32)
                .map(|i| {
                    if config.suppress_tokens.contains(&i) || i == no_timestamps || extra(i) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect();
            Ok(Tensor::new(v.as_slice(), &device)?)
        };
        let suppress = mask(&|_| false)?;
        // Every token after <|endoftext|> is special in Whisper vocabularies.
        let suppress_specials = mask(&|i| i > eot)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        // Candle's decoder is replaced by the cached one above, so only its encoder is kept.
        let Whisper { encoder, .. } = Whisper::load(&vb, config.clone())?;
        Ok(Self {
            model_id: model_id.into(),
            config,
            encoder,
            decoder,
            tokenizer,
            filters,
            device,
            suppress,
            suppress_specials,
            sot,
            eot,
            transcribe,
            translate,
            no_timestamps,
            no_speech,
            multilingual,
        })
    }

    /// Whether the model knows many languages (the `.en` models know English only).
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Encodes a window of mel frames and runs <|startoftranscript|>.
    fn window(&mut self, mel: &Tensor) -> Result<Window> {
        let features = self.encoder.forward(mel, true)?;
        let audio = self.decoder.audio_kv(&features)?;
        let mut start = self.decoder.empty_cache();
        let hidden = self.decoder.forward(&[self.sot], &audio, &mut start)?;
        let first_logits = self.decoder.logits(&hidden)?;
        Ok(Window { audio, start, first_logits })
    }

    /// How likely the window holds no speech at all.
    fn no_speech_prob(&self, win: &Window) -> Result<f64> {
        Ok(match self.no_speech {
            Some(ns) => softmax(&win.first_logits, 0)?.i(ns as usize)?.to_scalar::<f32>()? as f64,
            None => 0.0,
        })
    }

    /// The language Whisper hears in a window.
    fn detect_language(&self, win: &Window) -> Result<Option<String>> {
        if !self.multilingual {
            return Ok(Some("en".into()));
        }
        let codes: Vec<(&str, u32)> = LANGUAGES
            .iter()
            .filter_map(|(c, _)| self.tokenizer.token_to_id(&format!("<|{c}|>")).map(|id| (*c, id)))
            .collect();
        let ids: Vec<u32> = codes.iter().map(|(_, id)| *id).collect();
        let sub = win.first_logits.index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?;
        let probs: Vec<f32> = softmax(&sub, D::Minus1)?.to_vec1()?;
        Ok(probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| codes[i].0.to_string()))
    }

    /// Decodes one window, continuing after <|startoftranscript|>.
    fn decode(&self, win: &Window, opts: &DecodeOptions) -> Result<Decoded> {
        let mut prompt = Vec::new();
        if self.multilingual {
            if let Some(l) = opts.language {
                prompt.push(l);
            }
            prompt.push(if opts.translate { self.translate.unwrap_or(self.transcribe) } else { self.transcribe });
        }
        prompt.push(self.no_timestamps);
        let prefix = 1 + prompt.len();
        let suppress = if opts.live { &self.suppress_specials } else { &self.suppress };
        let mut cache = win.start.clone();
        let mut hidden = self.decoder.forward(&prompt, &win.audio, &mut cache)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut sum_logprob = 0f64;
        let mut looped = false;
        let mut rng_state = opts.seed.max(1);
        for _ in 0..opts.max_tokens {
            let logits = self.decoder.logits(&hidden)?.broadcast_add(suppress)?;
            let next = if opts.temperature > 0.0 {
                let probs: Vec<f32> = softmax(&(&logits / opts.temperature)?, 0)?.to_vec1()?;
                // xorshift sampling keeps this module free of extra RNG dependencies.
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let r = (rng_state % 1_000_000) as f32 / 1_000_000.0;
                let mut acc = 0f32;
                let mut chosen = (probs.len() - 1) as u32;
                for (j, p) in probs.iter().enumerate() {
                    acc += p;
                    if acc >= r {
                        chosen = j as u32;
                        break;
                    }
                }
                chosen
            } else {
                logits.argmax(0)?.to_scalar::<u32>()?
            };
            let prob = softmax(&logits, D::Minus1)?.i(next as usize)?.to_scalar::<f32>()? as f64;
            sum_logprob += prob.max(1e-12).ln();
            if next == self.eot || prefix + generated.len() >= self.config.max_target_positions {
                break;
            }
            generated.push(next);
            // Bail out of degenerate repetition loops.
            if stuck_in_loop(&generated, opts.live) {
                looped = true;
                break;
            }
            hidden = self.decoder.forward(&[next], &win.audio, &mut cache)?;
        }
        let text = self.tokenizer.decode(&generated, true).map_err(|e| anyhow!("{e}"))?;
        Ok(Decoded {
            text: text.trim().to_string(),
            avg_logprob: sum_logprob / (generated.len().max(1) as f64),
            tokens: generated.len(),
            looped,
        })
    }

    /// Log-mel spectrogram of 16 kHz samples, padded the way Whisper expects.
    fn mel(&self, pcm: &[f32]) -> Result<(Tensor, usize)> {
        let n_mels = self.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.config, pcm, &self.filters);
        let mel_len = mel.len() / n_mels;
        Ok((Tensor::from_vec(mel, (1, n_mels, mel_len), &self.device)?, mel_len))
    }

    /// Transcribes 16 kHz mono samples.
    pub fn run(
        &mut self,
        pcm: &[f32],
        language: Option<String>,
        translate: bool,
        cancel: &CancellationToken,
        progress: &mut dyn FnMut(f32),
    ) -> Result<(String, Vec<TranscriptSegment>, Option<String>)> {
        let (mel, mel_len) = self.mel(pcm)?;
        let content_frames = pcm.len() / w::HOP_LENGTH;
        if content_frames == 0 {
            return Ok((String::new(), Vec::new(), language));
        }
        let mut lang = language.filter(|l| !l.is_empty() && l != "auto");
        let mut lang_token = None;
        let mut segments = Vec::new();
        let mut seek = 0;
        while seek < content_frames {
            if cancel.is_cancelled() {
                anyhow::bail!("cancelled");
            }
            let len = w::N_FRAMES.min(mel_len - seek);
            // Encoded once; language detection and every temperature below reuse it.
            let win = self.window(&mel.narrow(2, seek, len)?)?;
            if seek == 0 {
                if lang.is_none() {
                    lang = self.detect_language(&win)?;
                }
                lang_token = lang.as_ref().and_then(|l| self.tokenizer.token_to_id(&format!("<|{l}|>")));
            }
            let ns = self.no_speech_prob(&win)?;
            let mut best = None;
            for (k, temp) in w::TEMPERATURES.iter().enumerate() {
                let opts = DecodeOptions {
                    language: lang_token,
                    translate,
                    temperature: *temp,
                    seed: 1234 + k as u64,
                    max_tokens: self.config.max_target_positions / 2,
                    live: false,
                };
                let d = self.decode(&win, &opts)?;
                let ok = d.avg_logprob >= w::LOGPROB_THRESHOLD || ns > w::NO_SPEECH_THRESHOLD;
                best = Some((d.text, d.avg_logprob));
                if ok {
                    break;
                }
            }
            if let Some((text, avg)) = best {
                let silent = ns > w::NO_SPEECH_THRESHOLD && avg < w::LOGPROB_THRESHOLD;
                if !silent && !text.is_empty() {
                    let start = seek as f64 * w::HOP_LENGTH as f64 / w::SAMPLE_RATE as f64;
                    let end_frame = (seek + w::N_FRAMES).min(content_frames);
                    segments.push(TranscriptSegment {
                        start,
                        end: end_frame as f64 * w::HOP_LENGTH as f64 / w::SAMPLE_RATE as f64,
                        text,
                    });
                }
            }
            seek += w::N_FRAMES;
            progress((seek.min(content_frames) as f32 / content_frames as f32).min(1.0));
        }
        let text = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        Ok((text, segments, lang))
    }

    /// One pass of live dictation: transcribes up to 30 seconds of 16 kHz audio in a single
    /// window, greedily (the same audio always gives the same text). The window is also used
    /// for language detection when `language` is `None`. The text is not filtered here: the
    /// caller weighs `no_speech_prob` and `avg_logprob` against what its voice detector heard.
    ///
    /// `full_window`: encode the standard 30-second window (zero-padded), as Whisper was
    /// trained. Otherwise only the audio plus a little silence is encoded, which is several
    /// times faster for short stretches but slightly less accurate: good for tentative text.
    ///
    /// A full-window pass whose greedy text comes out unsure while the window holds speech is
    /// decoded again with a little randomness (fixed seeds, so still repeatable), as file
    /// transcription does. Whisper Base sometimes answers a stretch that starts in the middle
    /// of a word with just "The" (measured: 14 s of clear speech, average log-probability
    /// -6.7), which would otherwise lose everything said in it. Only the decoder runs again.
    pub fn transcribe_live(&mut self, pcm: &[f32], language: Option<&str>, full_window: bool) -> Result<LivePass> {
        let pcm = &pcm[..pcm.len().min(w::N_SAMPLES)];
        if pcm.len() < w::HOP_LENGTH {
            return Ok(LivePass { language: language.map(str::to_string), ..Default::default() });
        }
        let (mel, mel_len) = self.mel(pcm)?;
        let frames = if full_window {
            w::N_FRAMES
        } else {
            // The audio plus two seconds of silence, in whole seconds (an even frame count).
            (pcm.len() / w::HOP_LENGTH + 2 * w::SAMPLE_RATE / w::HOP_LENGTH).div_ceil(100) * 100
        };
        let win = self.window(&mel.narrow(2, 0, frames.min(w::N_FRAMES).min(mel_len))?)?;
        let language = match language.filter(|l| !l.is_empty() && *l != "auto") {
            Some(l) => Some(l.to_string()),
            None => self.detect_language(&win)?,
        };
        let lang_token = language.as_ref().and_then(|l| self.tokenizer.token_to_id(&format!("<|{l}|>")));
        // Room for fast speech in any language, but no 200-token loops on a short clip.
        let max_tokens = (16.0 + 12.0 * pcm.len() as f64 / w::SAMPLE_RATE as f64) as usize;
        let no_speech_prob = self.no_speech_prob(&win)?;
        // At most three retries (0.2 to 0.6): hotter ones rarely help and can write long nonsense.
        let temperatures: &[f64] = if full_window { &w::TEMPERATURES[..4] } else { &[0.0] };
        let mut best: Option<Decoded> = None;
        for (k, &temperature) in temperatures.iter().enumerate() {
            let opts = DecodeOptions {
                language: lang_token,
                translate: false,
                temperature,
                seed: 1 + k as u64,
                max_tokens,
                live: true,
            };
            let d = self.decode(&win, &opts)?;
            // Sure enough, or Whisper thinks there is no speech at all (nothing to retry).
            let settled =
                no_speech_prob > w::NO_SPEECH_THRESHOLD || (d.avg_logprob >= w::LOGPROB_THRESHOLD && !d.looped);
            if !settled {
                tracing::debug!(temperature, avg_logprob = d.avg_logprob, looped = d.looped, "unsure live pass");
            }
            if settled || best.as_ref().is_none_or(|b| d.avg_logprob > b.avg_logprob) {
                best = Some(d);
            }
            if settled {
                break;
            }
        }
        let d = best.ok_or_else(|| anyhow!("no decoding attempt"))?;
        Ok(LivePass { text: d.text, avg_logprob: d.avg_logprob, no_speech_prob, language, tokens: d.tokens })
    }
}

/// Result of a transcription.
#[derive(Clone, Debug)]
pub struct TranscriptOutput {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub language: Option<String>,
}

struct Slot {
    model_id: String,
    /// Where it runs ("CPU", "NVIDIA GPU · CUDA"…).
    device_label: String,
    model: Arc<Mutex<LoadedWhisper>>,
    last_used: Instant,
}

/// Keeps the most recently used Whisper model in memory. File transcription and live
/// dictation share this one slot: loading a different model replaces the one in it (anyone
/// still using the old one keeps it until they finish).
#[derive(Default)]
pub struct WhisperCache {
    slot: Mutex<Option<Slot>>,
    /// One load at a time, so two callers never load the same model twice.
    loading: tokio::sync::Mutex<()>,
}

impl WhisperCache {
    fn slot(&self) -> std::sync::MutexGuard<'_, Option<Slot>> {
        self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn clear(&self) {
        *self.slot() = None;
    }

    /// Unloads the model when nobody has used it for `max_idle`. A model that is busy
    /// transcribing is never unloaded.
    pub fn reap(&self, max_idle: Duration) {
        let mut slot = self.slot();
        let idle = slot.as_ref().is_some_and(|s| s.model.try_lock().is_ok() && s.last_used.elapsed() > max_idle);
        if idle {
            *slot = None;
        }
    }

    /// Whether a model is in memory (also while it is busy).
    pub fn is_loaded(&self) -> bool {
        self.slot().is_some()
    }

    /// The library id of the model in memory (also while it is busy).
    pub fn loaded_id(&self) -> Option<String> {
        self.slot().as_ref().map(|s| s.model_id.clone())
    }

    /// Where the model in memory runs.
    pub fn loaded_device(&self) -> Option<String> {
        self.slot().as_ref().map(|s| s.device_label.clone())
    }

    /// Marks the model as just used, so the idle timer starts again.
    pub fn touch(&self, model_id: &str) {
        if let Some(s) = self.slot().as_mut().filter(|s| s.model_id == model_id) {
            s.last_used = Instant::now();
        }
    }

    /// The model, if it is the one in memory.
    pub fn get(&self, model_id: &str) -> Option<Arc<Mutex<LoadedWhisper>>> {
        let mut slot = self.slot();
        let s = slot.as_mut().filter(|s| s.model_id == model_id)?;
        s.last_used = Instant::now();
        Some(s.model.clone())
    }

    /// The model, loading it first when another one (or none) is in memory. Errors are
    /// sentences for people.
    pub async fn get_or_load(
        &self,
        state: &crate::SharedState,
        model_id: &str,
        dir: &Path,
    ) -> Result<Arc<Mutex<LoadedWhisper>>, String> {
        if let Some(m) = self.get(model_id) {
            return Ok(m);
        }
        let _loading = self.loading.lock().await;
        if let Some(m) = self.get(model_id) {
            return Ok(m);
        }
        let (device, label) = state.inference.device()?;
        let (dir, id) = (dir.to_path_buf(), model_id.to_string());
        let started = Instant::now();
        // People get a short reason (these sentences end up in the chat microphone's pop-up);
        // the log gets the whole error, paths and all.
        let loaded = tokio::task::spawn_blocking(move || LoadedWhisper::load(&dir, &id, device))
            .await
            .map_err(|e| {
                tracing::error!(model = %model_id, "loading the Whisper model crashed: {e}");
                "loading it crashed".to_string()
            })?
            .map_err(|e| {
                tracing::warn!(model = %model_id, "couldn't load the Whisper model: {e:#}");
                match file_trouble(&e) {
                    FileTrouble::Denied => "Local AI Studio isn't allowed to read its files (permission denied)",
                    FileTrouble::Missing => "some of its files are missing",
                    FileTrouble::Memory => "there isn't enough free memory",
                    FileTrouble::Other => "its files couldn't be read",
                }
                .to_string()
            })?;
        tracing::info!(model = %model_id, device = %label, ms = started.elapsed().as_millis() as u64, "loaded Whisper model");
        let model = Arc::new(Mutex::new(loaded));
        *self.slot() = Some(Slot {
            model_id: model_id.to_string(),
            device_label: label,
            model: model.clone(),
            last_used: Instant::now(),
        });
        Ok(model)
    }

    /// Transcribes an audio file with the given library model.
    #[allow(clippy::too_many_arguments)]
    pub async fn transcribe<F>(
        &self,
        state: &crate::SharedState,
        model: &ModelRow,
        audio_path: &Path,
        language: Option<String>,
        translate: bool,
        cancel: CancellationToken,
        mut progress: F,
    ) -> AppResult<TranscriptOutput>
    where
        F: FnMut(f32) + Send + 'static,
    {
        if model.kind != "transcription" || !model.analysis.runnable {
            return Err(AppError::bad_request(format!(
                "\"{}\" is not a usable speech-to-text model.",
                model.display_name
            )));
        }
        let pcm = crate::media::decode_audio_16k(audio_path)
            .await
            .map_err(|e| AppError::bad_request(format!("Could not read the audio file: {e}")))?;
        if pcm.len() < w::SAMPLE_RATE / 10 {
            return Err(AppError::bad_request("The audio is too short (less than 0.1 seconds)."));
        }
        let loaded = self
            .get_or_load(state, &model.id, &PathBuf::from(&model.path))
            .await
            .map_err(|e| AppError::bad_request(format!("Could not load the Whisper model: {e}")))?;
        let result = tokio::task::spawn_blocking(move || {
            let mut g = loaded.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            g.run(&pcm, language, translate, &cancel, &mut progress)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow!("transcription crashed: {e}")))?;
        // The idle timer starts when the transcription ends, not when it began.
        self.touch(&model.id);
        match result {
            Ok((text, segments, language)) => Ok(TranscriptOutput { text, segments, language }),
            Err(e) if e.to_string() == "cancelled" => Err(AppError::conflict("cancelled")),
            Err(e) => Err(AppError::Internal(e.context("transcription failed"))),
        }
    }
}

/// What kind of trouble an error about a model's files is: from the `io::Error` in its chain,
/// or else from its text (libraries that keep only the message).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileTrouble {
    Denied,
    Missing,
    Memory,
    Other,
}

pub fn file_trouble(e: &anyhow::Error) -> FileTrouble {
    use std::io::ErrorKind;
    for cause in e.chain() {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            match io.kind() {
                ErrorKind::PermissionDenied => return FileTrouble::Denied,
                ErrorKind::NotFound => return FileTrouble::Missing,
                ErrorKind::OutOfMemory => return FileTrouble::Memory,
                _ => {}
            }
        }
    }
    let text = format!("{e:#}").to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| text.contains(w));
    if has(&["permission denied", "access is denied", "os error 13)"]) {
        FileTrouble::Denied
    } else if has(&["no such file", "cannot find the file", "cannot find the path", "os error 2)"]) {
        FileTrouble::Missing
    } else if has(&["out of memory", "cannot allocate memory", "memory allocation"]) {
        FileTrouble::Memory
    } else {
        FileTrouble::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_errors_are_told_apart() {
        let io = |kind: std::io::ErrorKind| {
            anyhow::Error::new(std::io::Error::new(kind, "x")).context("/home/someone/model.safetensors")
        };
        assert_eq!(file_trouble(&io(std::io::ErrorKind::PermissionDenied)), FileTrouble::Denied);
        assert_eq!(file_trouble(&io(std::io::ErrorKind::NotFound)), FileTrouble::Missing);
        assert_eq!(file_trouble(&io(std::io::ErrorKind::InvalidData)), FileTrouble::Other);
        // Only the text is left (the library formatted the io::Error into its own error).
        let text = |t: &str| anyhow::anyhow!("{t}");
        let denied = r#"path: "/tmp/x/model.safetensors" Permission denied (os error 13)"#;
        assert_eq!(file_trouble(&text(denied)), FileTrouble::Denied);
        assert_eq!(file_trouble(&text("No such file or directory (os error 2)")), FileTrouble::Missing);
        assert_eq!(file_trouble(&text("CUDA_ERROR_OUT_OF_MEMORY: out of memory")), FileTrouble::Memory);
        assert_eq!(file_trouble(&text("shape mismatch for encoder.conv1.weight")), FileTrouble::Other);
        assert_eq!(file_trouble(&text("some other error (os error 20)")), FileTrouble::Other);
    }

    #[test]
    fn mel_scale_round_trips() {
        for f in [0.0, 440.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }

    #[test]
    fn mel_filters_have_whisper_shape_and_values() {
        let f = mel_filters(16000, 400, 80);
        assert_eq!(f.len(), 80 * 201);
        // Every filter is non-negative and has some support.
        for row in f.chunks(201) {
            assert!(row.iter().all(|v| *v >= 0.0));
            assert!(row.iter().any(|v| *v > 0.0));
        }
        // Reference values from librosa.filters.mel(sr=16000, n_fft=400, n_mels=80):
        // filter 0 peaks at bin 1 with weight ~0.02486.
        assert!((f[1] - 0.024_86).abs() < 5e-4, "got {}", f[1]);
        let f128 = mel_filters(16000, 400, 128);
        assert_eq!(f128.len(), 128 * 201);
    }
}
