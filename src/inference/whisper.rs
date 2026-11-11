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
use candle_nn::ops::softmax;
use candle_transformers::models::whisper::{self as w, audio, model::Whisper};
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

pub struct LoadedWhisper {
    pub model_id: String,
    model: Whisper,
    tokenizer: tokenizers::Tokenizer,
    filters: Vec<f32>,
    device: Device,
    suppress: Tensor,
    sot: u32,
    eot: u32,
    transcribe: u32,
    translate: Option<u32>,
    no_timestamps: u32,
    no_speech: Option<u32>,
    multilingual: bool,
    last_used: Instant,
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
        let suppress: Vec<f32> = (0..config.vocab_size as u32)
            .map(|i| if config.suppress_tokens.contains(&i) || i == no_timestamps { f32::NEG_INFINITY } else { 0.0 })
            .collect();
        let suppress = Tensor::new(suppress.as_slice(), &device)?;
        let model = Whisper::load(&vb, config)?;
        Ok(Self {
            model_id: model_id.into(),
            model,
            tokenizer,
            filters,
            device,
            suppress,
            sot,
            eot,
            transcribe,
            translate,
            no_timestamps,
            no_speech,
            multilingual,
            last_used: Instant::now(),
        })
    }

    fn detect_language(&mut self, mel: &Tensor) -> Result<Option<String>> {
        if !self.multilingual {
            return Ok(Some("en".into()));
        }
        let codes: Vec<(&str, u32)> = LANGUAGES
            .iter()
            .filter_map(|(c, _)| self.tokenizer.token_to_id(&format!("<|{c}|>")).map(|id| (*c, id)))
            .collect();
        let audio = self.model.encoder.forward(mel, true)?;
        let tokens = Tensor::new(&[[self.sot]], &self.device)?;
        let ys = self.model.decoder.forward(&tokens, &audio, true)?;
        let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let ids: Vec<u32> = codes.iter().map(|(_, id)| *id).collect();
        let sub = logits.index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?;
        let probs: Vec<f32> = softmax(&sub, D::Minus1)?.to_vec1()?;
        Ok(probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| codes[i].0.to_string()))
    }

    /// Decodes one 30 s window. Returns (text, avg log-prob, no-speech probability).
    fn decode_window(
        &mut self,
        mel: &Tensor,
        language: Option<u32>,
        translate: bool,
        temperature: f64,
        seed: u64,
    ) -> Result<(String, f64, f64)> {
        let audio = self.model.encoder.forward(mel, true)?;
        let mut tokens = vec![self.sot];
        if self.multilingual {
            if let Some(l) = language {
                tokens.push(l);
            }
            tokens.push(if translate { self.translate.unwrap_or(self.transcribe) } else { self.transcribe });
        }
        tokens.push(self.no_timestamps);
        let prefix = tokens.len();
        let sample_len = self.model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = 0f64;
        let mut rng_state = seed.max(1);
        for i in 0..sample_len {
            let t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&t, &audio, i == 0)?;
            if i == 0
                && let Some(ns) = self.no_speech
            {
                let first = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&first, 0)?.i(ns as usize)?.to_scalar::<f32>()? as f64;
            }
            let (_, seq_len, _) = ys.dims3()?;
            let logits = self.model.decoder.final_linear(&ys.i((..1, seq_len - 1..))?)?.i(0)?.i(0)?;
            let logits = logits.broadcast_add(&self.suppress)?;
            let next = if temperature > 0.0 {
                let probs: Vec<f32> = softmax(&(&logits / temperature)?, 0)?.to_vec1()?;
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
                let v: Vec<f32> = logits.to_vec1()?;
                v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(self.eot)
            };
            let prob = softmax(&logits, D::Minus1)?.i(next as usize)?.to_scalar::<f32>()? as f64;
            sum_logprob += prob.max(1e-12).ln();
            if next == self.eot || tokens.len() >= self.model.config.max_target_positions {
                break;
            }
            tokens.push(next);
            // Bail out of degenerate repetition loops.
            let n = tokens.len();
            if n > prefix + 24 && tokens[n - 12..].iter().all(|t| *t == next) {
                break;
            }
        }
        let generated = &tokens[prefix..];
        let text = self.tokenizer.decode(generated, true).map_err(|e| anyhow!("{e}"))?;
        let avg = sum_logprob / (generated.len().max(1) as f64);
        Ok((text.trim().to_string(), avg, no_speech_prob))
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
        self.last_used = Instant::now();
        let n_mels = self.model.config.num_mel_bins;
        let mel = audio::pcm_to_mel(&self.model.config, pcm, &self.filters);
        let mel_len = mel.len() / n_mels;
        let mel = Tensor::from_vec(mel, (1, n_mels, mel_len), &self.device)?;
        let content_frames = pcm.len() / w::HOP_LENGTH;
        if content_frames == 0 {
            return Ok((String::new(), Vec::new(), language));
        }
        let first = mel.narrow(2, 0, w::N_FRAMES.min(mel_len))?;
        let lang = match language.filter(|l| !l.is_empty() && l != "auto") {
            Some(l) => Some(l),
            None => self.detect_language(&first)?,
        };
        let lang_token = lang.as_ref().and_then(|l| self.tokenizer.token_to_id(&format!("<|{l}|>")));
        let mut segments = Vec::new();
        let mut seek = 0;
        while seek < content_frames {
            if cancel.is_cancelled() {
                anyhow::bail!("cancelled");
            }
            let len = w::N_FRAMES.min(mel_len - seek);
            let window = mel.narrow(2, seek, len)?;
            let mut best = None;
            for (k, temp) in w::TEMPERATURES.iter().enumerate() {
                let (text, avg, ns) = self.decode_window(&window, lang_token, translate, *temp, 1234 + k as u64)?;
                let ok = avg >= w::LOGPROB_THRESHOLD || ns > w::NO_SPEECH_THRESHOLD;
                best = Some((text, avg, ns));
                if ok {
                    break;
                }
            }
            if let Some((text, avg, ns)) = best {
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
}

/// Result of a transcription.
#[derive(Clone, Debug)]
pub struct TranscriptOutput {
    pub text: String,
    pub segments: Vec<TranscriptSegment>,
    pub language: Option<String>,
}

/// Keeps the most recently used Whisper model in memory.
#[derive(Default)]
pub struct WhisperCache {
    slot: Mutex<Option<Arc<Mutex<LoadedWhisper>>>>,
}

impl WhisperCache {
    pub fn clear(&self) {
        *self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    pub fn reap(&self, max_idle: Duration) {
        let mut slot = self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let idle =
            slot.as_ref().and_then(|m| m.try_lock().ok().map(|g| g.last_used.elapsed() > max_idle)).unwrap_or(false);
        if idle {
            *slot = None;
        }
    }

    pub fn loaded_id(&self) -> Option<String> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .and_then(|m| m.try_lock().ok().map(|g| g.model_id.clone()))
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
        let existing = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .filter(|m| m.try_lock().map(|g| g.model_id == model.id).unwrap_or(true));
        let loaded = match existing {
            Some(m) => m,
            None => {
                let (device, _) = state.inference.device().map_err(AppError::bad_request)?;
                let dir = PathBuf::from(&model.path);
                let id = model.id.clone();
                let m = tokio::task::spawn_blocking(move || LoadedWhisper::load(&dir, &id, device))
                    .await
                    .map_err(|e| AppError::Internal(anyhow!("whisper loading crashed: {e}")))?
                    .map_err(|e| AppError::bad_request(format!("Could not load the Whisper model: {e:#}")))?;
                let arc = Arc::new(Mutex::new(m));
                *self.slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(arc.clone());
                arc
            }
        };
        let result = tokio::task::spawn_blocking(move || {
            let mut g = loaded.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            g.run(&pcm, language, translate, &cancel, &mut progress)
        })
        .await
        .map_err(|e| AppError::Internal(anyhow!("transcription crashed: {e}")))?;
        match result {
            Ok((text, segments, language)) => Ok(TranscriptOutput { text, segments, language }),
            Err(e) if e.to_string() == "cancelled" => Err(AppError::conflict("cancelled")),
            Err(e) => Err(AppError::Internal(e.context("transcription failed"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
