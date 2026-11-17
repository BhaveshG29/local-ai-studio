//! Media helpers: audio decoding/resampling (pure Rust, Symphonia), image validation and
//! normalisation, and video frame extraction (via the `ffmpeg` command when installed).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};

use crate::config::Settings;
use crate::db::conversations::AttachmentKind;

/// Largest accepted upload (per file).
pub const MAX_UPLOAD_BYTES: usize = 512 * 1024 * 1024;
/// Longest edge of images sent to models.
const MAX_IMAGE_EDGE: u32 = 2048;

/// Detects what kind of media a file is from its first bytes (falls back to the extension).
pub fn sniff(bytes: &[u8], file_name: &str) -> Option<(AttachmentKind, &'static str)> {
    let starts = |sig: &[u8]| bytes.starts_with(sig);
    if starts(b"\x89PNG\r\n\x1a\n") {
        return Some((AttachmentKind::Image, "image/png"));
    }
    if starts(&[0xFF, 0xD8, 0xFF]) {
        return Some((AttachmentKind::Image, "image/jpeg"));
    }
    if starts(b"GIF87a") || starts(b"GIF89a") {
        return Some((AttachmentKind::Image, "image/gif"));
    }
    if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Some((AttachmentKind::Image, "image/webp"));
    }
    if starts(b"BM") && file_name.to_lowercase().ends_with(".bmp") {
        return Some((AttachmentKind::Image, "image/bmp"));
    }
    if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Some((AttachmentKind::Audio, "audio/wav"));
    }
    if starts(b"fLaC") {
        return Some((AttachmentKind::Audio, "audio/flac"));
    }
    if starts(b"ID3") || (bytes.len() > 1 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0) {
        return Some((AttachmentKind::Audio, "audio/mpeg"));
    }
    if starts(b"OggS") {
        return Some((AttachmentKind::Audio, "audio/ogg"));
    }
    let lower = file_name.to_lowercase();
    if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
        // MP4 family: audio-only M4A vs video.
        let brand = &bytes[8..12];
        if brand == b"M4A " || lower.ends_with(".m4a") {
            return Some((AttachmentKind::Audio, "audio/mp4"));
        }
        return Some((AttachmentKind::Video, if lower.ends_with(".mov") { "video/quicktime" } else { "video/mp4" }));
    }
    if starts(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some((AttachmentKind::Video, if lower.ends_with(".mkv") { "video/x-matroska" } else { "video/webm" }));
    }
    if lower.ends_with(".avi") && bytes.len() > 12 && &bytes[0..4] == b"RIFF" {
        return Some((AttachmentKind::Video, "video/x-msvideo"));
    }
    None
}

/// Validates an image and normalises it to PNG or JPEG (models reliably accept these),
/// downscaling very large images. Returns (bytes, mime, extension).
pub fn normalize_image(bytes: &[u8]) -> Result<(Vec<u8>, &'static str, &'static str)> {
    let reader =
        image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().context("not a readable image")?;
    let format = reader.format();
    let (w, h) = reader.into_dimensions().context("the image is damaged or in an unsupported format")?;
    if w == 0 || h == 0 || (w as u64) * (h as u64) > 80_000_000 {
        bail!("image dimensions {w}×{h} are not supported");
    }
    let keep = matches!(format, Some(image::ImageFormat::Png) | Some(image::ImageFormat::Jpeg))
        && w <= MAX_IMAGE_EDGE
        && h <= MAX_IMAGE_EDGE;
    if keep {
        let (mime, ext) =
            if format == Some(image::ImageFormat::Png) { ("image/png", "png") } else { ("image/jpeg", "jpg") };
        return Ok((bytes.to_vec(), mime, ext));
    }
    let img = image::load_from_memory(bytes).context("could not decode the image")?;
    let img = if w > MAX_IMAGE_EDGE || h > MAX_IMAGE_EDGE {
        img.resize(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let mut out = Vec::new();
    img.to_rgba8()
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .context("could not convert the image")?;
    Ok((out, "image/png", "png"))
}

/// Decodes any supported audio file to 16 kHz mono f32 samples.
pub async fn decode_audio_16k(path: &Path) -> Result<Vec<f32>> {
    let p = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let (samples, rate) = decode_audio(&p)?;
        Ok(resample(&samples, rate, 16_000))
    })
    .await
    .map_err(|e| anyhow!("audio decoding crashed: {e}"))?
}

/// Decodes an audio file to mono f32 at its native sample rate.
pub fn decode_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| {
            anyhow!("unsupported or damaged audio file ({e}); supported: WAV, MP3, FLAC, OGG/Vorbis, M4A/AAC")
        })?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("the file contains no audio track")?;
    let track_id = track.id;
    let rate = track.codec_params.sample_rate.context("unknown sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| anyhow!("unsupported audio codec: {e}"))?;
    let max_samples = rate as usize * 60 * 60 * 3; // 3 hours
    let mut out: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(anyhow!("error reading audio: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let channels = spec.channels.count().max(1);
                let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                buf.copy_interleaved_ref(decoded);
                for frame in buf.samples().chunks(channels) {
                    out.push(frame.iter().sum::<f32>() / channels as f32);
                }
                if out.len() > max_samples {
                    bail!("audio longer than 3 hours is not supported");
                }
            }
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow!("error decoding audio: {e}")),
        }
    }
    if out.is_empty() {
        bail!("no audio samples could be decoded");
    }
    Ok((out, rate))
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Half-width of the resampling filter, in zero crossings of the sinc.
const HALF_WIDTH: f64 = 16.0;
/// Up to this many filter phases are worked out once and kept in a table; stranger rate
/// pairs compute their weights as they go.
const MAX_TABLE_PHASES: u64 = 4096;

/// Weight of an input sample `d` input samples away from an output sample: a sinc low-pass
/// at `cutoff` (relative to the input rate), shaped by a Blackman window.
fn kernel(d: f64, cutoff: f64) -> f64 {
    let t = d * cutoff;
    let n = (t / HALF_WIDTH + 1.0) / 2.0;
    if !(0.0..=1.0).contains(&n) {
        return 0.0;
    }
    let window = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * n).cos() + 0.08 * (4.0 * std::f64::consts::PI * n).cos();
    cutoff * sinc(t) * window
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Whether [`StreamResampler`] keeps a table of weights for this rate pair (every usual
/// hardware rate has one). Other pairs, such as 8001 Hz, work out each weight as they go,
/// which is hundreds of times slower.
pub fn has_weight_table(from: u32, to: u32) -> bool {
    let (from, to) = (from.max(1) as u64, to.max(1) as u64);
    from == to || to / gcd(from, to) <= MAX_TABLE_PHASES
}

/// Band-limited resampling with a Blackman-windowed sinc kernel.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let mut r = StreamResampler::new(from, to);
    let mut out = r.push(input);
    out.extend(r.finish());
    out
}

/// Resamples audio that arrives in pieces (a microphone sending a chunk every few hundred
/// milliseconds). It keeps the few input samples the filter still needs between calls, so
/// the result is exactly what [`resample`] gives for the whole recording at once: no clicks
/// or gaps where the pieces meet. Memory use stays small however long the stream runs.
pub struct StreamResampler {
    from: u32,
    to: u32,
    /// `from / to` as a reduced fraction: output sample `i` sits at input position
    /// `i * step / phases`.
    step: u64,
    phases: u64,
    cutoff: f64,
    /// The filter reaches this many input samples to each side.
    half: i64,
    /// Normalised weights, `2 * half + 1` per phase, when the rate pair has few phases.
    table: Option<Vec<f32>>,
    /// Input samples still needed, starting at absolute index `buf_start`.
    buf: Vec<f32>,
    buf_start: u64,
    total_in: u64,
    next_out: u64,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Self {
        let (from64, to64) = (from.max(1) as u64, to.max(1) as u64);
        let g = gcd(from64, to64);
        let (step, phases) = (from64 / g, to64 / g);
        let cutoff = (to64 as f64 / from64 as f64).min(1.0) * 0.97;
        let half = (HALF_WIDTH / cutoff).ceil() as i64;
        let taps = (2 * half + 1) as usize;
        let table = (from != to && phases <= MAX_TABLE_PHASES).then(|| {
            let mut t = Vec::with_capacity(phases as usize * taps);
            for p in 0..phases {
                let frac = p as f64 / phases as f64;
                let w: Vec<f64> = (-half..=half).map(|j| kernel(frac - j as f64, cutoff)).collect();
                let sum: f64 = w.iter().sum();
                t.extend(w.iter().map(|v| (v / sum) as f32));
            }
            t
        });
        Self { from, to, step, phases, cutoff, half, table, buf: Vec::new(), buf_start: 0, total_in: 0, next_out: 0 }
    }

    /// Input rate (Hz).
    pub fn from_rate(&self) -> u32 {
        self.from
    }

    /// Output rate (Hz).
    pub fn to_rate(&self) -> u32 {
        self.to
    }

    /// Where output sample `i` falls in the input: the nearest input sample at or before it,
    /// and the filter phase.
    fn position(&self, i: u64) -> (i64, u64) {
        let num = i as u128 * self.step as u128;
        ((num / self.phases as u128) as i64, (num % self.phases as u128) as u64)
    }

    /// Adds input samples; returns the output samples that can be computed so far.
    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from == self.to {
            self.total_in += input.len() as u64;
            return input.to_vec();
        }
        let mut out = Vec::with_capacity((input.len() as u64 * self.phases / self.step) as usize + 1);
        // In blocks, so a whole recording never sits in the buffer twice.
        for block in input.chunks(1 << 16) {
            self.buf.extend_from_slice(block);
            self.total_in += block.len() as u64;
            loop {
                let (center, phase) = self.position(self.next_out);
                if center + self.half >= self.total_in as i64 {
                    break;
                }
                out.push(self.sample(center, phase));
                self.next_out += 1;
            }
            self.trim();
        }
        out
    }

    /// Ends the stream: returns the last samples (the filter treats what follows as absent).
    /// The total output length is the input length times `to / from`, rounded.
    pub fn finish(&mut self) -> Vec<f32> {
        if self.from == self.to {
            return Vec::new();
        }
        let total_out = ((self.total_in as f64) * (self.to as f64 / self.from as f64)).round() as u64;
        let mut out = Vec::new();
        while self.next_out < total_out {
            let (center, phase) = self.position(self.next_out);
            out.push(self.sample(center, phase));
            self.next_out += 1;
        }
        self.buf.clear();
        self.buf_start = self.total_in;
        out
    }

    fn sample(&self, center: i64, phase: u64) -> f32 {
        let (lo, hi) = (center - self.half, center + self.half);
        let taps = (2 * self.half + 1) as usize;
        if lo >= self.buf_start as i64
            && hi < self.total_in as i64
            && let Some(table) = &self.table
        {
            let w = &table[phase as usize * taps..(phase as usize + 1) * taps];
            let start = (lo - self.buf_start as i64) as usize;
            let s = &self.buf[start..start + taps];
            return s.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
        }
        // At the very start or end of the stream (or for rate pairs without a table): only
        // the samples that exist count, and the weights are normalised over those.
        let frac = phase as f64 / self.phases as f64;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for k in lo.max(self.buf_start as i64)..=hi.min(self.total_in as i64 - 1) {
            let Some(v) = self.buf.get((k - self.buf_start as i64) as usize) else {
                continue;
            };
            let wgt = kernel(frac + (center - k) as f64, self.cutoff);
            acc += *v as f64 * wgt;
            wsum += wgt;
        }
        if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 }
    }

    /// Forgets input samples no future output needs.
    fn trim(&mut self) {
        let (center, _) = self.position(self.next_out);
        let keep_from = (center - self.half).max(0) as u64;
        if keep_from > self.buf_start {
            let n = ((keep_from - self.buf_start) as usize).min(self.buf.len());
            self.buf.drain(..n);
            self.buf_start += n as u64;
        }
    }
}

/// The ffmpeg executable to use, if any.
pub fn ffmpeg_path(settings: &Settings) -> Option<PathBuf> {
    settings.media.ffmpeg_path.clone().filter(|p| p.is_file()).or_else(|| crate::util::find_executable("ffmpeg"))
}

/// How to install ffmpeg on this platform.
pub fn ffmpeg_install_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "Install ffmpeg with Homebrew (brew install ffmpeg), or set its location in Settings → Media."
    } else if cfg!(windows) {
        "Install ffmpeg (for example: winget install ffmpeg), then restart Local AI Studio, or set its location in Settings → Media."
    } else {
        "Install ffmpeg with your package manager (for example: sudo apt install ffmpeg), or set its location in Settings → Media."
    }
}

async fn run_ffmpeg(ffmpeg: &Path, args: &[String]) -> Result<std::process::Output> {
    let mut cmd = tokio::process::Command::new(ffmpeg);
    cmd.args(args).stdin(std::process::Stdio::null()).kill_on_drop(true);
    #[cfg(windows)]
    {
        cmd.creation_flags(0x0800_0000);
    }
    tokio::time::timeout(Duration::from_secs(120), cmd.output())
        .await
        .map_err(|_| anyhow!("ffmpeg took too long"))?
        .context("could not run ffmpeg")
}

/// Parses "Duration: 00:01:02.50" from ffmpeg's stderr.
pub fn parse_ffmpeg_duration(stderr: &str) -> Option<f64> {
    let idx = stderr.find("Duration: ")?;
    let rest = &stderr[idx + 10..];
    let ts = rest.split(',').next()?.trim();
    let parts: Vec<&str> = ts.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let s: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Extracts `n` frames spread evenly across a video. Returns paths relative to `uploads`.
pub async fn extract_frames(
    ffmpeg: Option<&Path>,
    video: &Path,
    uploads: &Path,
    n: usize,
) -> Result<(Vec<String>, Option<f64>)> {
    let ffmpeg =
        ffmpeg.ok_or_else(|| anyhow!("Video needs ffmpeg, which isn't installed. {}", ffmpeg_install_hint()))?;
    let probe = run_ffmpeg(ffmpeg, &["-hide_banner".into(), "-i".into(), video.to_string_lossy().to_string()]).await?;
    let stderr = String::from_utf8_lossy(&probe.stderr);
    let duration = parse_ffmpeg_duration(&stderr);
    if !stderr.contains("Video:") {
        bail!("this file has no video stream that ffmpeg can read");
    }
    let frames_dir = uploads.join("frames");
    tokio::fs::create_dir_all(&frames_dir).await?;
    let id = crate::util::new_id();
    let n = n.max(1);
    let mut out = Vec::new();
    for i in 0..n {
        let t = match duration {
            Some(d) if d > 0.0 => d * (i as f64 + 0.5) / n as f64,
            _ => i as f64,
        };
        let name = format!("frames/{id}_{i:02}.jpg");
        let path = uploads.join(&name);
        let args: Vec<String> = vec![
            "-hide_banner".into(),
            "-loglevel".into(),
            "error".into(),
            "-ss".into(),
            format!("{t:.3}"),
            "-i".into(),
            video.to_string_lossy().to_string(),
            "-frames:v".into(),
            "1".into(),
            "-vf".into(),
            "scale='min(768,iw)':-2".into(),
            "-q:v".into(),
            "3".into(),
            "-y".into(),
            path.to_string_lossy().to_string(),
        ];
        let res = run_ffmpeg(ffmpeg, &args).await?;
        if res.status.success() && path.exists() {
            out.push(name);
        }
    }
    if out.is_empty() {
        bail!("ffmpeg could not extract any frames from this video");
    }
    Ok((out, duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffing_detects_common_formats() {
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\nxxxx", "a.png").unwrap().0, AttachmentKind::Image);
        assert_eq!(sniff(&[0xFF, 0xD8, 0xFF, 0xE0], "a.jpg").unwrap().1, "image/jpeg");
        assert_eq!(sniff(b"RIFF\0\0\0\0WAVEfmt ", "a.wav").unwrap().0, AttachmentKind::Audio);
        assert_eq!(sniff(b"\0\0\0\x18ftypisom\0\0", "clip.mp4").unwrap().0, AttachmentKind::Video);
        assert_eq!(sniff(b"\0\0\0\x18ftypM4A \0\0", "a.m4a").unwrap().0, AttachmentKind::Audio);
        assert!(sniff(b"hello world", "notes.txt").is_none());
        assert!(sniff(b"MZ\x90\0", "evil.png").is_none());
    }

    #[test]
    fn resampling_preserves_a_tone() {
        // 440 Hz sine at 48 kHz → 16 kHz: frequency and amplitude must be preserved.
        let from = 48_000;
        let input: Vec<f32> =
            (0..48_000).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin()).collect();
        let out = resample(&input, from, 16_000);
        assert_eq!(out.len(), 16_000);
        let peak = out[1000..15000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {peak}");
        // zero crossings ≈ 2 * 440 per second
        let crossings = out[..16_000].windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        assert!((crossings as i64 - 880).abs() <= 4, "crossings {crossings}");
    }

    /// The resampler as it was before it could stream: every weight computed on the spot.
    fn reference_resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
        let ratio = to as f64 / from as f64;
        let cutoff = ratio.min(1.0) * 0.97;
        let scale = (HALF_WIDTH / cutoff).ceil() as i64;
        let out_len = ((input.len() as f64) * ratio).round() as usize;
        (0..out_len)
            .map(|i| {
                let x = i as f64 / ratio;
                let center = x.floor() as i64;
                let (mut acc, mut wsum) = (0.0f64, 0.0f64);
                for k in (center - scale)..=(center + scale) {
                    if k < 0 || k as usize >= input.len() {
                        continue;
                    }
                    let w = kernel(x - k as f64, cutoff);
                    acc += input[k as usize] as f64 * w;
                    wsum += w;
                }
                if wsum.abs() > 1e-9 { (acc / wsum) as f32 } else { 0.0 }
            })
            .collect()
    }

    /// Speech-like test signal: a few tones and a click.
    fn test_signal(rate: u32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                let tau = 2.0 * std::f32::consts::PI;
                0.3 * (tau * 220.0 * t).sin() + 0.2 * (tau * 1375.0 * t).sin() + if i == n / 3 { 0.5 } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn streaming_resampler_matches_resampling_everything_at_once() {
        for (from, to) in [(48_000, 16_000), (44_100, 16_000), (8_000, 16_000), (22_050, 16_000), (16_000, 16_000)] {
            let input = test_signal(from, 1.3);
            let whole = resample(&input, from, to);
            if from != to {
                let reference = reference_resample(&input, from, to);
                assert_eq!(whole.len(), reference.len(), "{from} → {to}");
                let worst = whole.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                assert!(worst < 1e-4, "{from} → {to}: differs from the reference by {worst}");
            }
            // Chunks of awkward, varying sizes (as a browser might send them), including empty ones.
            let mut r = StreamResampler::new(from, to);
            let mut streamed = Vec::new();
            let mut pos = 0;
            for size in [0usize, 1, 7, 4096, 333, 12_000, 1, 0, 9_999].iter().cycle() {
                if pos >= input.len() {
                    break;
                }
                let end = (pos + size).min(input.len());
                streamed.extend(r.push(&input[pos..end]));
                pos = end;
            }
            streamed.extend(r.finish());
            assert_eq!(streamed.len(), whole.len(), "{from} → {to}");
            let worst = streamed.iter().zip(&whole).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-5, "{from} → {to}: chunk joins differ by {worst}");
        }
    }

    #[test]
    fn streaming_resampler_keeps_little_memory() {
        let mut r = StreamResampler::new(48_000, 16_000);
        let chunk = test_signal(48_000, 0.25);
        let mut produced = 0;
        for _ in 0..400 {
            produced += r.push(&chunk).len();
            assert!(r.buf.len() < 1_000, "kept {} samples", r.buf.len());
        }
        produced += r.finish().len();
        assert_eq!(produced, 400 * 4_000);
    }

    #[test]
    fn usual_rates_have_a_weight_table() {
        for rate in [8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 192_000] {
            assert!(has_weight_table(rate, 16_000), "{rate} Hz");
            assert_eq!(StreamResampler::new(rate, 16_000).table.is_some(), rate != 16_000, "{rate} Hz");
        }
        for rate in [8_001, 22_051, 191_999] {
            assert!(!has_weight_table(rate, 16_000), "{rate} Hz");
            assert!(StreamResampler::new(rate, 16_000).table.is_none(), "{rate} Hz");
        }
    }

    #[test]
    fn wav_files_decode() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.wav");
        // 16-bit PCM mono 8 kHz, 0.5 s of a 1 kHz tone
        let rate = 8000u32;
        let samples: Vec<i16> = (0..4000)
            .map(|i| ((2.0 * std::f32::consts::PI * 1000.0 * i as f32 / rate as f32).sin() * 16000.0) as i16)
            .collect();
        let mut wav = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for s in &samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&p, wav).unwrap();
        let (pcm, r) = decode_audio(&p).unwrap();
        assert_eq!(r, 8000);
        assert_eq!(pcm.len(), 4000);
        let up = resample(&pcm, 8000, 16000);
        assert_eq!(up.len(), 8000);
    }

    #[test]
    fn ffmpeg_duration_parses() {
        let s = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'a.mp4':\n  Duration: 00:01:02.50, start: 0.000000, bitrate: 1 kb/s";
        assert_eq!(parse_ffmpeg_duration(s), Some(62.5));
        assert_eq!(parse_ffmpeg_duration("nothing"), None);
    }

    #[test]
    fn images_are_validated_and_normalised() {
        let mut png = Vec::new();
        image::RgbImage::new(4, 3).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let (b, mime, ext) = normalize_image(&png).unwrap();
        assert_eq!((mime, ext), ("image/png", "png"));
        assert_eq!(b, png);
        assert!(normalize_image(b"not an image").is_err());
    }
}
