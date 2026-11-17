//! The deciding half of live dictation, free of threads and models so tests can drive it
//! with synthetic audio and a fake transcriber:
//!
//! * [`Vad`]: an energy-based voice detector with a noise floor that follows steady
//!   background noise (fans, traffic, hum).
//! * [`LiveStream`]: turns microphone chunks into 16 kHz audio, keeps only what isn't final
//!   yet (long silences are dropped as they arrive), finds where utterances end, decides what
//!   the model should transcribe next, and turns the model's answers into final text
//!   (`text`) and tentative text (`partial`).
//! * [`clean_pass`]: the guard against Whisper's habit of "hearing" words in noise
//!   ("Thank you.", "you", "[BLANK_AUDIO]", the same phrase over and over).
//!
//! Positions are counted in 20 ms frames of the audio not yet committed.

use std::collections::VecDeque;
use std::sync::LazyLock;

use crate::inference::whisper::LivePass;
use crate::media::StreamResampler;

/// Whisper's sample rate.
pub const RATE: u32 = 16_000;
/// Samples per voice-detector frame (20 ms).
pub const FRAME: usize = 320;
/// Frames per second.
pub const FPS: f32 = 50.0;

fn frames(secs: f32) -> usize {
    (secs * FPS).round() as usize
}

/// Timing rules of the engine, in seconds.
#[derive(Clone, Debug)]
pub struct Tuning {
    /// Silence after speech that ends an utterance (its text becomes final).
    pub pause: f32,
    /// Silence kept after an utterance's last speech.
    pub tail: f32,
    /// Audio kept from just before speech starts (soft first sounds).
    pub preroll: f32,
    /// Silence after which the probably-final pass starts, before the pause is certain.
    pub eager: f32,
    /// Utterances with less speech than this (a click, a cough) are dropped untranscribed.
    pub min_voiced: f32,
    /// Audio without a pause is split once it gets this long.
    pub split_after: f32,
    /// The split goes at the quietest moment of this many seconds before `split_after`.
    pub split_search: f32,
    /// The most audio one pass may hold (Whisper's window is 30 seconds).
    pub max_pass: f32,
    /// New audio needed before the tentative text is worked out again (raised by the
    /// session when passes are slow).
    pub cadence: f32,
    /// Tentative passes over at most this much audio use Whisper's faster short window;
    /// longer ones use the full window (the short one saves little there, and Whisper
    /// sometimes skips the start of a long stretch with it).
    pub short_window: f32,
    /// "Detect automatically" keeps the language of the first pass with this much speech.
    pub language_lock: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            pause: 0.7,
            tail: 0.3,
            preroll: 0.5,
            eager: 0.3,
            min_voiced: 0.25,
            split_after: 20.0,
            split_search: 5.0,
            max_pass: 28.0,
            cadence: 1.0,
            short_window: 8.0,
            language_lock: 1.5,
        }
    }
}

/// What the voice detector found in one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mark {
    /// Loudness after removing hum and rumble, in dB below full scale.
    pub db: f32,
    pub voiced: bool,
}

/// Frames quieter than this are taken for a muted or not-yet-started microphone (digital
/// silence): they are never speech and say nothing about the room's noise.
const DEAD_DB: f32 = -90.0;
/// The noise floor never goes below this (very clean or noise-suppressed input).
const MIN_FLOOR_DB: f32 = -70.0;
/// How long the floor remembers quiet moments (4 s of frames).
const FLOOR_WINDOW: usize = 200;
/// The floor is the third-quietest frame of the window, so one or two odd frames (the
/// moment a microphone starts) can't drag it down.
const FLOOR_RANK: usize = 3;
/// Speech starts this far above the floor…
const ONSET_MARGIN_DB: f32 = 9.0;
/// …and continues while it stays this far above it.
const SUSTAIN_MARGIN_DB: f32 = 5.0;
/// However quiet the room, speech is at least this loud.
const ABS_ONSET_DB: f32 = -58.0;
const ABS_SUSTAIN_DB: f32 = -62.0;
/// Frames in a row above the onset level before speech counts as started (60 ms; clicks are
/// shorter).
const ONSET_FRAMES: u32 = 3;
/// Sound at least this loud, for at least `SOUND_SECS` in all, means the microphone works and
/// hears the room (noise, music…), whether or not anyone spoke.
const SOUND_DB: f32 = ABS_ONSET_DB;
const SOUND_SECS: f32 = 0.5;

/// Energy-based voice detector. The noise floor is one of the quietest frames of the last
/// few seconds, so it rises with a fan that turns on and falls when it stops; speech must
/// stand clearly above it.
#[derive(Clone, Debug, Default)]
pub struct Vad {
    /// High-pass filter state (previous input and output).
    hp_x: f32,
    hp_y: f32,
    history: VecDeque<f32>,
    onset_run: u32,
    in_speech: bool,
}

impl Vad {
    /// Classifies one frame.
    pub fn frame(&mut self, samples: &[f32]) -> Mark {
        // A first-order high-pass (about 100 Hz) keeps mains hum and rumble out of the level.
        let mut energy = 0f64;
        for &x in samples {
            let y = x - self.hp_x + 0.96 * self.hp_y;
            self.hp_x = x;
            self.hp_y = y;
            energy += (y as f64) * (y as f64);
        }
        let db = (10.0 * (energy / samples.len().max(1) as f64 + 1e-12).log10()) as f32;
        if db < DEAD_DB {
            self.in_speech = false;
            self.onset_run = 0;
            return Mark { db, voiced: false };
        }
        self.history.push_back(db.max(MIN_FLOOR_DB));
        if self.history.len() > FLOOR_WINDOW {
            self.history.pop_front();
        }
        let floor = self.floor_db().unwrap_or(MIN_FLOOR_DB);
        if self.in_speech {
            if db <= (floor + SUSTAIN_MARGIN_DB).max(ABS_SUSTAIN_DB) {
                self.in_speech = false;
                self.onset_run = 0;
            }
        } else if db > (floor + ONSET_MARGIN_DB).max(ABS_ONSET_DB) {
            self.onset_run += 1;
            self.in_speech = self.onset_run >= ONSET_FRAMES;
        } else {
            self.onset_run = 0;
        }
        Mark { db, voiced: self.in_speech }
    }

    /// The current noise floor, once something has been heard.
    pub fn floor_db(&self) -> Option<f32> {
        let mut lowest = [f32::INFINITY; FLOOR_RANK];
        for &v in &self.history {
            if v < lowest[FLOOR_RANK - 1] {
                lowest[FLOOR_RANK - 1] = v;
                lowest.sort_by(f32::total_cmp);
            }
        }
        let known = self.history.len().min(FLOOR_RANK);
        (known > 0).then(|| lowest[known - 1])
    }

    /// Forgets the speech in progress (after a pause), keeping what it learned about the room.
    pub fn reset_speech(&mut self) {
        self.in_speech = false;
        self.onset_run = 0;
    }
}

/// Work for the model: transcribe `pcm` (the first `end` frames of the uncommitted audio).
#[derive(Clone, Debug)]
pub struct Job {
    /// The result becomes final text and the audio is dropped (otherwise it is tentative).
    pub commit: bool,
    /// Use Whisper's full 30-second window (final quality) rather than the faster short one.
    pub full_window: bool,
    pub end: usize,
    generation: u64,
    /// The first half of a stretch whose final pass went wrong (see [`looks_broken`]).
    retry: bool,
    pub pcm: Vec<f32>,
    /// The language to transcribe in; `None` asks the model to detect it.
    pub language: Option<String>,
    /// How much of the audio the voice detector took for speech.
    pub voiced_secs: f32,
}

struct LastPass {
    generation: u64,
    end: usize,
    full_window: bool,
    text: Option<String>,
    /// A full-window pass that went wrong: its text can't become final.
    broken: bool,
}

/// Stretches at least this long (seconds) whose final pass went wrong are worked out again
/// in two halves; shorter ones keep what the pass gave.
const REDO_MIN_SECS: f32 = 4.0;
/// Speech gives Whisper at least this many tokens per second the voice detector heard
/// (about 3 to 5 in English and other languages measured; fewer means words went missing).
const MIN_TOKENS_PER_VOICED_SEC: f32 = 1.5;

/// Whether a final-quality pass over speech most likely went wrong. Whisper Base sometimes
/// answers a long stretch of clear speech with a few words ("the next summer." for 14 s of
/// a read-out text) or with made-up sentences, unsure of itself, depending only on where the
/// stretch happens to start; the same words cut a little differently come out fine.
pub fn looks_broken(out: &LivePass, voiced_secs: f32) -> bool {
    voiced_secs >= 3.0
        && out.no_speech_prob < 0.6
        && (out.avg_logprob < -1.0 || (out.tokens as f32) < MIN_TOKENS_PER_VOICED_SEC * voiced_secs)
}

/// One dictation's audio and text. See the module documentation.
pub struct LiveStream {
    tuning: Tuning,
    fixed_language: Option<String>,
    resampler: Option<StreamResampler>,
    vad: Vad,
    /// 16 kHz samples short of a whole frame.
    carry: Vec<f32>,
    /// Audio not yet committed, and what the detector found in each of its frames.
    audio: Vec<f32>,
    marks: Vec<Mark>,
    /// Recent silence, kept in case speech starts (it is not in `audio` yet).
    preroll_audio: VecDeque<f32>,
    preroll_marks: VecDeque<Mark>,
    in_utterance: bool,
    silence_run: usize,
    /// One past the last voiced frame of the utterance in progress.
    voiced_end: usize,
    utterance_voiced: usize,
    /// Where finished utterances end (ascending).
    boundaries: Vec<usize>,
    /// Bumped on every commit: positions of older jobs no longer mean anything.
    generation: u64,
    last_pass: Option<LastPass>,
    /// The final pass over the first this-many frames went wrong (in this generation): they
    /// are worked out again in two halves.
    redo: Option<(u64, usize)>,
    /// The tentative text is out of date and should be redone without waiting for new audio.
    partial_due: bool,
    /// Start the probably-final pass after `Tuning::eager` of silence (worth it only when
    /// full passes are quick; see [`Self::set_eager`]).
    eager: bool,
    text: String,
    partial: String,
    language: Option<String>,
    language_locked: bool,
    heard_speech: bool,
    /// Frames at least `SOUND_DB` loud.
    loud_frames: usize,
    input_secs: f64,
}

impl LiveStream {
    /// `language`: a Whisper language code, or `None` to detect it.
    pub fn new(language: Option<String>, tuning: Tuning) -> Self {
        Self {
            tuning,
            language: language.clone(),
            fixed_language: language,
            resampler: None,
            vad: Vad::default(),
            carry: Vec::new(),
            audio: Vec::new(),
            marks: Vec::new(),
            preroll_audio: VecDeque::new(),
            preroll_marks: VecDeque::new(),
            in_utterance: false,
            silence_run: 0,
            voiced_end: 0,
            utterance_voiced: 0,
            boundaries: Vec::new(),
            generation: 0,
            last_pass: None,
            redo: None,
            partial_due: false,
            eager: true,
            text: String::new(),
            partial: String::new(),
            language_locked: false,
            heard_speech: false,
            loud_frames: 0,
            input_secs: 0.0,
        }
    }

    /// Final text so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Tentative text for the stretch not yet final ("" when none).
    pub fn partial(&self) -> &str {
        &self.partial
    }

    /// The language chosen, or the one detected (once known).
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Seconds of audio received.
    pub fn input_secs(&self) -> f64 {
        self.input_secs
    }

    /// Whether the voice detector has heard speech yet.
    pub fn heard_speech(&self) -> bool {
        self.heard_speech
    }

    /// Whether the microphone heard more than a quiet room (see `SOUND_DB`): with no speech,
    /// that is background noise rather than a muted or silent microphone.
    pub fn heard_sound(&self) -> bool {
        self.loud_frames >= frames(SOUND_SECS)
    }

    /// Seconds of audio kept that aren't final yet.
    pub fn uncommitted_secs(&self) -> f32 {
        self.marks.len() as f32 / FPS
    }

    /// Nothing is waiting to be transcribed (true after [`Self::finish_input`] once every
    /// pass is done).
    pub fn is_drained(&self) -> bool {
        self.marks.is_empty() && !self.in_utterance
    }

    /// Speech heard that isn't final yet.
    pub fn has_pending_speech(&self) -> bool {
        self.voiced(0, self.marks.len()) > 0
    }

    /// Sets how much new audio is needed before the tentative text is redone.
    pub fn set_cadence(&mut self, secs: f32) {
        self.tuning.cadence = secs;
    }

    pub fn cadence(&self) -> f32 {
        self.tuning.cadence
    }

    /// Whether to start the probably-final pass early, in the silence after speech. Its
    /// result becomes the final text when the pause is confirmed, so the text is final
    /// sooner; but when speech goes on it was a long pass for tentative text, which slows
    /// the next updates. Sessions turn it off when full passes are slow.
    pub fn set_eager(&mut self, on: bool) {
        self.eager = on;
    }

    /// Adds microphone samples (mono, at `rate` Hz, any chunk size).
    pub fn push(&mut self, samples: &[f32], rate: u32) {
        self.input_secs += samples.len() as f64 / rate.max(1) as f64;
        if self.resampler.as_ref().is_some_and(|r| r.from_rate() != rate)
            && let Some(mut old) = self.resampler.take()
        {
            let tail = old.finish();
            self.feed(&tail);
        }
        let out = self.resampler.get_or_insert_with(|| StreamResampler::new(rate, RATE)).push(samples);
        self.feed(&out);
    }

    /// No more audio for now (pause or stop): whatever is pending becomes one last utterance,
    /// which the next jobs commit.
    pub fn finish_input(&mut self) {
        if let Some(mut r) = self.resampler.take() {
            let tail = r.finish();
            self.feed(&tail);
        }
        if !self.carry.is_empty() {
            let mut f = std::mem::take(&mut self.carry);
            f.resize(FRAME, 0.0);
            self.frame(&f);
        }
        if self.in_utterance {
            let len = self.marks.len();
            if len > self.boundaries.last().copied().unwrap_or(0) {
                self.boundaries.push(len);
            }
            self.in_utterance = false;
            self.silence_run = 0;
            self.utterance_voiced = 0;
        }
        self.preroll_audio.clear();
        self.preroll_marks.clear();
        self.vad.reset_speech();
    }

    /// Audio may come again after [`Self::finish_input`] (resume after a pause). The new
    /// audio starts a new stream; the voice detector keeps what it learned about the room.
    pub fn resume_input(&mut self) {
        self.resampler = None;
        self.carry.clear();
        self.vad.reset_speech();
    }

    fn feed(&mut self, samples: &[f32]) {
        let mut s = samples;
        if !self.carry.is_empty() {
            let take = (FRAME - self.carry.len()).min(s.len());
            self.carry.extend_from_slice(&s[..take]);
            s = &s[take..];
            if self.carry.len() < FRAME {
                return;
            }
            let f = std::mem::take(&mut self.carry);
            self.frame(&f);
        }
        let (frames, rest) = s.as_chunks::<FRAME>();
        for f in frames {
            self.frame(f);
        }
        self.carry.extend_from_slice(rest);
    }

    fn frame(&mut self, f: &[f32]) {
        let mark = self.vad.frame(f);
        if mark.db > SOUND_DB {
            self.loud_frames = self.loud_frames.saturating_add(1);
        }
        if !self.in_utterance {
            self.preroll_audio.extend(f);
            self.preroll_marks.push_back(mark);
            self.trim_preroll();
            if mark.voiced {
                // Speech started: keep the moments before it too.
                self.audio.extend(self.preroll_audio.drain(..));
                self.marks.extend(self.preroll_marks.drain(..));
                self.in_utterance = true;
                self.silence_run = 0;
                self.utterance_voiced = 1;
                self.voiced_end = self.marks.len();
            }
            return;
        }
        self.audio.extend_from_slice(f);
        self.marks.push(mark);
        if mark.voiced {
            self.silence_run = 0;
            self.voiced_end = self.marks.len();
            self.utterance_voiced += 1;
            if self.utterance_voiced >= frames(self.tuning.min_voiced) {
                self.heard_speech = true;
            }
        } else {
            self.silence_run += 1;
            if self.silence_run >= frames(self.tuning.pause) {
                self.end_utterance();
            }
        }
    }

    /// The utterance in progress is over: it ends a little after its last speech, and the
    /// silence after that becomes the run-up for the next one.
    fn end_utterance(&mut self) {
        let len = self.marks.len();
        let end = (self.voiced_end + frames(self.tuning.tail)).min(len);
        self.preroll_marks = self.marks.split_off(end).into();
        self.preroll_audio = self.audio.split_off(end * FRAME).into();
        self.trim_preroll();
        if end > self.boundaries.last().copied().unwrap_or(0) {
            self.boundaries.push(end);
        }
        self.in_utterance = false;
        self.silence_run = 0;
        self.utterance_voiced = 0;
    }

    /// Silence older than the run-up is dropped: long recordings of silence cost nothing.
    fn trim_preroll(&mut self) {
        let keep = frames(self.tuning.preroll) + ONSET_FRAMES as usize;
        while self.preroll_marks.len() > keep {
            self.preroll_marks.pop_front();
            self.preroll_audio.drain(..FRAME.min(self.preroll_audio.len()));
        }
    }

    fn voiced(&self, from: usize, to: usize) -> usize {
        self.marks[from.min(self.marks.len())..to.min(self.marks.len())].iter().filter(|m| m.voiced).count()
    }

    /// Loudness around frame `i` (60 ms average).
    fn smoothed(&self, i: usize) -> f32 {
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(self.marks.len());
        let w = &self.marks[lo..hi];
        w.iter().map(|m| m.db).sum::<f32>() / w.len().max(1) as f32
    }

    /// Where to split audio that has gone on too long without a pause: the quietest moment
    /// in the last few seconds before the limit (the latest one among equally quiet ones).
    fn quietest_cut(&self) -> usize {
        let hi = frames(self.tuning.split_after).min(self.marks.len());
        let lo = hi.saturating_sub(frames(self.tuning.split_search)).max(1);
        (lo..hi).min_by(|&a, &b| self.smoothed(a).total_cmp(&self.smoothed(b)).then(b.cmp(&a))).unwrap_or(hi)
    }

    /// Where to cut a stretch of `end` frames in two: its quietest moment around the middle.
    fn middle_cut(&self, end: usize) -> usize {
        let end = end.min(self.marks.len());
        let (lo, hi) = ((end * 3 / 10).max(1), (end * 7 / 10).max(2));
        (lo..hi).min_by(|&a, &b| self.smoothed(a).total_cmp(&self.smoothed(b)).then(b.cmp(&a))).unwrap_or(end / 2)
    }

    /// The last pass, when its text can stand for the audio up to `end`: a pass with final
    /// quality over exactly that audio, or over audio that differs from it only by silence,
    /// having heard the last words and a moment of quiet after them. (Stop pressed just after
    /// the early final pass started is the common case: its text is final as is.)
    fn reusable(&self, end: usize) -> Option<&LastPass> {
        let p = self.last_pass.as_ref()?;
        if p.generation != self.generation || !p.full_window {
            return None;
        }
        if p.end == end {
            return Some(p);
        }
        let (lo, hi) = (p.end.min(end), p.end.max(end));
        let last_speech = self.marks[..end.min(self.marks.len())].iter().rposition(|m| m.voiced).map_or(0, |i| i + 1);
        let usable = hi - lo <= frames(self.tuning.pause)
            && self.voiced(lo, hi) == 0
            && p.end >= last_speech + frames(self.tuning.eager);
        usable.then_some(p)
    }

    fn language_for_pass(&self) -> Option<String> {
        self.fixed_language.clone().or_else(|| self.language.clone().filter(|_| self.language_locked))
    }

    fn job(&self, commit: bool, full_window: bool, end: usize) -> Job {
        let end = end.min(self.marks.len());
        Job {
            commit,
            full_window,
            end,
            generation: self.generation,
            retry: false,
            pcm: self.audio[..end * FRAME].to_vec(),
            language: self.language_for_pass(),
            voiced_secs: self.voiced(0, end) as f32 / FPS,
        }
    }

    /// What the model should do next, if anything. Steps that need no model (dropping a
    /// cough, reusing a pass that already covered a finished utterance) happen right here.
    pub fn next_job(&mut self) -> Option<Job> {
        let min_voiced = frames(self.tuning.min_voiced);
        loop {
            let len = self.marks.len();
            if len == 0 {
                return None;
            }
            if let Some(&b) = self.boundaries.first()
                && self.voiced(0, b) < min_voiced
            {
                self.commit(b, None);
                continue;
            }
            // A final pass went wrong: the same audio again, in two halves (the first now;
            // the rest follows as usual).
            if let Some((generation, end)) = self.redo
                && generation == self.generation
            {
                let cut = self.middle_cut(end);
                return Some(Job { retry: true, ..self.job(true, true, cut) });
            }
            // Finished utterances: everything up to the latest pause that fits in one pass.
            // When the model has fallen behind, several utterances go in one pass.
            if let Some(b) = self.boundaries.iter().rev().copied().find(|b| *b <= frames(self.tuning.max_pass)) {
                if let Some(p) = self.reusable(b) {
                    if p.broken && b >= frames(REDO_MIN_SECS) {
                        self.redo = Some((self.generation, b));
                    } else {
                        let text = p.text.clone();
                        self.commit(b, text);
                    }
                    continue;
                }
                return Some(self.job(true, true, b));
            }
            // Too long without a pause.
            if len >= frames(self.tuning.split_after) {
                let cut = self.quietest_cut();
                if self.voiced(0, cut) < min_voiced {
                    self.commit(cut, None);
                    continue;
                }
                return Some(self.job(true, true, cut));
            }
            if !self.in_utterance || self.voiced(0, len) < min_voiced {
                return None;
            }
            let covered = self.last_pass.as_ref().filter(|p| p.generation == self.generation);
            // Silence after speech: the next pass is probably the final one, so it uses the
            // full window and its text can be committed as it is once the pause is certain.
            let final_covered = covered
                .is_some_and(|p| p.full_window && p.end >= (self.voiced_end + frames(self.tuning.tail)).min(len));
            if self.eager && self.silence_run >= frames(self.tuning.eager) && !final_covered {
                return Some(self.job(false, true, len));
            }
            let covered_end = covered.map(|p| p.end).unwrap_or(0);
            if covered_end < len && (self.partial_due || len >= covered_end + frames(self.tuning.cadence)) {
                return Some(self.job(false, len > frames(self.tuning.short_window), len));
            }
            return None;
        }
    }

    /// Takes a pass's answer for a job from [`Self::next_job`].
    pub fn apply(&mut self, job: &Job, out: LivePass) {
        if job.generation != self.generation {
            return;
        }
        let broken = job.full_window && looks_broken(&out, job.voiced_secs);
        let text = clean_pass(&out, job.voiced_secs);
        // Only a pass that heard words tells the language: Whisper also "detects" one in a
        // cough, music or a door, and locking that would garble everything said afterwards.
        if self.fixed_language.is_none()
            && !self.language_locked
            && !broken
            && text.is_some()
            && let Some(l) = out.language.clone()
        {
            self.language = Some(l);
            self.language_locked = job.voiced_secs >= self.tuning.language_lock;
        }
        if job.commit {
            if broken && !job.retry && job.end >= frames(REDO_MIN_SECS) {
                // Not final like this: the same audio is worked out again in two halves.
                tracing::debug!(
                    secs = job.end as f32 / FPS,
                    "a final voice input pass went wrong; trying again in halves"
                );
                self.redo = Some((self.generation, job.end));
                return;
            }
            self.commit(job.end, text);
        } else {
            // A pass that went wrong doesn't replace better tentative text. Nor is text without a
            // single word shown (music can give "1.5x2.0x2.0"): the final pass decides about it.
            if !broken {
                self.partial = text.clone().filter(|t| has_word(t)).unwrap_or_default();
            }
            self.partial_due = false;
            self.last_pass = Some(LastPass {
                generation: self.generation,
                end: job.end,
                full_window: job.full_window,
                text,
                broken,
            });
        }
    }

    /// Makes the first `end` frames final: their text joins `text` and their audio goes.
    fn commit(&mut self, end: usize, text: Option<String>) {
        let shown = std::mem::take(&mut self.partial);
        if let Some(t) = &text {
            self.text = join_text(&self.text, t);
        }
        let end = end.min(self.marks.len());
        self.audio.drain(..end * FRAME);
        self.marks.drain(..end);
        self.boundaries.retain(|b| *b > end);
        for b in &mut self.boundaries {
            *b -= end;
        }
        self.voiced_end = self.voiced_end.saturating_sub(end);
        self.generation += 1;
        self.last_pass = None;
        self.redo = None;
        let speech_left = self.voiced(0, self.marks.len());
        // Until the next pass, the tentative text keeps what it said beyond the part just
        // made final (when speech is left for it to belong to), so words don't blink out.
        if speech_left > 0 {
            self.partial = text.as_deref().and_then(|t| remainder(&shown, t)).unwrap_or_default();
        }
        // More speech is already waiting: redo its tentative text straight away.
        self.partial_due = self.in_utterance && speech_left >= frames(self.tuning.min_voiced);
    }
}

/// Whether `text` holds a word: two letters or more in a row (digits and stray letters don't
/// count; Chinese and Japanese characters are letters).
fn has_word(text: &str) -> bool {
    text.split(|c: char| !c.is_alphabetic()).any(|w| w.chars().nth(1).is_some())
}

/// What `shown` says beyond `committed`, when it starts with the same words.
fn remainder(shown: &str, committed: &str) -> Option<String> {
    let shown_words: Vec<&str> = shown.split_whitespace().collect();
    let committed: Vec<String> = committed.split_whitespace().map(normalize).filter(|w| !w.is_empty()).collect();
    let (mut i, mut k) = (0, 0);
    while k < committed.len() && i < shown_words.len() {
        let w = normalize(shown_words[i]);
        if !w.is_empty() {
            if w != committed[k] {
                return None;
            }
            k += 1;
        }
        i += 1;
    }
    (k == committed.len()).then(|| shown_words[i..].join(" "))
}

/// Joins two pieces of transcript with a space, except before punctuation and between
/// Chinese, Japanese or Korean characters (written without spaces).
pub fn join_text(a: &str, b: &str) -> String {
    let (a, b) = (a.trim_end(), b.trim());
    if a.is_empty() || b.is_empty() {
        return format!("{a}{b}");
    }
    let first = b.chars().next().unwrap_or(' ');
    let last = a.chars().last().unwrap_or(' ');
    let glued = matches!(first, ',' | '.' | '!' | '?' | ';' | ':' | ')' | '%' | '、' | '。' | '，' | '！' | '？')
        || (is_cjk(last) && is_cjk(first));
    if glued { format!("{a}{b}") } else { format!("{a} {b}") }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF)
}

/// Lowercase letters, digits and apostrophes, single spaces: for comparing phrases.
fn normalize(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '\'' { c.to_lowercase().next().unwrap_or(c) } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// What Whisper says when it hears silence, breathing or noise (it learned these from
/// subtitles). Said for real they are kept, but only when the model is sure.
const STOCK_PHRASES: &[&str] = &[
    "you",
    "thank you",
    "thank you very much",
    "thank you so much",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "thank you for listening",
    "please subscribe",
    "subscribe",
    "bye",
    "bye bye",
    "goodbye",
    "the end",
    "so",
    "uh",
    "um",
    "hmm",
    "mm",
    "huh",
    "ah",
    "oh",
    "i",
    "a",
    "the",
    "music",
    "silence",
    "blank audio",
];

/// Credits from subtitle files: never real dictation.
const CREDIT_MARKERS: &[&str] = &[
    "amara org",
    "subtitles by",
    "captions by",
    "transcribed by",
    "translated by",
    "subscribe to",
    "like and subscribe",
    "untertitel",
    "sous titres",
    "продолжение следует",
    "ご視聴ありがとうございました",
    "字幕",
];

/// Words inside brackets that mark a sound rather than speech: "(upbeat music)",
/// "(laughs)", "(door closes)".
const SOUND_WORDS: &[&str] = &[
    "music",
    "laugh",
    "applause",
    "silence",
    "noise",
    "cough",
    "sigh",
    "inaudible",
    "unintelligible",
    "blank",
    "static",
    "crosstalk",
    "beep",
    "breath",
    "throat",
    "sniff",
    "typing",
    "footsteps",
    "wind",
    "speaking",
    "foreign",
    "mumbl",
    "whisper",
    "background",
    "chuckl",
    "giggl",
    "groan",
    "gasp",
    "clap",
    "ring",
    "buzz",
    "click",
    "bell",
    "door",
    "sound",
    "audio",
    "video",
];

static BRACKETS: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\[[^\]]*\]|\*[^*]*\*|[♪♫][^♪♫]*[♪♫]|[♪♫]").expect("valid pattern"));
static PARENS: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"\(([^)]*)\)").expect("valid pattern"));
/// "garden.We": a sentence end Whisper wrote without its space.
static GLUED_SENTENCE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(\p{Ll}[.!?])(\p{Lu})").expect("valid pattern"));

/// Removes sound descriptions: anything in square brackets or between asterisks, music
/// notes, and parentheses about sounds.
fn strip_annotations(text: &str) -> String {
    let text = BRACKETS.replace_all(text, " ");
    let whole = text.trim().starts_with('(') && text.trim().ends_with(')') && PARENS.find_iter(&text).count() == 1;
    let text = PARENS.replace_all(&text, |c: &regex::Captures| {
        let inner = c[1].to_lowercase();
        if whole || SOUND_WORDS.iter().any(|w| inner.contains(w)) { " ".to_string() } else { c[0].to_string() }
    });
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Collapses a phrase said three or more times in a row (a single word: four or more) to
/// one, which is how Whisper loops.
fn collapse_repeats(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let norm: Vec<String> = words.iter().map(|w| normalize(w)).collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let mut collapsed = false;
        for p in 1..=12usize {
            if i + 2 * p > words.len() {
                break;
            }
            let block = &norm[i..i + p];
            if block.iter().all(|w| w.is_empty()) {
                continue;
            }
            let mut reps = 1;
            while i + (reps + 1) * p <= words.len() && norm[i + reps * p..i + (reps + 1) * p] == *block {
                reps += 1;
            }
            if reps >= if p == 1 { 4 } else { 3 } {
                out.extend(&words[i..i + p]);
                i += reps * p;
                collapsed = true;
                break;
            }
        }
        if !collapsed {
            out.push(words[i]);
            i += 1;
        }
    }
    out.join(" ")
}

/// Decides what to keep from a pass: `None` when it most likely heard no words. Whisper's
/// own rule (likely no speech and low confidence) comes first; stock phrases and text far
/// too long for the speech heard must also be confident; sound descriptions and repetition
/// loops are removed.
pub fn clean_pass(out: &LivePass, voiced_secs: f32) -> Option<String> {
    if out.no_speech_prob > 0.6 && out.avg_logprob < -1.0 {
        return None;
    }
    if out.no_speech_prob > 0.85 && out.avg_logprob < -0.6 {
        return None;
    }
    let text = collapse_repeats(&strip_annotations(&out.text));
    let text = GLUED_SENTENCE.replace_all(&text, "$1 $2").into_owned();
    if !text.chars().any(char::is_alphanumeric) {
        return None;
    }
    let norm = normalize(&text);
    if CREDIT_MARKERS.iter().any(|m| norm.contains(m)) {
        return None;
    }
    let words = text.split_whitespace().count() as f32;
    let suspicious = STOCK_PHRASES.contains(&norm.as_str()) || words > 4.0 + 6.0 * voiced_secs;
    if suspicious && (out.no_speech_prob > 0.15 || out.avg_logprob < -0.7 || voiced_secs < 0.35) {
        return None;
    }
    if out.avg_logprob < -1.4 && voiced_secs < 1.0 {
        return None;
    }
    Some(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // ---- synthetic audio ----------------------------------------------------------------

    /// Deterministic white noise at roughly `db` dB below full scale.
    pub fn noise(secs: f32, db: f32, seed: u32, rate: u32) -> Vec<f32> {
        let amp = 10f32.powf(db / 20.0) * 3f32.sqrt();
        let mut x = 0x9E37_79B9u32 ^ seed;
        (0..(secs * rate as f32) as usize)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    pub fn silence(secs: f32, rate: u32) -> Vec<f32> {
        vec![0.0; (secs * rate as f32) as usize]
    }

    /// The fake vocabulary: each "word" is a tone burst at its own pitch.
    pub const WORDS: &[(&str, f32)] = &[
        ("alpha", 300.0),
        ("bravo", 450.0),
        ("charlie", 600.0),
        ("delta", 750.0),
        ("echo", 900.0),
        ("foxtrot", 1050.0),
        ("golf", 1200.0),
        ("hotel", 1350.0),
    ];

    /// One spoken "word": 0.35 s of its tone with soft edges, about 20 dB below full scale.
    pub fn word(name: &str, rate: u32) -> Vec<f32> {
        let f = WORDS.iter().find(|(w, _)| *w == name).expect("known word").1;
        let n = (0.35 * rate as f32) as usize;
        let ramp = (0.02 * rate as f32) as usize;
        (0..n)
            .map(|i| {
                let env = (i.min(n - 1 - i) as f32 / ramp as f32).min(1.0);
                0.1 * env * (2.0 * std::f32::consts::PI * f * i as f32 / rate as f32).sin()
            })
            .collect()
    }

    /// Words separated by short gaps (one utterance), over light noise.
    pub fn sentence(words: &[&str], rate: u32) -> Vec<f32> {
        let mut out = Vec::new();
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                out.extend(noise(0.12, -60.0, i as u32, rate));
            }
            out.extend(word(w, rate));
        }
        out
    }

    /// A transcriber that "hears" the tone bursts: finds loud stretches and names each by
    /// its pitch. A burst cut in two would be named twice, so bad cuts show up in tests.
    pub fn fake_transcribe(pcm: &[f32], language: Option<&str>) -> LivePass {
        let mut words = Vec::new();
        let mut run: Vec<f32> = Vec::new();
        let mut flush = |run: &mut Vec<f32>| {
            if run.len() >= FRAME * 3 {
                let crossings = run.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
                let freq = crossings as f32 / 2.0 / (run.len() as f32 / RATE as f32);
                let nearest = WORDS.iter().min_by(|a, b| (a.1 - freq).abs().total_cmp(&(b.1 - freq).abs()));
                if let Some((w, _)) = nearest {
                    words.push(*w);
                }
            }
            run.clear();
        };
        for f in pcm.chunks(FRAME) {
            let e = f.iter().map(|x| x * x).sum::<f32>() / f.len() as f32;
            if 10.0 * (e + 1e-12).log10() > -35.0 {
                run.extend_from_slice(f);
            } else {
                flush(&mut run);
            }
        }
        flush(&mut run);
        LivePass {
            text: words.join(" "),
            avg_logprob: -0.2,
            no_speech_prob: 0.01,
            language: Some(language.unwrap_or("en").to_string()),
            tokens: words.len(),
        }
    }

    /// Feeds `audio` in `chunk`-second pieces, running every job the stream asks for after
    /// each piece (a model faster than real time). Returns the jobs run.
    fn run_live(s: &mut LiveStream, audio: &[f32], rate: u32, chunk: f32) -> Vec<Job> {
        let mut jobs = Vec::new();
        for piece in audio.chunks((chunk * rate as f32) as usize) {
            s.push(piece, rate);
            drain(s, &mut jobs);
        }
        jobs
    }

    fn drain(s: &mut LiveStream, jobs: &mut Vec<Job>) {
        while let Some(job) = s.next_job() {
            let out = fake_transcribe(&job.pcm, job.language.as_deref());
            s.apply(&job, out);
            jobs.push(job);
        }
    }

    fn stop(s: &mut LiveStream, jobs: &mut Vec<Job>) {
        s.finish_input();
        drain(s, jobs);
        assert!(s.is_drained());
    }

    fn stream() -> LiveStream {
        LiveStream::new(Some("en".into()), Tuning::default())
    }

    // ---- voice detector -----------------------------------------------------------------

    fn voiced_frames(audio: &[f32]) -> Vec<bool> {
        let mut v = Vad::default();
        audio.as_chunks::<FRAME>().0.iter().map(|f| v.frame(f).voiced).collect()
    }

    #[test]
    fn silence_and_steady_noise_are_not_speech() {
        assert!(!voiced_frames(&silence(5.0, RATE)).contains(&true));
        for db in [-65.0, -50.0, -40.0, -30.0] {
            let v = voiced_frames(&noise(8.0, db, 7, RATE));
            assert!(!v.contains(&true), "noise at {db} dB was taken for speech");
        }
        // A microphone that starts with zeros, then hisses.
        let mut a = silence(0.5, RATE);
        a.extend(noise(5.0, -45.0, 3, RATE));
        assert!(!voiced_frames(&a).contains(&true));
    }

    #[test]
    fn background_noise_is_sound_but_not_speech() {
        let heard = |audio: &[f32]| {
            let mut s = stream();
            s.push(audio, RATE);
            (s.heard_sound(), s.heard_speech())
        };
        assert_eq!(heard(&silence(3.0, RATE)), (false, false), "a silent microphone");
        assert_eq!(heard(&noise(3.0, -70.0, 5, RATE)), (false, false), "a quiet room");
        assert_eq!(heard(&noise(3.0, -30.0, 6, RATE)), (true, false), "loud steady noise");
        // A click isn't enough.
        let mut click = silence(2.0, RATE);
        for x in &mut click[8_000..8_200] {
            *x = 0.3;
        }
        assert_eq!(heard(&click), (false, false), "a click");
        assert!(heard(&sentence(&["alpha", "bravo", "charlie"], RATE)).0, "speech is sound too");
    }

    #[test]
    fn speech_is_found_over_background_noise() {
        // Five seconds of fan noise, then words 15 dB louder than it.
        let mut a = noise(5.0, -38.0, 1, RATE);
        let start = a.len() / FRAME;
        let words = sentence(&["alpha", "bravo", "charlie"], RATE);
        let n = words.len() / FRAME;
        let bed = noise(words.len() as f32 / RATE as f32, -38.0, 2, RATE);
        a.extend(words.iter().zip(&bed).map(|(w, b)| w + b));
        a.extend(noise(2.0, -38.0, 4, RATE));
        let v = voiced_frames(&a);
        assert!(!v[..start].contains(&true), "noise before the words was taken for speech");
        let during = v[start..start + n].iter().filter(|x| **x).count();
        assert!(during as f32 > n as f32 * 0.6, "only {during} of {n} frames of words were heard");
        assert!(!v[start + n + 10..].contains(&true), "noise after the words was taken for speech");
    }

    #[test]
    fn clicks_are_not_speech() {
        let mut a = noise(2.0, -60.0, 9, RATE);
        for i in 0..5 {
            let at = (0.3 + i as f32 * 0.3) * RATE as f32;
            for s in &mut a[at as usize..at as usize + 200] {
                *s += 0.3;
            }
        }
        assert!(!voiced_frames(&a).contains(&true));
    }

    // ---- commit policy ------------------------------------------------------------------

    #[test]
    fn a_pause_makes_the_text_final() {
        let mut s = stream();
        let mut audio = silence(0.6, 48_000);
        audio.extend(sentence(&["alpha", "bravo", "charlie"], 48_000));
        audio.extend(noise(1.2, -60.0, 1, 48_000));
        audio.extend(sentence(&["delta", "echo"], 48_000));
        audio.extend(noise(1.2, -60.0, 2, 48_000));
        let mut jobs = run_live(&mut s, &audio, 48_000, 0.25);
        // Both utterances are final before stopping; the stop has nothing left to do.
        assert_eq!(s.text(), "alpha bravo charlie delta echo");
        assert_eq!(s.partial(), "");
        let before = jobs.len();
        stop(&mut s, &mut jobs);
        assert_eq!(jobs.len(), before);
        assert!(s.heard_speech());
        // Each final text came from a full-window pass, never from a tentative one.
        assert!(jobs.iter().filter(|j| j.commit).all(|j| j.full_window));
        assert!(jobs.iter().any(|j| !j.commit && !j.full_window), "no tentative text was shown");
    }

    #[test]
    fn the_early_final_pass_is_reused_at_the_pause() {
        let mut s = stream();
        let mut audio = sentence(&["golf", "hotel", "alpha", "bravo"], RATE);
        audio.extend(noise(1.5, -60.0, 1, RATE));
        let jobs = run_live(&mut s, &audio, RATE, 0.1);
        assert_eq!(s.text(), "golf hotel alpha bravo");
        // The pass started 0.3 s into the silence became the final text: no extra pass.
        assert_eq!(
            jobs.iter().filter(|j| j.commit).count(),
            0,
            "{:?}",
            jobs.iter().map(|j| j.commit).collect::<Vec<_>>()
        );
        assert_eq!(jobs.iter().filter(|j| j.full_window).count(), 1);
    }

    #[test]
    fn stopping_just_after_the_early_final_pass_reuses_it() {
        let mut s = stream();
        let mut audio = sentence(&["golf", "hotel", "alpha"], RATE);
        // Stop is pressed half a second after the last word.
        audio.extend(noise(0.5, -60.0, 1, RATE));
        let mut jobs = run_live(&mut s, &audio, RATE, 0.1);
        assert!(jobs.iter().any(|j| j.full_window && !j.commit), "no early final pass");
        let before = jobs.len();
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), "golf hotel alpha");
        assert_eq!(jobs.len(), before, "stopping ran another pass over the same words");
    }

    /// What Whisper Base answered for 14 s of clear speech that started mid-word.
    fn broken_pass() -> LivePass {
        LivePass { text: "The".into(), avg_logprob: -6.7, no_speech_prob: 0.13, language: Some("en".into()), tokens: 1 }
    }

    #[test]
    fn a_final_pass_that_went_wrong_is_redone_in_halves() {
        let names: Vec<&str> = (0..24).map(|i| WORDS[(i + i / WORDS.len()) % WORDS.len()].0).collect();
        let mut audio = sentence(&names, RATE);
        audio.extend(noise(1.0, -60.0, 1, RATE));
        assert!(audio.len() as f32 / RATE as f32 > 2.0 * REDO_MIN_SECS);
        // Everything arrives while the model is busy, so one final pass covers it all.
        let mut s = stream();
        s.push(&audio, RATE);
        let mut jobs = Vec::new();
        while let Some(job) = s.next_job() {
            let out = if jobs.is_empty() { broken_pass() } else { fake_transcribe(&job.pcm, job.language.as_deref()) };
            s.apply(&job, out);
            jobs.push(job);
        }
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), names.join(" "), "words were lost or doubled");
        let commits: Vec<&Job> = jobs.iter().filter(|j| j.commit).collect();
        assert_eq!(commits.len(), 3, "the pass that went wrong, then two halves");
        assert!(commits[1].retry && !commits[2].retry);
        let first_half = commits[1].pcm.len() as f32 / commits[0].pcm.len() as f32;
        assert!((0.3..=0.7).contains(&first_half), "cut at {first_half} of the stretch");

        // An early final pass that went wrong isn't made final either.
        let mut s = stream();
        let mut audio = sentence(&names[..12], RATE);
        audio.extend(noise(1.5, -60.0, 2, RATE));
        let mut jobs = Vec::new();
        for piece in audio.chunks(RATE as usize / 10) {
            s.push(piece, RATE);
            while let Some(job) = s.next_job() {
                let early = !job.commit && job.full_window;
                let out = if early { broken_pass() } else { fake_transcribe(&job.pcm, job.language.as_deref()) };
                s.apply(&job, out);
                jobs.push(job);
            }
        }
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), names[..12].join(" "));
        assert!(jobs.iter().any(|j| !j.commit && j.full_window), "no early final pass");
        assert!(jobs.iter().any(|j| j.retry), "{} jobs, none redone", jobs.len());
    }

    #[test]
    fn broken_passes_are_recognised() {
        let pass = |text: &str, tokens: usize, lp: f64, ns: f64| LivePass {
            text: text.into(),
            avg_logprob: lp,
            no_speech_prob: ns,
            language: Some("en".into()),
            tokens,
        };
        // Measured on Whisper Base: 14 s of a read-out text, 13 s of it speech.
        assert!(looks_broken(&pass("The", 1, -6.7, 0.13), 13.0));
        assert!(looks_broken(&pass("the next summer.", 5, -1.95, 0.31), 13.0));
        assert!(looks_broken(&pass("The United States will be able to make the decision", 30, -1.64, 0.13), 13.0));
        let fine = "last week and the city council approved the budget on Monday, construction should start";
        assert!(!looks_broken(&pass(fine, 45, -0.15, 0.04), 13.0));
        // Short stretches and noise are left to the hallucination guard.
        assert!(!looks_broken(&pass("The", 1, -6.7, 0.13), 2.0));
        assert!(!looks_broken(&pass("you", 1, -1.2, 0.8), 10.0));
    }

    #[test]
    fn slow_models_dont_start_the_final_pass_early() {
        let mut s = stream();
        s.set_eager(false);
        let mut audio = sentence(&["golf", "hotel", "alpha", "bravo"], RATE);
        // A breath between phrases, shorter than a pause.
        audio.extend(noise(0.5, -60.0, 1, RATE));
        audio.extend(sentence(&["charlie"], RATE));
        audio.extend(noise(1.5, -60.0, 2, RATE));
        let jobs = run_live(&mut s, &audio, RATE, 0.1);
        assert_eq!(s.text(), "golf hotel alpha bravo charlie");
        // The only full-window pass is the one that made the text final.
        let full: Vec<&Job> = jobs.iter().filter(|j| j.full_window).collect();
        assert_eq!(full.len(), 1);
        assert!(full[0].commit);
    }

    #[test]
    fn tentative_text_follows_the_cadence() {
        let mut s = stream();
        let words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"];
        let audio = sentence(&words, RATE);
        let mut seen = Vec::new();
        let mut passes = 0;
        for piece in audio.chunks(RATE as usize / 4) {
            s.push(piece, RATE);
            while let Some(job) = s.next_job() {
                assert!(!job.commit && !job.full_window);
                let out = fake_transcribe(&job.pcm, job.language.as_deref());
                s.apply(&job, out);
                passes += 1;
                seen.push(s.partial().to_string());
            }
        }
        let secs = audio.len() as f32 / RATE as f32;
        // About one pass per second of audio, each showing more of the sentence.
        assert!(passes as f32 >= secs - 1.5 && passes as f32 <= secs + 0.5, "{passes} passes for {secs} s");
        for pair in seen.windows(2) {
            assert!(pair[1].starts_with(&pair[0]) || pair[0].is_empty(), "{seen:?}");
        }
        assert_eq!(s.text(), "");
        let mut jobs = Vec::new();
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), words.join(" "));
    }

    #[test]
    fn stopping_mid_sentence_keeps_every_word() {
        let mut s = stream();
        let mut jobs = run_live(&mut s, &sentence(&["echo", "delta", "charlie"], 44_100), 44_100, 0.37);
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), "echo delta charlie");
        assert_eq!(s.partial(), "");
    }

    #[test]
    fn long_speech_without_pauses_is_split_between_words() {
        let mut s = stream();
        // A word order that never repeats itself (a repeated phrase would look like a loop).
        let names: Vec<&str> = (0..100).map(|i| WORDS[(i + i / WORDS.len()) % WORDS.len()].0).collect();
        let audio = sentence(&names, RATE);
        assert!(audio.len() as f32 / RATE as f32 > 45.0);
        let mut jobs = Vec::new();
        for piece in audio.chunks(RATE as usize / 2) {
            s.push(piece, RATE);
            while let Some(job) = s.next_job() {
                let out = fake_transcribe(&job.pcm, job.language.as_deref());
                let shown_before = format!("{} {}", s.text(), s.partial());
                s.apply(&job, out);
                if job.commit {
                    // Words shown after the split stay on screen until the next pass.
                    let shown = format!("{} {}", s.text(), s.partial());
                    assert!(shown.split_whitespace().count() >= shown_before.split_whitespace().count(), "{shown}");
                }
                jobs.push(job);
            }
        }
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), names.join(" "), "words were lost, doubled or reordered at a split");
        let commits: Vec<&Job> = jobs.iter().filter(|j| j.commit).collect();
        assert!(commits.len() >= 3, "{} commits", commits.len());
        for j in &jobs {
            assert!(j.pcm.len() <= 28 * RATE as usize, "a pass held {} s", j.pcm.len() / RATE as usize);
            // Tentative passes use the short window for short stretches only.
            if !j.commit {
                assert_eq!(j.full_window, j.pcm.len() > 8 * RATE as usize, "{} s", j.pcm.len() / RATE as usize);
            }
        }
        // Splits happen near the 20 s limit, not at the first small gap.
        for j in &commits[..commits.len() - 1] {
            let secs = j.pcm.len() as f32 / RATE as f32;
            assert!((15.0..=20.0).contains(&secs), "split after {secs} s");
        }
    }

    #[test]
    fn a_model_that_falls_behind_catches_up_in_order() {
        let mut s = stream();
        let mut audio = Vec::new();
        let mut expected = Vec::new();
        for i in 0..12 {
            let words: Vec<&str> = (0..5).map(|k| WORDS[(i + k) % WORDS.len()].0).collect();
            audio.extend(sentence(&words, RATE));
            audio.extend(noise(1.0, -60.0, i as u32, RATE));
            expected.extend(words);
        }
        // The model is busy elsewhere while all of it arrives.
        s.push(&audio, RATE);
        assert!(s.uncommitted_secs() < audio.len() as f32 / RATE as f32, "silence between sentences was kept");
        let mut jobs = Vec::new();
        drain(&mut s, &mut jobs);
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), expected.join(" "));
        assert!(jobs.len() < 12, "{} passes for 12 sentences: they should be merged", jobs.len());
        assert!(jobs.iter().all(|j| j.pcm.len() <= 28 * RATE as usize));
    }

    #[test]
    fn silence_and_noise_never_reach_the_model() {
        let mut s = stream();
        let mut audio = silence(20.0, 48_000);
        audio.extend(noise(20.0, -42.0, 5, 48_000));
        audio.extend(silence(5.0, 48_000));
        let mut jobs = run_live(&mut s, &audio, 48_000, 0.5);
        // Memory stays at the short run-up, however long the silence.
        assert!(s.audio.len() + s.preroll_audio.len() <= RATE as usize, "{} samples kept", s.audio.len());
        stop(&mut s, &mut jobs);
        assert!(jobs.is_empty(), "the model was asked to transcribe silence");
        assert_eq!(s.text(), "");
        assert!(!s.heard_speech());
        assert!((s.input_secs() - 45.0).abs() < 0.01);
    }

    #[test]
    fn a_cough_is_dropped_without_asking_the_model() {
        let mut s = stream();
        let mut audio = noise(1.0, -60.0, 1, RATE);
        audio.extend(noise(0.12, -15.0, 2, RATE));
        audio.extend(noise(2.0, -60.0, 3, RATE));
        let mut jobs = run_live(&mut s, &audio, RATE, 0.25);
        stop(&mut s, &mut jobs);
        assert!(jobs.is_empty());
        assert!(s.is_drained());
    }

    #[test]
    fn pause_and_resume() {
        let mut s = stream();
        let mut jobs = run_live(&mut s, &sentence(&["alpha", "bravo"], 48_000), 48_000, 0.25);
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), "alpha bravo");
        s.resume_input();
        let mut more = run_live(&mut s, &sentence(&["charlie"], 16_000), 16_000, 0.25);
        stop(&mut s, &mut more);
        assert_eq!(s.text(), "alpha bravo charlie");
    }

    #[test]
    fn automatic_language_is_kept_once_enough_speech_was_heard() {
        let mut s = LiveStream::new(None, Tuning::default());
        let mut audio = sentence(&["alpha", "bravo"], RATE);
        audio.extend(noise(1.0, -60.0, 1, RATE));
        let mut jobs = run_live(&mut s, &audio, RATE, 0.25);
        // Two short words: detected but not locked.
        assert_eq!(s.language(), Some("en"));
        assert!(jobs.iter().all(|j| j.language.is_none()));
        let words = ["bravo", "charlie", "delta", "echo", "foxtrot", "golf"];
        jobs.extend(run_live(&mut s, &sentence(&words, RATE), RATE, 0.25));
        stop(&mut s, &mut jobs);
        assert!(s.language_locked);
        assert_eq!(jobs.last().unwrap().language.as_deref(), Some("en"), "the locked language is used");
        // A chosen language is always used.
        let mut s = LiveStream::new(Some("de".into()), Tuning::default());
        let mut jobs = run_live(&mut s, &sentence(&["alpha", "bravo"], RATE), RATE, 0.25);
        stop(&mut s, &mut jobs);
        assert!(jobs.iter().all(|j| j.language.as_deref() == Some("de")));
        assert_eq!(s.language(), Some("de"));
    }

    #[test]
    fn the_language_comes_only_from_passes_that_heard_words() {
        let mut s = LiveStream::new(None, Tuning::default());
        // A long sound the voice detector took for speech, in which Whisper "hears" a
        // Japanese subtitle credit.
        let mut audio = sentence(&["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"], RATE);
        audio.extend(noise(1.0, -60.0, 1, RATE));
        s.push(&audio, RATE);
        let job = s.next_job().expect("a final pass");
        assert!(job.commit && job.voiced_secs >= 1.5, "{} s of speech", job.voiced_secs);
        let credit = LivePass {
            text: "ご視聴ありがとうございました".into(),
            avg_logprob: -0.3,
            no_speech_prob: 0.2,
            language: Some("ja".into()),
            tokens: 8,
        };
        s.apply(&job, credit);
        assert_eq!((s.text(), s.language(), s.language_locked), ("", None, false));
        // Real words decide it.
        let mut jobs =
            run_live(&mut s, &sentence(&["golf", "hotel", "alpha", "bravo", "charlie", "delta"], RATE), RATE, 0.25);
        stop(&mut s, &mut jobs);
        assert_eq!(s.text(), "golf hotel alpha bravo charlie delta");
        assert_eq!((s.language(), s.language_locked), (Some("en"), true));
    }

    #[test]
    fn tentative_text_needs_a_word() {
        assert!(!has_word("1.5x2.0x2.0x2.0"));
        assert!(!has_word("2.5 % 7"));
        assert!(has_word("Beep") && has_word("9 o'clock") && has_word("你好"));
        let mut s = stream();
        s.push(&sentence(&["alpha", "bravo", "charlie", "delta"], RATE), RATE);
        let job = std::iter::from_fn(|| s.next_job()).find(|j| !j.commit).expect("a tentative pass");
        let pass = |text: &str| LivePass {
            text: text.into(),
            avg_logprob: -0.3,
            no_speech_prob: 0.05,
            language: Some("en".into()),
            tokens: 8,
        };
        s.apply(&job, pass("1.5x2.0x2.0x2.0"));
        assert_eq!(s.partial(), "", "music's digits aren't shown as words");
        s.apply(&job, pass("alpha bravo"));
        assert_eq!(s.partial(), "alpha bravo");
    }

    #[test]
    fn hallucinations_of_the_model_are_dropped() {
        let pass = |text: &str, ns: f64, lp: f64| LivePass {
            text: text.into(),
            avg_logprob: lp,
            no_speech_prob: ns,
            language: Some("en".into()),
            tokens: 3,
        };
        // Whisper's own silence rule.
        assert_eq!(clean_pass(&pass("Okay, let's go.", 0.7, -1.2), 2.0), None);
        // Stock phrases need confidence.
        assert_eq!(clean_pass(&pass("Thank you.", 0.3, -0.4), 1.0), None);
        assert_eq!(clean_pass(&pass(" you", 0.05, -0.9), 1.0), None);
        assert_eq!(clean_pass(&pass("Thank you.", 0.02, -0.2), 0.8).as_deref(), Some("Thank you."));
        assert_eq!(clean_pass(&pass("Thank you.", 0.02, -0.2), 0.2), None);
        // Sound descriptions and credits.
        assert_eq!(clean_pass(&pass("[BLANK_AUDIO]", 0.01, -0.1), 2.0), None);
        assert_eq!(clean_pass(&pass("(upbeat music)", 0.01, -0.1), 2.0), None);
        assert_eq!(clean_pass(&pass("♪ la la la ♪", 0.01, -0.1), 2.0), None);
        assert_eq!(clean_pass(&pass("Subtitles by the Amara.org community", 0.01, -0.1), 3.0), None);
        assert_eq!(clean_pass(&pass("Hello (laughs) there.", 0.01, -0.1), 2.0).as_deref(), Some("Hello there."));
        assert_eq!(
            clean_pass(&pass("The total (before tax) is ten.", 0.01, -0.1), 2.0).as_deref(),
            Some("The total (before tax) is ten.")
        );
        // Loops collapse to one.
        assert_eq!(
            clean_pass(&pass("I will go. I will go. I will go. I will go.", 0.01, -0.3), 3.0).as_deref(),
            Some("I will go.")
        );
        assert_eq!(clean_pass(&pass("Thank you. Thank you. Thank you.", 0.4, -0.5), 1.0), None);
        assert_eq!(clean_pass(&pass("no no no", 0.01, -0.2), 1.0).as_deref(), Some("no no no"));
        // Far too many words for a short sound.
        assert_eq!(clean_pass(&pass("one two three four five six seven eight", 0.3, -0.5), 0.3), None);
        // Low-confidence blips.
        assert_eq!(clean_pass(&pass("Hmm, the.", 0.1, -1.6), 0.5), None);
        // Ordinary speech passes.
        assert_eq!(
            clean_pass(&pass("Please send the report by Friday.", 0.02, -0.25), 2.5).as_deref(),
            Some("Please send the report by Friday.")
        );
        assert_eq!(clean_pass(&pass("...", 0.02, -0.25), 2.5), None);
        // A sentence end without its space gets one; abbreviations stay as they are.
        assert_eq!(
            clean_pass(&pass("Its own garden.We also met the U.S. team.", 0.02, -0.2), 3.0).as_deref(),
            Some("Its own garden. We also met the U.S. team.")
        );
    }

    #[test]
    fn tentative_text_beyond_a_commit_is_kept() {
        assert_eq!(remainder("Hello there. How are you", "Hello there.").as_deref(), Some("How are you"));
        assert_eq!(remainder("hello, there how", "Hello there").as_deref(), Some("how"));
        assert_eq!(remainder("Hello there", "Hello there.").as_deref(), Some(""));
        assert_eq!(remainder("Yellow there friend", "Hello there"), None);
        assert_eq!(remainder("Hello", "Hello there"), None);
    }

    #[test]
    fn pieces_join_naturally() {
        assert_eq!(join_text("", "Hello."), "Hello.");
        assert_eq!(join_text("Hello.", "How are you?"), "Hello. How are you?");
        assert_eq!(join_text("Wait", ", really?"), "Wait, really?");
        assert_eq!(join_text("你好。", "谢谢"), "你好。谢谢");
        assert_eq!(join_text("Done.", ""), "Done.");
    }
}
