//! Live dictation sessions. Each session has a worker task that hands its audio to the
//! speech model a stretch at a time (see [`super::stream`]) and keeps the text up to date;
//! the HTTP handlers only add audio and read the state, so a slow model never slows down
//! the microphone.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use futures_util::future::BoxFuture;
use serde::Serialize;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::check::CheckCache;
use super::stream::{LiveStream, Tuning};
use crate::inference::whisper::LivePass;

/// Where a session is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// The speech model is being loaded; audio is kept until it is ready.
    Loading,
    /// Taking audio; the text is up to date.
    Listening,
    /// Taking audio; the model is working (or waiting for the model to be free).
    Transcribing,
    /// Not taking audio; everything said before the pause is final.
    Paused,
    /// Stopped; the last words are being made final.
    Finishing,
    /// Stopped; `text` is the final transcript.
    Done,
    /// Something went wrong (`error` says what). The text so far is kept.
    Error,
}

/// What the page shows. The transcript is `text`, then `partial` after a space.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct SessionState {
    pub id: String,
    pub status: SessionStatus,
    /// Final text.
    pub text: String,
    /// Tentative text for the stretch still being worked on ("" when none). It may change.
    pub partial: String,
    /// The language chosen, or the detected one (`None` until known).
    pub language: Option<String>,
    /// Seconds of audio received.
    pub seconds: f64,
    /// The voice detector has heard speech.
    pub speech: bool,
    /// Something louder than a quiet room was heard (steady background noise counts, unlike
    /// for `speech`).
    pub sound: bool,
    /// Audio was received whose words aren't final yet: more text may still come. After a
    /// pause or stop this stays true while the speech model is loading or busy elsewhere.
    pub pending: bool,
    pub error: Option<String>,
}

/// Runs the speech model for sessions: Whisper Base in the app (see `super::BaseEngine`),
/// fakes in tests.
pub trait LiveEngine: Send + Sync + 'static {
    /// Gets the model ready, loading it if needed. Errors are sentences for people.
    fn prepare(&self) -> BoxFuture<'static, Result<(), String>>;
    /// Transcribes 16 kHz audio (see `LoadedWhisper::transcribe_live`). `Ok(None)` means the
    /// model is busy with something else, such as a file transcription: the session keeps
    /// its audio and asks again shortly.
    fn transcribe(
        &self,
        pcm: Vec<f32>,
        language: Option<String>,
        full_window: bool,
    ) -> BoxFuture<'static, Result<Option<LivePass>, String>>;
    /// Called regularly while a session is open, so the idle timer doesn't unload the model.
    fn keep_alive(&self) {}
}

/// How long things may take, and how many sessions there may be.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Sessions open at once (finished ones don't count).
    pub max_sessions: usize,
    /// A session nobody has asked about for this long (closed tab, lost network) is dropped.
    pub idle_expiry: Duration,
    /// A finished session stays readable this long.
    pub done_linger: Duration,
    /// When the limit is reached, recording sessions silent for this long are taken for
    /// abandoned and closed to make room.
    pub abandoned_after: Duration,
    /// How long pause and stop wait for the last words to become final.
    pub pause_wait: Duration,
    pub stop_wait: Duration,
    /// Audio not yet transcribed (the model busy elsewhere) beyond which a session gives up.
    pub max_backlog_secs: f32,
    /// How soon to ask again when the model is busy.
    pub busy_retry: Duration,
    /// How often an idle worker checks the clock.
    pub tick: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_sessions: 4,
            idle_expiry: Duration::from_secs(90),
            done_linger: Duration::from_secs(120),
            abandoned_after: Duration::from_secs(15),
            pause_wait: Duration::from_secs(20),
            stop_wait: Duration::from_secs(45),
            max_backlog_secs: 300.0,
            busy_retry: Duration::from_millis(200),
            tick: Duration::from_millis(250),
        }
    }
}

/// Why a request about a session failed.
#[derive(Clone, Debug, PartialEq)]
pub enum DictationError {
    /// No such session (never existed, discarded or expired).
    NotFound,
    /// Too many sessions at once.
    TooMany,
    /// Whisper Base can't be used; the sentence says why.
    Unavailable(String),
    /// The request itself is wrong (unknown language…).
    BadRequest(String),
    /// The session can't do that any more (resume after stop…).
    NotActive(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Recording,
    Paused,
    Finishing,
    Done,
    Failed,
}

struct Inner {
    phase: Phase,
    stream: LiveStream,
    error: Option<String>,
    model_ready: bool,
    pass_running: bool,
    waiting_for_model: bool,
    last_request: Instant,
    ended_at: Option<Instant>,
    /// Average time of a tentative pass and of a full-window pass (seconds).
    pass_secs: Option<f32>,
    full_pass_secs: Option<f32>,
}

/// Full-window passes at most this slow (seconds) are started early, in the silence after
/// speech (see `LiveStream::set_eager`).
const EAGER_MAX_PASS_SECS: f32 = 1.0;

fn average(avg: Option<f32>, secs: f32) -> f32 {
    match avg {
        Some(a) => a * 0.7 + secs * 0.3,
        None => secs,
    }
}

impl Inner {
    fn status(&self) -> SessionStatus {
        match self.phase {
            Phase::Failed => SessionStatus::Error,
            Phase::Done => SessionStatus::Done,
            Phase::Finishing => SessionStatus::Finishing,
            Phase::Paused => SessionStatus::Paused,
            Phase::Recording if !self.model_ready => SessionStatus::Loading,
            Phase::Recording if self.pass_running || (self.waiting_for_model && self.stream.has_pending_speech()) => {
                SessionStatus::Transcribing
            }
            Phase::Recording => SessionStatus::Listening,
        }
    }

    fn snapshot(&self, id: &str) -> SessionState {
        SessionState {
            id: id.to_string(),
            status: self.status(),
            text: self.stream.text().to_string(),
            partial: self.stream.partial().to_string(),
            language: self.stream.language().map(str::to_string),
            seconds: (self.stream.input_secs() * 100.0).round() / 100.0,
            speech: self.stream.heard_speech(),
            sound: self.stream.heard_sound(),
            pending: self.open() && !self.stream.is_drained(),
            error: self.error.clone(),
        }
    }

    fn fail(&mut self, message: String) {
        self.phase = Phase::Failed;
        self.error = Some(message);
        self.ended_at = Some(Instant::now());
    }

    /// Not finished: it holds the model (and blocks deleting it).
    fn open(&self) -> bool {
        !matches!(self.phase, Phase::Done | Phase::Failed)
    }

    fn expired(&self, limits: &Limits) -> bool {
        match self.ended_at {
            Some(t) => t.elapsed() > limits.done_linger,
            None => self.last_request.elapsed() > limits.idle_expiry,
        }
    }

    fn can_work(&self) -> bool {
        self.model_ready && matches!(self.phase, Phase::Recording | Phase::Paused | Phase::Finishing)
    }

    /// Why the audio waiting for the model piled up (see `Limits::max_backlog_secs`).
    fn backlog_reason(&self) -> &'static str {
        if !self.model_ready {
            "Whisper Base took too long to load"
        } else if self.waiting_for_model || (self.pass_secs.is_none() && self.full_pass_secs.is_none()) {
            "the speech model was busy with something else for too long"
        } else {
            // Its own passes are slower than speech: a slow computer, or one busy with other work.
            "this computer couldn't transcribe as fast as you spoke (it may be busy with other work)"
        }
    }

    /// A stopped session whose last words are final is done.
    fn settle(&mut self) {
        if self.phase == Phase::Finishing && self.model_ready && self.stream.is_drained() {
            self.phase = Phase::Done;
            self.ended_at = Some(Instant::now());
        }
    }

    /// Learns how long passes take. Slow tentative passes run less often (about 1.5 times
    /// the pass time apart, at least a second), so the final text never falls behind; slow
    /// full passes aren't started before a pause is certain.
    fn record_pass(&mut self, full_window: bool, took: Duration) {
        let secs = took.as_secs_f32();
        if full_window {
            let avg = average(self.full_pass_secs, secs);
            self.full_pass_secs = Some(avg);
            self.stream.set_eager(avg <= EAGER_MAX_PASS_SECS);
        } else {
            let avg = average(self.pass_secs, secs);
            self.pass_secs = Some(avg);
            self.stream.set_cadence((avg * 1.5).clamp(1.0, 6.0));
        }
    }
}

struct Session {
    id: String,
    inner: Mutex<Inner>,
    /// Wakes the worker when audio or a request arrives.
    wake: Notify,
    /// Bumped whenever the state changes, for requests waiting on it.
    changed: watch::Sender<u64>,
    cancel: CancellationToken,
}

impl Session {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn state(&self) -> SessionState {
        self.lock().snapshot(&self.id)
    }

    fn bump(&self) {
        self.changed.send_modify(|v| *v = v.wrapping_add(1));
    }

    /// Waits (at most `max`) until `done` holds, or the session ends (discarded, expired or
    /// the app shutting down: nothing more will change, so waiting longer is pointless).
    async fn wait_for(&self, max: Duration, done: impl Fn(&Inner) -> bool) {
        let mut rx = self.changed.subscribe();
        let deadline = Instant::now() + max;
        loop {
            if done(&self.lock()) || self.cancel.is_cancelled() {
                return;
            }
            tokio::select! {
                changed = tokio::time::timeout_at(deadline, rx.changed()) => {
                    if !matches!(changed, Ok(Ok(()))) {
                        return;
                    }
                }
                _ = self.cancel.cancelled() => return,
            }
        }
    }
}

type SessionMap = Arc<Mutex<HashMap<String, Arc<Session>>>>;

fn lock_map(map: &SessionMap) -> MutexGuard<'_, HashMap<String, Arc<Session>>> {
    map.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whose turn it is with the speech model. Sessions queue for it in the order they ask, so
/// two recordings at once share the model: without the queue, a session talking without
/// a break starts its next pass the moment the last one ends and the other waits until it
/// stops (measured: a second recording's last words took 30 s to arrive).
type Turns = Arc<tokio::sync::Mutex<()>>;

/// Live dictation sessions, and what the app knows about Whisper Base's files.
pub struct DictationHub {
    sessions: SessionMap,
    turns: Turns,
    limits: Limits,
    tuning: Tuning,
    /// File checks and remembered load failures (see `super::whisper_base_status`).
    pub(crate) checks: CheckCache,
}

impl Default for DictationHub {
    fn default() -> Self {
        Self::with_limits(Limits::default(), Tuning::default())
    }
}

impl DictationHub {
    pub fn with_limits(limits: Limits, tuning: Tuning) -> Self {
        Self { sessions: Arc::default(), turns: Turns::default(), limits, tuning, checks: CheckCache::default() }
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    fn get(&self, id: &str) -> Result<Arc<Session>, DictationError> {
        let s = lock_map(&self.sessions).get(id).cloned().ok_or(DictationError::NotFound)?;
        if s.lock().expired(&self.limits) {
            self.remove(id);
            return Err(DictationError::NotFound);
        }
        Ok(s)
    }

    fn remove(&self, id: &str) {
        if let Some(s) = lock_map(&self.sessions).remove(id) {
            s.cancel.cancel();
        }
    }

    /// Drops expired sessions.
    pub fn reap(&self) {
        let mut map = lock_map(&self.sessions);
        map.retain(|_, s| {
            let keep = !s.lock().expired(&self.limits);
            if !keep {
                s.cancel.cancel();
            }
            keep
        });
    }

    /// Sessions not finished yet.
    pub fn open_count(&self) -> usize {
        lock_map(&self.sessions).values().filter(|s| s.lock().open()).count()
    }

    /// Whether a session holds Whisper Base (recording, paused or finishing).
    pub fn in_use(&self) -> bool {
        self.open_count() > 0
    }

    /// Whether a session is recording or finishing right now (paused ones aren't).
    pub fn recording(&self) -> bool {
        lock_map(&self.sessions).values().any(|s| matches!(s.lock().phase, Phase::Recording | Phase::Finishing))
    }

    /// Ends every session (shutdown).
    pub fn stop_all(&self) {
        for (_, s) in lock_map(&self.sessions).drain() {
            s.cancel.cancel();
        }
    }

    /// Makes room for one more session in `map`, closing recordings that look abandoned.
    /// The caller adds the new session while still holding the map, so two sessions started
    /// at the same moment can't both take the last place.
    fn make_room(&self, map: &mut HashMap<String, Arc<Session>>) -> Result<(), DictationError> {
        let open = |map: &HashMap<String, Arc<Session>>| map.values().filter(|s| s.lock().open()).count();
        if open(map) < self.limits.max_sessions {
            return Ok(());
        }
        let mut abandoned: Vec<(Instant, String)> = map
            .iter()
            .filter_map(|(id, s)| {
                let i = s.lock();
                (i.phase == Phase::Recording && i.last_request.elapsed() > self.limits.abandoned_after)
                    .then(|| (i.last_request, id.clone()))
            })
            .collect();
        abandoned.sort();
        for (_, id) in abandoned {
            if open(map) < self.limits.max_sessions {
                break;
            }
            if let Some(s) = map.remove(&id) {
                tracing::info!(session = %id, "closing a voice input session that looks abandoned");
                s.cancel.cancel();
            }
        }
        if open(map) < self.limits.max_sessions { Ok(()) } else { Err(DictationError::TooMany) }
    }

    /// Opens a session on `engine`. `language`: a Whisper language code, or `None` to
    /// detect it. `model_ready`: the model is already in memory (no "loading" step).
    pub fn start_with_engine(
        &self,
        engine: Arc<dyn LiveEngine>,
        language: Option<String>,
        model_ready: bool,
    ) -> Result<SessionState, DictationError> {
        self.reap();
        let id = crate::util::new_id();
        let session = Arc::new(Session {
            id: id.clone(),
            inner: Mutex::new(Inner {
                phase: Phase::Recording,
                stream: LiveStream::new(language, self.tuning.clone()),
                error: None,
                model_ready,
                pass_running: false,
                waiting_for_model: false,
                last_request: Instant::now(),
                ended_at: None,
                pass_secs: None,
                full_pass_secs: None,
            }),
            wake: Notify::new(),
            changed: watch::channel(0).0,
            cancel: CancellationToken::new(),
        });
        {
            let mut map = lock_map(&self.sessions);
            self.make_room(&mut map)?;
            map.insert(id.clone(), session.clone());
        }
        let state = session.state();
        let (map, limits, turns) = (self.sessions.clone(), self.limits.clone(), self.turns.clone());
        tokio::spawn(async move {
            let watched = session.clone();
            let run = std::panic::AssertUnwindSafe(worker(map, session, engine, limits, turns));
            // A bug in the worker must not leave a session that says "transcribing" forever
            // (and keeps Whisper Base from being deleted while the page keeps asking).
            if futures_util::FutureExt::catch_unwind(run).await.is_err() {
                tracing::error!(session = %watched.id, "the voice input worker crashed");
                watched
                    .lock()
                    .fail("Voice input stopped because of an unexpected problem. The text so far is kept.".into());
                watched.cancel.cancel();
                watched.bump();
            }
        });
        tracing::debug!(session = %id, "voice input session started");
        Ok(state)
    }

    /// The session's state (also keeps it from expiring).
    pub fn state(&self, id: &str) -> Result<SessionState, DictationError> {
        let s = self.get(id)?;
        let mut i = s.lock();
        i.last_request = Instant::now();
        Ok(i.snapshot(id))
    }

    /// Adds microphone audio (mono samples at `rate` Hz). Ignored unless recording.
    pub fn push_audio(&self, id: &str, samples: &[f32], rate: u32) -> Result<SessionState, DictationError> {
        let s = self.get(id)?;
        let state = {
            let mut i = s.lock();
            i.last_request = Instant::now();
            if i.phase == Phase::Recording {
                i.stream.push(samples, rate);
                if i.stream.uncommitted_secs() > self.limits.max_backlog_secs {
                    let why = i.backlog_reason();
                    i.fail(format!("Voice input stopped: {why}. The text so far is kept."));
                }
            }
            i.snapshot(id)
        };
        s.wake.notify_one();
        s.bump();
        Ok(state)
    }

    /// Stops taking audio and makes what was said so far final (waiting a little for it).
    pub async fn pause(&self, id: &str) -> Result<SessionState, DictationError> {
        let s = self.get(id)?;
        {
            let mut i = s.lock();
            i.last_request = Instant::now();
            if i.phase == Phase::Recording {
                i.stream.finish_input();
                i.phase = Phase::Paused;
            }
        }
        s.wake.notify_one();
        s.bump();
        s.wait_for(self.limits.pause_wait, |i| i.phase != Phase::Paused || i.stream.is_drained()).await;
        Ok(self.state(id).unwrap_or_else(|_| s.state()))
    }

    /// Takes audio again after a pause.
    pub fn resume(&self, id: &str) -> Result<SessionState, DictationError> {
        let s = self.get(id)?;
        let state = {
            let mut i = s.lock();
            i.last_request = Instant::now();
            match i.phase {
                Phase::Paused => {
                    i.stream.resume_input();
                    i.phase = Phase::Recording;
                }
                Phase::Recording => {}
                Phase::Finishing | Phase::Done => {
                    return Err(DictationError::NotActive("This recording has already been stopped.".into()));
                }
                Phase::Failed => {
                    return Err(DictationError::NotActive(
                        i.error.clone().unwrap_or_else(|| "This recording has stopped because of an error.".into()),
                    ));
                }
            }
            i.snapshot(id)
        };
        s.wake.notify_one();
        s.bump();
        Ok(state)
    }

    /// Stops for good: every word received becomes final. Waits for that (at most
    /// `Limits::stop_wait`; after that the state says "finishing" and the text arrives later).
    pub async fn stop(&self, id: &str) -> Result<SessionState, DictationError> {
        let s = self.get(id)?;
        {
            let mut i = s.lock();
            i.last_request = Instant::now();
            match i.phase {
                Phase::Recording => {
                    i.stream.finish_input();
                    i.phase = Phase::Finishing;
                }
                Phase::Paused => i.phase = Phase::Finishing,
                Phase::Finishing | Phase::Done | Phase::Failed => {}
            }
            i.settle();
        }
        s.wake.notify_one();
        s.bump();
        s.wait_for(self.limits.stop_wait, |i| matches!(i.phase, Phase::Done | Phase::Failed)).await;
        Ok(self.state(id).unwrap_or_else(|_| s.state()))
    }

    /// Discards a session and its audio. False when there was no such session.
    pub fn discard(&self, id: &str) -> bool {
        let found = lock_map(&self.sessions).remove(id);
        if let Some(s) = &found {
            s.cancel.cancel();
        }
        found.is_some()
    }
}

/// One session's worker: loads the model, then runs whatever pass the session needs next.
async fn worker(map: SessionMap, s: Arc<Session>, engine: Arc<dyn LiveEngine>, limits: Limits, turns: Turns) {
    let ready = tokio::select! {
        r = engine.prepare() => Some(r),
        _ = s.cancel.cancelled() => None,
    };
    if let Some(ready) = ready {
        {
            let mut i = s.lock();
            match ready {
                Ok(()) => i.model_ready = true,
                Err(e) if i.phase != Phase::Failed => i.fail(e),
                Err(_) => {}
            }
        }
        s.bump();
        run_passes(&s, engine.as_ref(), &limits, &turns).await;
    }
    s.cancel.cancel();
    let mut map = lock_map(&map);
    if map.get(&s.id).is_some_and(|x| Arc::ptr_eq(x, &s)) {
        map.remove(&s.id);
    }
}

async fn run_passes(s: &Session, engine: &dyn LiveEngine, limits: &Limits, turns: &Turns) {
    loop {
        if s.cancel.is_cancelled() {
            return;
        }
        let (job, expired, changed, open) = {
            let mut i = s.lock();
            let expired = i.expired(limits);
            let before = i.phase;
            let job = if expired || !i.can_work() { None } else { i.stream.next_job() };
            if job.is_none() {
                i.settle();
            }
            (job, expired, i.phase != before, i.open())
        };
        if expired {
            tracing::debug!(session = %s.id, "voice input session expired");
            return;
        }
        if changed {
            s.bump();
        }
        if open {
            engine.keep_alive();
        }
        let Some(mut job) = job else {
            tokio::select! {
                _ = s.wake.notified() => {}
                _ = tokio::time::sleep(limits.tick) => {}
                _ = s.cancel.cancelled() => return,
            }
            continue;
        };
        // Wait for this session's turn with the model (another recording is using it).
        let turn = match turns.try_lock() {
            Ok(turn) => turn,
            Err(_) => {
                s.lock().waiting_for_model = true;
                s.bump();
                let turn = tokio::select! {
                    turn = turns.lock() => turn,
                    _ = s.cancel.cancelled() => return,
                };
                // More audio arrived (or the session was paused) while waiting: plan again.
                let mut i = s.lock();
                match if i.can_work() { i.stream.next_job() } else { None } {
                    Some(fresh) => job = fresh,
                    None => {
                        i.waiting_for_model = false;
                        drop(i);
                        s.bump();
                        continue;
                    }
                }
                turn
            }
        };
        s.lock().pass_running = true;
        s.bump();
        let started = Instant::now();
        let pcm = std::mem::take(&mut job.pcm);
        let result = tokio::select! {
            r = engine.transcribe(pcm, job.language.clone(), job.full_window) => r,
            _ = s.cancel.cancelled() => return,
        };
        // The next session in line may use the model while this one takes in the result.
        drop(turn);
        let busy = matches!(result, Ok(None));
        {
            let mut i = s.lock();
            i.pass_running = false;
            i.waiting_for_model = busy;
            match result {
                Ok(Some(out)) => {
                    tracing::debug!(
                        session = %s.id,
                        kind = match (job.commit, job.full_window) {
                            (true, _) => "final",
                            (false, true) => "tentative, full window",
                            (false, false) => "tentative, short window",
                        },
                        audio_secs = job.end as f32 / super::stream::FPS,
                        ms = started.elapsed().as_millis() as u64,
                        cadence_secs = i.stream.cadence(),
                        "voice input pass"
                    );
                    i.record_pass(job.full_window, started.elapsed());
                    i.stream.apply(&job, out);
                    i.settle();
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(session = %s.id, "voice input pass failed: {e}");
                    i.fail(format!("Transcription stopped: {e}. The text so far is kept."));
                }
            }
        }
        s.bump();
        if busy {
            tokio::select! {
                _ = tokio::time::sleep(limits.busy_retry) => {}
                _ = s.cancel.cancelled() => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::stream::tests::{WORDS, fake_transcribe, noise, sentence};
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A model stand-in: hears the test tones, and can be slow, busy or broken on request.
    #[derive(Default)]
    struct FakeEngine {
        load_error: Option<String>,
        load_delay: Duration,
        /// The first this-many passes find the model busy.
        busy_passes: AtomicUsize,
        /// Passes from this number on (1-based) fail; 0: none do.
        fail_from: AtomicUsize,
        /// Passes panic (a bug in the engine).
        crash: bool,
        passes: AtomicUsize,
        keep_alives: AtomicUsize,
    }

    impl LiveEngine for FakeEngine {
        fn prepare(&self) -> BoxFuture<'static, Result<(), String>> {
            let (err, delay) = (self.load_error.clone(), self.load_delay);
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                err.map_or(Ok(()), Err)
            })
        }

        fn transcribe(
            &self,
            pcm: Vec<f32>,
            language: Option<String>,
            _full_window: bool,
        ) -> BoxFuture<'static, Result<Option<LivePass>, String>> {
            let busy = self.busy_passes.load(Ordering::SeqCst) > 0;
            if busy {
                self.busy_passes.fetch_sub(1, Ordering::SeqCst);
                return Box::pin(async { Ok(None) });
            }
            if self.crash {
                return Box::pin(async { panic!("a bug in the speech engine") });
            }
            let n = self.passes.fetch_add(1, Ordering::SeqCst) + 1;
            let from = self.fail_from.load(Ordering::SeqCst);
            let fail = from > 0 && n >= from;
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                if fail {
                    Err("the model ran out of memory".into())
                } else {
                    Ok(Some(fake_transcribe(&pcm, language.as_deref())))
                }
            })
        }

        fn keep_alive(&self) {
            self.keep_alives.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn quick_limits() -> Limits {
        Limits {
            max_sessions: 2,
            idle_expiry: Duration::from_millis(600),
            done_linger: Duration::from_millis(600),
            abandoned_after: Duration::from_millis(300),
            pause_wait: Duration::from_secs(5),
            stop_wait: Duration::from_secs(5),
            max_backlog_secs: 30.0,
            busy_retry: Duration::from_millis(10),
            tick: Duration::from_millis(10),
        }
    }

    fn hub() -> DictationHub {
        DictationHub::with_limits(quick_limits(), Tuning::default())
    }

    /// Sends audio in quarter-second chunks, as the page does (but faster than real time).
    fn send(hub: &DictationHub, id: &str, audio: &[f32], rate: u32) {
        for chunk in audio.chunks(rate as usize / 4) {
            let st = hub.push_audio(id, chunk, rate).unwrap();
            assert!(st.error.is_none(), "{st:?}");
        }
    }

    fn two_sentences(rate: u32) -> Vec<f32> {
        let mut a = noise(0.5, -60.0, 1, rate);
        a.extend(sentence(&["alpha", "bravo", "charlie"], rate));
        a.extend(noise(1.0, -60.0, 2, rate));
        a.extend(sentence(&["delta", "echo"], rate));
        a
    }

    #[tokio::test]
    async fn a_session_goes_from_loading_to_done() {
        let hub = hub();
        let engine = Arc::new(FakeEngine { load_delay: Duration::from_millis(100), ..Default::default() });
        let st = hub.start_with_engine(engine.clone(), None, false).unwrap();
        assert_eq!(st.status, SessionStatus::Loading);
        assert_eq!((st.text.as_str(), st.partial.as_str(), st.speech, st.seconds), ("", "", false, 0.0));
        let id = st.id;
        // Audio sent while the model loads is kept.
        send(&hub, &id, &two_sentences(48_000), 48_000);
        assert!(hub.in_use() && hub.recording());
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Done, "{st:?}");
        assert_eq!(st.text, "alpha bravo charlie delta echo");
        assert_eq!(st.partial, "");
        assert!(st.speech);
        assert_eq!(st.language.as_deref(), Some("en"));
        assert!((st.seconds - 3.61).abs() < 0.02, "{}", st.seconds);
        assert!(engine.keep_alives.load(Ordering::SeqCst) > 0);
        assert!(!hub.in_use());
        // Finished: still readable, nothing more accepted.
        let again = hub.push_audio(&id, &noise(1.0, -20.0, 3, 16_000), 16_000).unwrap();
        assert_eq!(again.seconds, st.seconds);
        assert_eq!(hub.stop(&id).await.unwrap().text, st.text);
        assert!(matches!(hub.resume(&id), Err(DictationError::NotActive(_))));
        // And dropped after a while.
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert_eq!(hub.state(&id), Err(DictationError::NotFound));
    }

    #[tokio::test]
    async fn pause_finalises_and_resume_continues() {
        let hub = hub();
        let id = hub.start_with_engine(Arc::new(FakeEngine::default()), Some("en".into()), true).unwrap().id;
        send(&hub, &id, &sentence(&["golf", "hotel"], 16_000), 16_000);
        let st = hub.pause(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Paused);
        assert_eq!((st.text.as_str(), st.partial.as_str()), ("golf hotel", ""));
        // Audio while paused is ignored.
        let ignored = hub.push_audio(&id, &sentence(&["alpha"], 16_000), 16_000).unwrap();
        assert_eq!(ignored.seconds, st.seconds);
        assert!(hub.in_use() && !hub.recording());
        let st = hub.resume(&id).unwrap();
        assert_eq!(st.status, SessionStatus::Listening);
        send(&hub, &id, &sentence(&["charlie"], 16_000), 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.text, "golf hotel charlie");
    }

    #[tokio::test]
    async fn a_pause_says_when_words_are_still_coming() {
        let hub = DictationHub::with_limits(
            Limits { pause_wait: Duration::from_millis(200), ..quick_limits() },
            Tuning::default(),
        );
        // The model is busy elsewhere (a file transcription): the pause can't wait for the words.
        let engine = Arc::new(FakeEngine { busy_passes: AtomicUsize::new(usize::MAX / 2), ..Default::default() });
        let id = hub.start_with_engine(engine.clone(), Some("en".into()), true).unwrap().id;
        send(&hub, &id, &sentence(&["golf", "hotel"], 16_000), 16_000);
        let st = hub.pause(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Paused);
        assert_eq!((st.text.as_str(), st.partial.as_str(), st.pending, st.speech), ("", "", true, true));
        // Once the model is free, the words arrive while paused, and nothing is pending.
        engine.busy_passes.store(0, Ordering::SeqCst);
        let mut st = hub.state(&id).unwrap();
        for _ in 0..200 {
            if !st.pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            st = hub.state(&id).unwrap();
        }
        assert_eq!((st.status, st.text.as_str(), st.pending), (SessionStatus::Paused, "golf hotel", false));
        // Silence leaves nothing pending either; a finished session never is.
        hub.resume(&id).unwrap();
        send(&hub, &id, &noise(1.0, -60.0, 5, 16_000), 16_000);
        assert!(!hub.pause(&id).await.unwrap().pending);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!((st.status, st.pending, st.text.as_str()), (SessionStatus::Done, false, "golf hotel"));
    }

    #[tokio::test]
    async fn nothing_heard_gives_empty_text() {
        let hub = hub();
        let engine = Arc::new(FakeEngine::default());
        let id = hub.start_with_engine(engine.clone(), None, true).unwrap().id;
        let mut audio = vec![0.0; 16_000 * 2];
        audio.extend(noise(3.0, -45.0, 4, 16_000));
        send(&hub, &id, &audio, 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Done);
        assert_eq!((st.text.as_str(), st.speech, st.language), ("", false, None));
        assert_eq!(engine.passes.load(Ordering::SeqCst), 0, "the model was run on silence");
    }

    #[tokio::test]
    async fn a_busy_model_is_caught_up_with() {
        let hub = hub();
        let engine = Arc::new(FakeEngine { busy_passes: AtomicUsize::new(25), ..Default::default() });
        let id = hub.start_with_engine(engine.clone(), Some("en".into()), true).unwrap().id;
        send(&hub, &id, &two_sentences(16_000), 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.text, "alpha bravo charlie delta echo");
        assert_eq!(engine.busy_passes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn loading_and_transcription_failures_are_reported() {
        let hub = hub();
        let engine =
            Arc::new(FakeEngine { load_error: Some("Whisper Base couldn't be loaded.".into()), ..Default::default() });
        let id = hub.start_with_engine(engine, None, false).unwrap().id;
        send(&hub, &id, &sentence(&["alpha"], 16_000), 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Error);
        assert_eq!(st.error.as_deref(), Some("Whisper Base couldn't be loaded."));
        assert!(!hub.in_use());

        // A pass that fails keeps the text already final.
        let engine = Arc::new(FakeEngine::default());
        let id = hub.start_with_engine(engine.clone(), Some("en".into()), true).unwrap().id;
        let mut audio = sentence(&["alpha", "bravo"], 16_000);
        audio.extend(noise(1.0, -60.0, 1, 16_000));
        send(&hub, &id, &audio, 16_000);
        for _ in 0..200 {
            if hub.state(&id).unwrap().text == "alpha bravo" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        engine.fail_from.store(engine.passes.load(Ordering::SeqCst) + 1, Ordering::SeqCst);
        send(&hub, &id, &sentence(&["charlie", "delta", "echo", "foxtrot"], 16_000), 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Error, "{st:?}");
        assert!(st.error.as_deref().unwrap().contains("ran out of memory"), "{st:?}");
        assert_eq!(st.text, "alpha bravo", "{st:?}");
    }

    #[tokio::test]
    async fn sessions_expire_and_are_limited() {
        let hub = hub();
        let a = hub.start_with_engine(Arc::new(FakeEngine::default()), None, true).unwrap().id;
        let b = hub.start_with_engine(Arc::new(FakeEngine::default()), None, true).unwrap().id;
        // Both were just used: no room for a third.
        assert_eq!(hub.start_with_engine(Arc::new(FakeEngine::default()), None, true), Err(DictationError::TooMany));
        // Paused sessions aren't taken for abandoned; recording ones that went quiet are.
        hub.pause(&a).await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        hub.state(&a).unwrap();
        let c = hub.start_with_engine(Arc::new(FakeEngine::default()), None, true).unwrap().id;
        assert_eq!(hub.state(&b), Err(DictationError::NotFound));
        // A session nobody asks about expires, and its worker ends.
        tokio::time::sleep(Duration::from_millis(800)).await;
        assert_eq!(hub.state(&a), Err(DictationError::NotFound));
        assert_eq!(hub.state(&c), Err(DictationError::NotFound));
        assert_eq!(lock_map(&hub.sessions).len(), 0);
        // Discarding.
        let d = hub.start_with_engine(Arc::new(FakeEngine::default()), None, true).unwrap().id;
        assert!(hub.discard(&d));
        assert!(!hub.discard(&d));
        assert_eq!(hub.state(&d), Err(DictationError::NotFound));
    }

    #[tokio::test]
    async fn waiting_requests_end_when_the_session_does() {
        let limits =
            Limits { stop_wait: Duration::from_secs(30), pause_wait: Duration::from_secs(30), ..quick_limits() };
        let hub = Arc::new(DictationHub::with_limits(limits, Tuning::default()));
        // The model stays busy with something else, so the last words never become final.
        let busy = || Arc::new(FakeEngine { busy_passes: AtomicUsize::new(usize::MAX / 2), ..Default::default() });
        let id = hub.start_with_engine(busy(), Some("en".into()), true).unwrap().id;
        send(&hub, &id, &sentence(&["alpha", "bravo"], 16_000), 16_000);
        let stopping = {
            let (hub, id) = (hub.clone(), id.clone());
            tokio::spawn(async move { hub.stop(&id).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!stopping.is_finished());
        // Discarding it answers the waiting stop at once, not 30 seconds later.
        assert!(hub.discard(&id));
        let st = tokio::time::timeout(Duration::from_secs(2), stopping).await.expect("stop kept waiting").unwrap();
        assert_eq!(st.unwrap().status, SessionStatus::Finishing);

        // The same when the app shuts down during a pause.
        let id = hub.start_with_engine(busy(), Some("en".into()), true).unwrap().id;
        send(&hub, &id, &sentence(&["charlie"], 16_000), 16_000);
        let pausing = {
            let (hub, id) = (hub.clone(), id.clone());
            tokio::spawn(async move { hub.pause(&id).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        hub.stop_all();
        let st = tokio::time::timeout(Duration::from_secs(2), pausing).await.expect("pause kept waiting").unwrap();
        assert_eq!(st.unwrap().status, SessionStatus::Paused);
    }

    #[tokio::test]
    async fn a_crash_in_the_worker_becomes_an_error() {
        let hub = hub();
        let id =
            hub.start_with_engine(Arc::new(FakeEngine { crash: true, ..Default::default() }), None, true).unwrap().id;
        send(&hub, &id, &sentence(&["alpha", "bravo"], 16_000), 16_000);
        let st = hub.stop(&id).await.unwrap();
        assert_eq!(st.status, SessionStatus::Error, "{st:?}");
        assert!(st.error.as_deref().unwrap().contains("unexpected problem"), "{st:?}");
        // It no longer holds Whisper Base (Delete works), and it is still readable for a while.
        assert!(!hub.in_use());
        assert_eq!(hub.state(&id).unwrap().status, SessionStatus::Error);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_session_limit_holds_when_sessions_start_at_once() {
        let hub = Arc::new(hub());
        let starts: Vec<_> = (0..16)
            .map(|_| {
                let hub = hub.clone();
                tokio::task::spawn_blocking(move || hub.start_with_engine(Arc::new(FakeEngine::default()), None, true))
            })
            .collect();
        let mut started = 0;
        for s in starts {
            match s.await.unwrap() {
                Ok(_) => started += 1,
                Err(e) => assert_eq!(e, DictationError::TooMany),
            }
        }
        assert_eq!(started, hub.limits().max_sessions);
        assert_eq!(hub.open_count(), hub.limits().max_sessions);
    }

    /// One model shared by every session, the way Whisper Base is: a pass that finds it
    /// in use is "busy" (as with a file transcription). Records who ran each pass.
    #[derive(Default)]
    struct SharedModel {
        model: std::sync::Mutex<()>,
        passes: std::sync::Mutex<Vec<&'static str>>,
    }

    struct OnShared {
        name: &'static str,
        shared: Arc<SharedModel>,
    }

    impl LiveEngine for OnShared {
        fn prepare(&self) -> BoxFuture<'static, Result<(), String>> {
            Box::pin(async { Ok(()) })
        }

        fn transcribe(
            &self,
            pcm: Vec<f32>,
            language: Option<String>,
            _full_window: bool,
        ) -> BoxFuture<'static, Result<Option<LivePass>, String>> {
            let (name, shared) = (self.name, self.shared.clone());
            Box::pin(async move {
                tokio::task::spawn_blocking(move || {
                    let Ok(_in_use) = shared.model.try_lock() else {
                        return Ok(None);
                    };
                    shared.passes.lock().unwrap().push(name);
                    // Long next to the moment between two passes, as with the real model.
                    std::thread::sleep(Duration::from_millis(40));
                    Ok(Some(fake_transcribe(&pcm, language.as_deref())))
                })
                .await
                .map_err(|e| e.to_string())?
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recordings_take_turns_with_the_model() {
        let limits = Limits {
            idle_expiry: Duration::from_secs(60),
            stop_wait: Duration::from_secs(20),
            max_backlog_secs: 10_000.0,
            busy_retry: Duration::from_millis(50),
            ..quick_limits()
        };
        // Tentative text as often as possible, so A always has work for the model.
        let hub = Arc::new(DictationHub::with_limits(limits, Tuning { cadence: 0.1, ..Tuning::default() }));
        let shared = Arc::new(SharedModel::default());
        let engine = |name| Arc::new(OnShared { name, shared: shared.clone() });
        let a = hub.start_with_engine(engine("a"), Some("en".into()), true).unwrap().id;
        let b = hub.start_with_engine(engine("b"), Some("en".into()), true).unwrap().id;
        // A talks without a break (faster than real time)…
        let words: Vec<&str> = (0..40).map(|i| WORDS[(i + i / WORDS.len()) % WORDS.len()].0).collect();
        let speech = sentence(&words, 16_000);
        let talking = {
            let (hub, a) = (hub.clone(), a.clone());
            tokio::spawn(async move {
                loop {
                    for chunk in speech.chunks(4_000) {
                        if hub.push_audio(&a, chunk, 16_000).is_err() {
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                }
            })
        };
        for _ in 0..500 {
            if shared.passes.lock().unwrap().len() >= 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(shared.passes.lock().unwrap().len() >= 3, "A never kept the model busy");
        // …while B says two words and stops: B's final pass comes next, not after A stops.
        send(&hub, &b, &sentence(&["golf", "hotel"], 16_000), 16_000);
        let before = shared.passes.lock().unwrap().len();
        let st = hub.stop(&b).await.unwrap();
        assert_eq!((st.status, st.text.as_str()), (SessionStatus::Done, "golf hotel"), "{st:?}");
        let passes = shared.passes.lock().unwrap().clone();
        let meanwhile = passes[before..].iter().filter(|n| **n == "a").count();
        assert!(meanwhile <= 2, "B waited for {meanwhile} passes of A: {:?}", &passes[before..]);
        assert!(!talking.is_finished(), "A stopped talking too early for this test to mean anything");
        talking.abort();
    }

    #[tokio::test]
    async fn too_much_waiting_audio_ends_the_session() {
        let hub = hub();
        let engine = Arc::new(FakeEngine { busy_passes: AtomicUsize::new(usize::MAX / 2), ..Default::default() });
        let id = hub.start_with_engine(engine, Some("en".into()), true).unwrap().id;
        let words: Vec<&str> = (0..80).map(|i| ["alpha", "bravo", "charlie"][i % 3]).collect();
        let audio = sentence(&words, 16_000);
        let mut last = None;
        for chunk in audio.chunks(4_000) {
            last = Some(hub.push_audio(&id, chunk, 16_000).unwrap());
        }
        let st = last.unwrap();
        assert_eq!(st.status, SessionStatus::Error);
        assert!(st.error.unwrap().contains("busy with something else"));
    }

    #[test]
    fn a_backlog_says_why_it_built_up() {
        let mut i = Inner {
            phase: Phase::Recording,
            stream: LiveStream::new(None, Tuning::default()),
            error: None,
            model_ready: false,
            pass_running: false,
            waiting_for_model: false,
            last_request: Instant::now(),
            ended_at: None,
            pass_secs: None,
            full_pass_secs: None,
        };
        assert!(i.backlog_reason().contains("load"));
        i.model_ready = true;
        assert!(i.backlog_reason().contains("busy with something else"), "no pass has run yet");
        i.pass_secs = Some(2.5);
        assert!(i.backlog_reason().contains("couldn't transcribe as fast as you spoke"));
        i.waiting_for_model = true;
        assert!(i.backlog_reason().contains("busy with something else"));
    }
}
