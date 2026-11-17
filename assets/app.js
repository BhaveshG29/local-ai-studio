/*
 * Local AI Studio: the app's only script. Every page works without it; it adds two things.
 *
 *  1. Enter-to-send in chat (forms with data-enter-send): Enter sends, Shift+Enter starts a
 *     new line. Empty messages, double sends and chats without a model are refused.
 *  2. Voice input: the chat microphone and the live panel on the Speech models page record
 *     the microphone and send it in small chunks to Local AI Studio, where Whisper Base turns
 *     it into text (src/web/speech.rs). Every word shown comes from the server's answers;
 *     nothing is transcribed or made up here. While a recording is open, leaving the page
 *     asks first and other forms wait (either would end it and lose its text).
 *
 * The same file is the AudioWorklet module that reads the microphone (the first branch
 * below), so the app keeps a single script.
 *
 * Pages load it with `async`: a chat page that streams an answer is parsed to its end only
 * when the answer is complete. So it listens on `document` (event delegation) and sets up
 * each voice-input area as soon as that area has been parsed.
 */
"use strict";

if (typeof registerProcessor === "function") {
  // ---- In the AudioWorklet (the audio thread) -------------------------------------------------
  // Mixes the microphone down to mono and posts it to the page in blocks of about 100 ms.
  // The message "end" posts what is left, answers "flushed" and stops for good.
  registerProcessor(
    "las-mic",
    class extends AudioWorkletProcessor {
      constructor() {
        super();
        this.size = Math.max(128, Math.round(sampleRate / 10));
        this.block = new Float32Array(this.size);
        this.used = 0;
        this.running = true;
        this.port.onmessage = (event) => {
          if (event.data !== "end") return;
          this.post();
          this.running = false;
          this.port.postMessage("flushed");
        };
      }

      post() {
        if (this.used === 0) return;
        const samples = this.used === this.size ? this.block : this.block.slice(0, this.used);
        this.port.postMessage(samples, [samples.buffer]);
        this.block = new Float32Array(this.size);
        this.used = 0;
      }

      process(inputs) {
        const channels = inputs[0];
        if (!this.running || !channels || channels.length === 0) return this.running;
        const count = channels.length;
        const frames = channels[0].length;
        for (let i = 0; i < frames; i++) {
          let sum = 0;
          for (let c = 0; c < count; c++) sum += channels[c][i];
          this.block[this.used++] = sum / count;
          if (this.used === this.size) this.post();
        }
        return true;
      }
    }
  );
} else if (typeof document !== "undefined") {
  // ---- In the page --------------------------------------------------------------------------
  // (Strict mode: everything declared in this block stays in it; nothing becomes global.)

  /** This script's own URL, with its ?v= version: the AudioWorklet loads the same file. */
  const SCRIPT_URL =
    (document.currentScript && document.currentScript.src) || new URL("/assets/app.js", document.baseURI).href;

  /** Seconds of audio per request (the server wants chunks of 250 to 500 ms). */
  const CHUNK_SECONDS = 0.4;
  /** Requests carry at most 2 MiB of samples: the server may refuse more before reading it. */
  const MAX_REQUEST_SAMPLES = (2 * 1024 * 1024) / 4;
  /** While paused, ask about the session this often: it is dropped after 90 s without requests. */
  const KEEP_ALIVE_MS = 30000;
  /** How often to ask whether the last words are final. */
  const POLL_MS = 1000;
  /** Stop waiting for the last words after this long. */
  const FINISH_LIMIT_MS = 3 * 60 * 1000;
  /** Status messages are written (and so announced) a moment later, so a burst of changes is
   * read out once. */
  const MESSAGE_DELAY_MS = 250;
  /** The samples go to the server as little-endian floats; almost every machine is one. */
  const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

  /** The microphone's shortcut (accesskey="m") as this browser spells it. */
  const MIC_KEY = (() => {
    const ua = navigator.userAgent || "";
    if (/Mac|iPhone|iPad|iPod/.test(navigator.platform || "") || /Mac OS X/.test(ua)) return "Ctrl+Option+M";
    if (/Firefox\//.test(ua)) return "Alt+Shift+M";
    return "Alt+M";
  })();

  /** A recording is open in these states. */
  const ACTIVE = new Set(["requesting", "loading", "recording", "paused", "finishing"]);
  /** The text box belongs to the recording (read-only) in these states. */
  const LOCKED = new Set(["requesting", "loading", "recording", "finishing"]);

  const SAY = {
    requesting: "Waiting for microphone permission… Allow the microphone if your browser asks.",
    loading: "Loading Whisper Base… Keep talking; what you say is kept.",
    recording: "Listening. Click the microphone again to stop.",
    paused: "Paused. Your text so far is final; you can edit it or resume.",
    pausedLate: "Paused. Your last words are still being transcribed; they go in when you resume or stop.",
    pausing: "Pausing…",
    finishing: "Finishing…",
    done: "Done. Check the text, then press Enter to send it.",
    silence: "Didn't hear anything. Check that the right microphone is selected and not muted.",
    noWords: "Couldn't make out any words. Try again a little closer to the microphone.",
    noiseOnly: "Heard only background noise, no words. Speak closer to the microphone, or somewhere quieter.",
    interrupted:
      "The browser paused the microphone (a call, or another app, may be using it). Your text so far is kept; resume when you're ready.",
    leaveChat: "Stop voice input first: leaving this page ends the recording, and its text would be lost.",
    leavePanel: "Stop the live transcription first: this reloads the page, which ends the recording and loses its text.",
    tooMany: "Too many recordings are open. Close other tabs that use voice input.",
    badRate:
      "This microphone records at a rate voice input can't use. Pick another microphone, or another format for it, in your system's sound settings.",
    offline: "Lost the connection to Local AI Studio. Your text so far is kept.",
    slow: "The last words are taking too long to arrive. Your text so far is kept.",
    copied: "Copied.",
    notCopied: "Couldn't copy. Select the text and press Ctrl+C (⌘+C on a Mac).",
    cleared: "Cleared.",
  };

  /** The live panel's messages. Its state name shows beside them (LABELS); screen readers
   * hear it at the start of each message instead (see say()). */
  const PANEL_SAY = {
    idle: "Press Start and speak. The text appears below as you talk.",
    requesting: "Allow the microphone if your browser asks.",
    loading: "Loading Whisper Base… what you say meanwhile is kept.",
    recording: "Speak now; the text appears below.",
    paused: SAY.paused,
    pausing: "Making what you said final…",
    finishing: "Making the last words final…",
    done: "Copy the text, download it, or press Start to add more.",
  };

  /** The live panel's state names (exactly these seven are ever shown). */
  const LABELS = {
    idle: "Inactive",
    requesting: "Waiting for microphone permission",
    loading: "Recording",
    recording: "Recording",
    paused: "Inactive",
    finishing: "Processing",
    done: "Completed",
    error: "Error",
    unavailable: "Inactive",
  };

  const SEND_PROBLEMS = {
    model: "Pick a model in Chat settings first.",
    generating: "An answer is still being written. Wait for it, or press Stop.",
    empty: "Type a message or attach a file first.",
  };

  // ---- Errors ---------------------------------------------------------------------------------

  /** The server answered with an error: `message` is its sentence for people. */
  class ApiError extends Error {
    constructor(message, status, code) {
      super(message);
      this.status = status;
      this.code = code || "";
    }
  }

  /** Local AI Studio couldn't be reached (stopped, or the request timed out). */
  class NetworkError extends Error {}

  /** The microphone can't be used; `kind` says why (see micProblemText). */
  class MicError extends Error {
    constructor(kind, detail) {
      super(detail || kind);
      this.kind = kind;
    }
  }

  /** What to do about a microphone problem. */
  function micProblemText(kind, panel, detail) {
    const again = panel ? "then press Start again" : "then click the microphone again";
    const port = location.port ? ":" + location.port : "";
    switch (kind) {
      case "denied":
        return (
          "Microphone access is blocked. Allow it for this site: click the icon left of the address bar, then " +
          `Site settings → Microphone → Allow, ${again}. If the browser itself may not use the microphone, ` +
          "allow it in your system settings too (Windows: Settings → Privacy & security → Microphone; " +
          "macOS: System Settings → Privacy & Security → Microphone; Linux: the input device in your sound settings)."
        );
      case "nodevice":
        return `No microphone was found. Connect one, or check the input device in your system's sound settings, ${again}.`;
      case "busy":
        return `The microphone can't be read: another app is probably using it. Close that app, ${again}.`;
      case "insecure":
        return (
          "Browsers only allow microphones on secure pages. Open Local AI Studio on this computer at " +
          `http://localhost${port} or http://127.0.0.1${port} to use voice input.`
        );
      case "lost":
        return `The microphone stopped: it was unplugged, or access to it was taken away. Your text so far is kept. Check the microphone, ${again}.`;
      case "unsupported":
        return "This browser can't record audio for voice input. Use a current version of Chrome, Edge, Firefox or Safari.";
      default:
        return `The microphone couldn't be started${detail ? ` (${detail})` : ""}. Check it, ${again}.`;
    }
  }

  /** Which microphone problem a getUserMedia() error means. */
  function micErrorKind(error) {
    switch (error && error.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
      case "SecurityError":
        return "denied";
      case "NotFoundError":
      case "DevicesNotFoundError":
      case "OverconstrainedError":
        return "nodevice";
      case "NotReadableError":
      case "TrackStartError":
      case "AbortError":
        return "busy";
      default:
        return "failed";
    }
  }

  /** Why this browser can't record here at all, or null. Microphones need a secure page
   * (localhost counts); elsewhere browsers leave out navigator.mediaDevices. */
  function microphoneSupport() {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
      // On a secure page the browser (or a policy) simply doesn't offer it.
      return window.isSecureContext === false ? "insecure" : "unsupported";
    }
    if (!(window.AudioContext || window.webkitAudioContext) || typeof AudioWorkletNode !== "function") {
      return "unsupported";
    }
    return null;
  }

  let permissionQuery = null;
  /** The microphone's PermissionStatus, or null where browsers can't say. */
  function microphonePermission() {
    if (!permissionQuery) {
      permissionQuery = Promise.resolve()
        .then(() =>
          navigator.permissions && navigator.permissions.query
            ? navigator.permissions.query({ name: "microphone" })
            : null
        )
        .catch(() => null);
    }
    return permissionQuery;
  }

  // ---- Talking to Local AI Studio --------------------------------------------------------------

  /** A JSON request. Resolves to the answer (null for 204); rejects with ApiError or NetworkError. */
  async function api(method, url, body, timeout) {
    const init = { method, credentials: "same-origin", cache: "no-store", headers: {} };
    if (body instanceof ArrayBuffer) {
      init.headers["Content-Type"] = "application/octet-stream";
      init.body = body;
    } else if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const abort = typeof AbortController === "function" ? new AbortController() : null;
    if (abort) init.signal = abort.signal;
    const timer = abort ? setTimeout(() => abort.abort(), timeout || 30000) : 0;
    try {
      let response;
      try {
        response = await fetch(url, init);
      } catch (error) {
        throw new NetworkError(error && error.message);
      }
      if (response.status === 204) return null;
      let data = null;
      try {
        data = await response.json();
      } catch (error) {
        data = null;
      }
      if (response.ok && data) return data;
      const message = (data && data.error) || `Local AI Studio answered with error ${response.status}.`;
      throw new ApiError(message, response.status, data && data.code);
    } finally {
      clearTimeout(timer);
    }
  }

  function sessionUrl(id, action) {
    return "/api/speech/sessions/" + encodeURIComponent(id) + (action ? "/" + action : "");
  }

  /** Ends a session on the server; `keepalive` lets the request outlive a closing page. */
  function discardSession(id, keepalive) {
    try {
      fetch(sessionUrl(id), { method: "DELETE", credentials: "same-origin", keepalive: !!keepalive }).catch(() => {});
    } catch (error) {
      // Nothing to do: the server drops sessions nobody asks about.
    }
  }

  // ---- Audio ------------------------------------------------------------------------------------

  /** Made inside the click that starts recording: browsers only let audio start from a click. */
  function newAudioContext() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    if (ctx.state === "suspended") ctx.resume().catch(() => {});
    return ctx;
  }

  function closeContext(ctx) {
    if (ctx && ctx.state !== "closed") ctx.close().catch(() => {});
  }

  /** Releases the microphone (the browser's recording sign goes out) and the audio graph. */
  function closeCapture(capture) {
    if (capture.closed) return;
    capture.closed = true;
    clearTimeout(capture.stallTimer);
    capture.ctx.onstatechange = null;
    for (const track of capture.stream.getTracks()) track.stop();
    for (const node of [capture.source, capture.node, capture.mute]) {
      try {
        if (node) node.disconnect();
      } catch (error) {
        // Already disconnected.
      }
    }
    if (capture.node) capture.node.port.onmessage = null;
    closeContext(capture.ctx);
  }

  /** Asks the worklet for the samples it still holds; resolves once they have arrived. */
  function flushWorklet(capture) {
    return new Promise((resolve) => {
      const finish = () => {
        clearTimeout(timer);
        capture.flushed = null;
        resolve();
      };
      // A suspended audio thread may never answer; a few lost milliseconds are fine then.
      const timer = setTimeout(finish, 1000);
      capture.flushed = finish;
      try {
        capture.node.port.postMessage("end");
      } catch (error) {
        finish();
      }
    });
  }

  function concat(parts, length) {
    if (parts.length === 1) return parts[0];
    const out = new Float32Array(length);
    let at = 0;
    for (const part of parts) {
      out.set(part, at);
      at += part.length;
    }
    return out;
  }

  /** The request body: the samples as little-endian 32-bit floats. */
  function audioBody(samples) {
    if (!LITTLE_ENDIAN) {
      const view = new DataView(new ArrayBuffer(samples.length * 4));
      samples.forEach((value, i) => view.setFloat32(i * 4, value, true));
      return view.buffer;
    }
    const whole = samples.byteOffset === 0 && samples.byteLength === samples.buffer.byteLength;
    return whole ? samples.buffer : samples.slice().buffer;
  }

  // ---- Text -------------------------------------------------------------------------------------

  /** What was said since `base` (the session's text length when this stretch began). */
  function spokenText(state, base) {
    const final = (state.text || "").slice(base).trim();
    const partial = (state.partial || "").trim();
    return final && partial ? final + " " + partial : final || partial;
  }

  /** The text box's value with `spoken` between the text before and after it, with spaces
   * where words would otherwise run together, and the selection to show. Text that was
   * selected when the recording began stays (and stays selected) until words come to
   * replace it, so a recording that hears nothing changes nothing. */
  function compose(segment, spoken) {
    if (!spoken) {
      const start = segment.prefix.length;
      return { value: segment.prefix + segment.selected + segment.suffix, start, end: start + segment.selected.length };
    }
    const before = segment.prefix && !/\s$/.test(segment.prefix) ? " " : "";
    const after = segment.suffix && !/^[\s.,;:!?)\]}]/.test(segment.suffix) ? " " : "";
    const head = segment.prefix + before + spoken;
    return { value: head + after + segment.suffix, start: head.length, end: head.length };
  }

  function setSelection(box, start, end) {
    try {
      box.setSelectionRange(start, end);
    } catch (error) {
      // Not a text field any more.
    }
    box.lasCaretSet = true;
    if (end >= box.value.length) box.scrollTop = box.scrollHeight;
  }

  function clockText(ms) {
    const seconds = Math.floor(ms / 1000);
    return Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0");
  }

  /** transcript-YYYY-MM-DD-HHMM.txt, in local time. */
  function transcriptName(date) {
    const two = (n) => String(n).padStart(2, "0");
    const day = `${date.getFullYear()}-${two(date.getMonth() + 1)}-${two(date.getDate())}`;
    return `transcript-${day}-${two(date.getHours())}${two(date.getMinutes())}.txt`;
  }

  function setIcon(button, name) {
    const use = button.querySelector("svg use");
    const href = "/assets/icons.svg#" + name;
    if (use && use.getAttribute("href") !== href) use.setAttribute("href", href);
  }

  function setText(element, text) {
    if (element && element.textContent !== text) element.textContent = text;
  }

  /** aria-disabled rather than disabled: the button keeps keyboard focus while it waits. */
  function setAriaDisabled(element, off) {
    if (!element) return;
    if (off) element.setAttribute("aria-disabled", "true");
    else element.removeAttribute("aria-disabled");
  }

  function copyBySelecting(box) {
    const { selectionStart, selectionEnd } = box;
    box.focus();
    box.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (error) {
      ok = false;
    }
    box.setSelectionRange(selectionStart, selectionEnd);
    return ok;
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // ---- Voice input ------------------------------------------------------------------------------

  /**
   * Voice input for one [data-dictation] area: the chat composer, or the live panel
   * (data-dictation-mode="panel"). The area's data-state says where it is (CSS shows the
   * controls from it): idle, requesting, loading, recording, paused, finishing, done, error,
   * unavailable.
   *
   * Recording: the microphone feeds an AudioWorklet; its samples are cut into chunks of
   * CHUNK_SECONDS and sent one request at a time; each answer is the session's state, whose
   * text replaces what this recording put in the box so far.
   */
  class Dictation {
    constructor(root) {
      this.root = root;
      this.panel = root.getAttribute("data-dictation-mode") === "panel";
      this.state = "idle";
      this.whisperProblem = false; // Whisper Base isn't ready: the microphone never records
      this.micProblem = null; // the microphone can't be used: "denied", "nodevice"…
      this.micProblemText = "";
      this.session = null; // { id } of the server session; null while (re)starting it
      this.last = null; // the session's last state
      this.segment = null; // where the words go: { prefix, selected, suffix, base }
      this.selection = null; // [start, end] in the box once the words are in
      this.capture = null; // { ctx, stream, source, node, mute, rate, closed }
      this.pending = []; // samples not yet cut into a chunk
      this.pendingLength = 0;
      this.queue = []; // chunks waiting to be sent: { samples, rate }
      this.sending = false;
      this.drainWaiters = [];
      this.run = 0; // bumped to call off a start or resume that is still on its way
      this.pendingCtx = null; // the AudioContext of a start or resume still waiting for the microphone
      this.pausing = null; // the pause under way (a promise)
      this.pausingNow = false; // …while it can still be cut short by Stop
      this.cutPause = false; // Stop came during the pause: it ends without settling
      this.stopping = null;
      this.shown = 0; // how much of the session's final text the box has (its length)
      this.heard = false; // some final text came back during this recording
      this.speech = false; // the server's voice detector heard speech
      this.sound = false; // …or at least background noise (the microphone works)
      this.busy = false; // tentative text is being worked on ("Transcribing")
      this.elapsed = 0;
      this.since = 0;
      this.clockTimer = 0;
      this.keepAliveTimer = 0;
      this.message = "";
      this.messageTimer = 0;
      this.notice = ""; // the message is why a send was refused (see notify())
      this.checkedMicrophone = false;
    }

    // -- The area's parts, looked up each time (cheap; a streamed page may add them late).

    part(selector) {
      return this.root.querySelector(selector);
    }

    get output() {
      return this.part("[data-dictation-output]");
    }

    get toggle() {
      return this.part('[data-dictation-action="toggle"]');
    }

    get pauseButton() {
      return this.part('[data-dictation-action="pause"], [data-dictation-action="resume"]');
    }

    active() {
      return ACTIVE.has(this.state);
    }

    /** Sets the area up once it is in the page (safe to repeat). */
    prepare() {
      // Without the script these buttons open a "needs JavaScript" pop-up.
      for (const button of this.root.querySelectorAll("[data-dictation-action][popovertarget]")) {
        button.removeAttribute("popovertarget");
      }
      if (this.part("[data-dictation-unavailable]")) this.whisperProblem = true;
      if (this.whisperProblem && this.state === "idle") this.state = "unavailable";
      if (!this.whisperProblem && !this.checkedMicrophone) {
        this.checkedMicrophone = true;
        this.checkMicrophone();
      }
      this.refresh();
    }

    /** Shows a microphone that can't work before anyone clicks it (no permission, not a
     * secure page). The chat shows it on the button only; the panel also says so. */
    checkMicrophone() {
      const problem = microphoneSupport();
      if (problem) {
        this.showMicProblem(problem, !this.panel);
        return;
      }
      microphonePermission().then((status) => {
        if (!status) return;
        const update = () => {
          if (this.active() || this.whisperProblem) return;
          if (status.state === "denied") {
            // Quiet only while nothing else is being said (at load); never wipes a message
            // that already explains the problem (access taken away mid-recording).
            if (this.micProblem !== "denied") this.showMicProblem("denied", !this.panel && !this.message);
          } else if (this.micProblem === "denied") {
            this.clearMicProblem();
          }
        };
        update();
        status.addEventListener("change", update);
      });
    }

    act(action) {
      switch (action) {
        case "toggle":
          if (this.active()) this.stop(false);
          else this.start();
          break;
        case "start":
          this.start();
          break;
        case "pause":
          this.pause();
          break;
        case "resume":
          this.resume();
          break;
        case "stop":
          this.stop(false);
          break;
        case "copy":
          this.copy();
          break;
        case "clear":
          this.clear();
          break;
        case "download":
          this.download();
          break;
      }
    }

    // -- Start --------------------------------------------------------------------------------

    start() {
      if (this.whisperProblem || this.active()) return;
      const problem = microphoneSupport();
      if (problem) {
        this.showMicProblem(problem, false);
        return;
      }
      let ctx;
      try {
        ctx = newAudioContext();
      } catch (error) {
        this.showMicProblem("unsupported", false);
        return;
      }
      const run = ++this.run;
      this.pendingCtx = ctx;
      this.micProblem = null;
      this.heard = false;
      this.speech = false;
      this.sound = false;
      this.busy = false;
      this.elapsed = 0;
      this.session = null;
      this.last = null;
      this.shown = 0;
      this.beginSegment(0);
      // (The panel's Start hides itself now; keepFocus() moves the focus on to Stop.)
      this.setState("requesting");
      this.begin(run, ctx).catch((error) => this.fail(error, run));
    }

    /** Opens the session first (so nobody is asked for the microphone when Whisper Base
     * can't run), then the microphone. */
    async begin(run, ctx) {
      let state;
      try {
        state = await api("POST", "/api/speech/sessions", { language: this.language() });
      } catch (error) {
        closeContext(ctx);
        throw error;
      }
      if (run !== this.run) {
        closeContext(ctx);
        discardSession(state.id);
        return;
      }
      this.session = { id: state.id };
      this.last = state;
      let capture;
      try {
        capture = await this.openCapture(ctx, run);
      } catch (error) {
        closeContext(ctx);
        throw error;
      }
      if (!capture || run !== this.run) {
        if (capture) closeCapture(capture);
        else closeContext(ctx);
        return;
      }
      this.pendingCtx = null;
      this.capture = capture;
      this.startClock();
      this.setState(this.last && this.last.status === "loading" ? "loading" : "recording");
      this.pump();
    }

    /** The microphone into an AudioWorklet. The graph ends in a silent gain, so nothing is
     * ever heard, but browsers keep the worklet running. */
    async openCapture(ctx, run) {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      } catch (error) {
        throw new MicError(micErrorKind(error), error && error.message);
      }
      if (run !== this.run) {
        // Called off while the browser asked for permission: let the microphone go at once.
        for (const track of stream.getTracks()) track.stop();
        return null;
      }
      const capture = { ctx, stream, rate: Math.round(ctx.sampleRate), closed: false, flushed: null, stallTimer: 0 };
      try {
        await ctx.audioWorklet.addModule(SCRIPT_URL);
        capture.source = ctx.createMediaStreamSource(stream);
        capture.node = new AudioWorkletNode(ctx, "las-mic", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        });
        capture.mute = ctx.createGain();
        capture.mute.gain.value = 0;
        capture.node.port.onmessage = (event) => this.onWorklet(capture, event.data);
        capture.source.connect(capture.node);
        capture.node.connect(capture.mute);
        capture.mute.connect(ctx.destination);
      } catch (error) {
        for (const track of stream.getTracks()) track.stop();
        throw new MicError("failed", error && error.message);
      }
      for (const track of stream.getAudioTracks()) {
        track.addEventListener("ended", () => this.onTrackEnded(capture));
        track.addEventListener("mute", () => this.watchStall(capture));
      }
      ctx.onstatechange = () => this.onContextChange(capture);
      if (ctx.state === "suspended") ctx.resume().catch(() => {});
      return capture;
    }

    /** The browser suspended or interrupted the audio (Safari does for a phone call, Siri or
     * another app taking the microphone): carry on if it can; otherwise pause, so the page
     * never claims to listen while nothing is recorded. */
    onContextChange(capture) {
      if (capture.closed || capture !== this.capture) return;
      const state = capture.ctx.state;
      if (state === "suspended" || state === "interrupted") {
        capture.ctx.resume().catch(() => {});
        this.watchStall(capture);
      }
    }

    /** Audio stopped flowing (the context paused, or the track muted): give it a second to
     * come back, then pause. */
    watchStall(capture) {
      if (capture.closed || capture !== this.capture || capture.stallTimer) return;
      capture.stallTimer = setTimeout(() => {
        capture.stallTimer = 0;
        if (capture.closed || capture !== this.capture) return;
        const muted = capture.stream.getAudioTracks().some((track) => track.muted);
        if (capture.ctx.state === "running" && !muted) return;
        if (this.state === "recording" || this.state === "loading") this.pause(SAY.interrupted);
      }, 1000);
    }

    onWorklet(capture, data) {
      if (data === "flushed") {
        if (capture.flushed) capture.flushed();
        return;
      }
      if (capture.closed || !(data instanceof Float32Array)) return;
      this.pending.push(data);
      this.pendingLength += data.length;
      if (this.pendingLength >= capture.rate * CHUNK_SECONDS) this.cutChunk(capture.rate);
    }

    /** The microphone went away mid-recording (unplugged, or permission taken back): keep
     * what was heard, like Stop, then say what happened. */
    onTrackEnded(capture) {
      if (capture.closed || capture !== this.capture) return;
      this.stop(false).then((ok) => {
        if (!ok) return;
        microphonePermission().then((status) => {
          if (!this.active()) this.showMicProblem(status && status.state === "denied" ? "denied" : "lost", false);
        });
      });
    }

    // -- Sending audio ----------------------------------------------------------------------

    cutChunk(rate) {
      if (this.pendingLength === 0) return;
      const samples = concat(this.pending, this.pendingLength);
      this.pending = [];
      this.pendingLength = 0;
      this.queue.push({ samples, rate });
      this.pump();
    }

    /** The next request: one chunk, or several merged when the queue has backed up. */
    takeChunk() {
      const first = this.queue.shift();
      const parts = [first.samples];
      let length = first.samples.length;
      while (
        this.queue.length > 0 &&
        this.queue[0].rate === first.rate &&
        length + this.queue[0].samples.length <= MAX_REQUEST_SAMPLES
      ) {
        const next = this.queue.shift();
        parts.push(next.samples);
        length += next.samples.length;
      }
      return { samples: concat(parts, length), rate: first.rate };
    }

    /** Sends the queue, one request at a time. Audio waits while there is no session. */
    async pump() {
      if (this.sending) return;
      this.sending = true;
      let run = this.run;
      // New sessions since the server last took audio: one that keeps losing them isn't
      // asked for new ones forever.
      let replaced = 0;
      try {
        while (this.queue.length > 0 && this.session) {
          run = this.run;
          const session = this.session;
          const chunk = this.takeChunk();
          let state;
          try {
            const url = sessionUrl(session.id, "audio") + "?rate=" + chunk.rate;
            state = await api("POST", url, audioBody(chunk.samples), 30000);
          } catch (error) {
            if (error instanceof ApiError && error.status === 404 && this.session === session && replaced < 2) {
              // The server dropped the session (it expired): carry on in a new one.
              replaced++;
              this.queue.unshift(chunk);
              await this.replaceSession();
              continue;
            }
            throw error;
          }
          replaced = 0;
          this.applyState(state);
        }
      } catch (error) {
        // A request of a recording that has ended meanwhile mustn't stop a newer one.
        this.fail(error, run);
      } finally {
        this.sending = false;
        this.drained();
      }
    }

    /** Resolves once every chunk has been sent. */
    drain() {
      if (!this.sending && this.queue.length === 0) return Promise.resolve();
      return new Promise((resolve) => this.drainWaiters.push(resolve));
    }

    drained() {
      if (this.sending || this.queue.length > 0) return;
      const waiters = this.drainWaiters;
      this.drainWaiters = [];
      for (const resolve of waiters) resolve();
    }

    async replaceSession() {
      const run = this.run;
      this.commitSegment();
      this.session = null;
      this.shown = 0;
      const state = await api("POST", "/api/speech/sessions", { language: this.language() });
      if (run !== this.run) {
        discardSession(state.id);
        return;
      }
      this.session = { id: state.id };
      this.last = state;
    }

    /** Stops recording and sends what is left; the session stays open. */
    async endCapture() {
      const capture = this.capture;
      if (!capture) return;
      await flushWorklet(capture);
      if (this.capture === capture) this.capture = null;
      if (capture.closed) return;
      this.cutChunk(capture.rate);
      closeCapture(capture);
    }

    // -- What the server says -----------------------------------------------------------------

    applyState(state) {
      if (!state || !this.session || state.id !== this.session.id) return;
      this.last = state;
      if (state.status === "error") {
        this.fail(new ApiError(state.error || "Transcription stopped. Your text so far is kept.", 200, "failed"));
        return;
      }
      if (this.state === "loading" && state.status !== "loading") this.setState("recording");
      else if (this.state === "recording" && state.status === "loading") this.setState("loading");
      if (this.state === "loading" || this.state === "recording" || this.state === "finishing") this.show(state);
      this.setBusy(this.state === "recording" && (state.status === "transcribing" || !!state.partial));
    }

    /** Puts what was said into the box, between the text that was before and after it. */
    show(state) {
      const box = this.output;
      if (!box || !this.segment) return;
      if ((state.text || "").slice(this.segment.base).trim()) this.heard = true;
      if (state.speech) this.speech = true;
      if (state.sound) this.sound = true;
      this.shown = (state.text || "").length;
      const { value, start, end } = compose(this.segment, spokenText(state, this.segment.base));
      this.selection = [start, end];
      if (box.value === value) return;
      box.value = value;
      setSelection(box, start, end);
      box.dispatchEvent(new Event("input", { bubbles: true }));
    }

    /** New words go where the caret is (the chat; they replace the selection, once there are
     * some, unless `keepSelected`: then they go after it) or at the end (the panel). `base`:
     * the session's text length now; only what comes after it is new. */
    beginSegment(base, keepSelected) {
      const box = this.output;
      if (!box) {
        this.segment = null;
        return;
      }
      const value = box.value;
      let from = value.length;
      let to = value.length;
      // Before anyone has touched the box its caret may sit at the start: add at the end.
      if (!this.panel && (box.lasTouched || box.lasCaretSet)) {
        from = keepSelected ? box.selectionEnd : box.selectionStart;
        to = box.selectionEnd;
      }
      this.segment = { prefix: value.slice(0, from), selected: value.slice(from, to), suffix: value.slice(to), base };
      this.selection = [from, to];
    }

    /** Keeps what is in the box as it is, for a new session that starts from nothing. */
    commitSegment() {
      const box = this.output;
      if (!box || !this.segment) return;
      const value = box.value;
      const [start, end] = this.selection || [value.length, value.length];
      this.segment = { prefix: value.slice(0, start), selected: value.slice(start, end), suffix: value.slice(end), base: 0 };
    }

    // -- Pause and resume ---------------------------------------------------------------------

    /** `reason`: what to say instead of the usual "Paused." (the browser paused the audio). */
    pause(reason) {
      if (this.state !== "recording" && this.state !== "loading") return;
      const run = this.run;
      this.cutPause = false;
      this.pausingNow = true; // (before doPause shows "Pausing…": Stop stays usable meanwhile)
      this.pausing = this.doPause(run, reason)
        .catch((error) => this.fail(error, run))
        .finally(() => {
          this.pausing = null;
          this.pausingNow = false;
        });
    }

    async doPause(run, reason) {
      this.setState("finishing", this.panel ? PANEL_SAY.pausing : SAY.pausing);
      this.setBusy(false);
      this.stopClock();
      await this.endCapture();
      await this.drain();
      // Stop came meanwhile (cutPause): it takes over from here.
      if (run !== this.run || this.cutPause) return;
      const session = this.session;
      let state = null;
      if (session) {
        try {
          state = await api("POST", sessionUrl(session.id, "pause"), undefined, 40000);
          // The server waits up to 20 s for the words so far. While Whisper Base is still
          // loading or busy elsewhere (a file transcription), they come later (`pending`):
          // keep asking for a while.
          const deadline = Date.now() + FINISH_LIMIT_MS;
          while (state.status === "paused" && (state.partial || state.pending) && Date.now() < deadline) {
            this.show(state);
            await sleep(POLL_MS);
            if (run !== this.run || this.cutPause) return;
            state = await api("GET", sessionUrl(session.id));
          }
        } catch (error) {
          if (!(error instanceof ApiError && error.status === 404)) throw error;
          this.session = null; // it expired: Resume starts a new one
          state = null;
        }
      }
      if (run !== this.run || this.cutPause) return;
      if (state && state.status === "error") {
        throw new ApiError(state.error || "Transcription stopped. Your text so far is kept.", 200, "failed");
      }
      if (state) this.show(state);
      // Words still to come go in when the recording resumes or stops (see carryOn, finish).
      const late = !!(state && state.status === "paused" && state.pending);
      this.setState("paused", reason || (late ? SAY.pausedLate : undefined));
      this.startKeepAlive();
    }

    resume() {
      if (this.state !== "paused") return;
      let ctx;
      try {
        ctx = newAudioContext();
      } catch (error) {
        this.showMicProblem("unsupported", false);
        return;
      }
      const run = ++this.run;
      this.pendingCtx = ctx;
      this.stopKeepAlive();
      this.setState("requesting");
      this.carryOn(run, ctx).catch((error) => this.fail(error, run));
    }

    /** The microphone again, then the same session (or a new one if it has ended). Text
     * changed while paused stays; new words go where the caret is now. */
    async carryOn(run, ctx) {
      let capture;
      try {
        capture = await this.openCapture(ctx, run);
      } catch (error) {
        closeContext(ctx);
        throw error;
      }
      if (!capture || run !== this.run) {
        if (capture) closeCapture(capture);
        else closeContext(ctx);
        return;
      }
      this.pendingCtx = null;
      const session = this.session;
      this.session = null; // the audio waits until the server takes it again
      this.capture = capture;
      this.beginSegment(0);
      this.startClock();
      let state = null;
      let resumed = false;
      if (session) {
        try {
          state = await api("POST", sessionUrl(session.id, "resume"));
          resumed = true;
        } catch (error) {
          if (!(error instanceof ApiError && (error.status === 404 || error.status === 409))) {
            if (run === this.run) this.session = session;
            throw error;
          }
          if (error.status === 409) discardSession(session.id);
        }
        if (run !== this.run) {
          discardSession(session.id);
          return;
        }
      }
      if (!state) {
        state = await api("POST", "/api/speech/sessions", { language: this.language() });
        if (run !== this.run) {
          discardSession(state.id);
          return;
        }
      }
      // The same session: words that became final only after the pause (Whisper Base was
      // busy) go in first, where the caret is, then the new ones. A new session: from nothing.
      if (!resumed) this.shown = 0;
      if (this.segment) this.segment.base = Math.min(this.shown, state.text ? state.text.length : 0);
      this.session = { id: state.id };
      this.last = state;
      this.setState(state.status === "loading" ? "loading" : "recording");
      this.pump();
    }

    startKeepAlive() {
      this.stopKeepAlive();
      this.keepAliveTimer = setInterval(() => this.keepAlive(), KEEP_ALIVE_MS);
    }

    stopKeepAlive() {
      clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = 0;
    }

    /** While paused: asking keeps the session from expiring. The box isn't touched (it may
     * have been edited). */
    keepAlive() {
      const session = this.session;
      if (this.state !== "paused" || !session) return;
      api("GET", sessionUrl(session.id), undefined, 15000).then(
        (state) => {
          if (state.status === "error" && this.session === session) {
            this.session = null;
            discardSession(session.id);
          }
        },
        (error) => {
          if (error instanceof ApiError && error.status === 404 && this.session === session) this.session = null;
        }
      );
    }

    // -- Stop ---------------------------------------------------------------------------------

    /** Stops recording and puts the final text in the box. Never sends the message. Resolves
     * to false if that failed (the problem is shown and the text kept). */
    stop(forSend) {
      if (this.stopping) return this.stopping;
      if (!this.active()) return Promise.resolve(true);
      this.stopping = this.finish(!!forSend)
        .catch((error) => {
          this.fail(error);
          return false;
        })
        .finally(() => {
          this.stopping = null;
        });
      return this.stopping;
    }

    async finish(forSend) {
      if (this.pausing) {
        // Stop while pausing: the pause needn't settle first (it may wait for a busy model).
        this.cutPause = true;
        this.pausingNow = false;
        this.refresh();
        this.say(this.panel ? PANEL_SAY.finishing : SAY.finishing);
        await this.pausing;
      }
      if (this.state === "requesting") return this.cancel();
      if (!this.active()) return this.state !== "error";
      const run = this.run;
      const wasPaused = this.state === "paused";
      this.setState("finishing", this.panel ? PANEL_SAY.finishing : SAY.finishing);
      this.setBusy(false);
      this.stopClock();
      this.stopKeepAlive();
      await this.endCapture();
      await this.drain();
      if (run !== this.run) return false;
      const session = this.session;
      let state = null;
      if (session) {
        try {
          state = await this.finalState(session, run, !wasPaused);
        } catch (error) {
          // A session that has ended already leaves the text as it is.
          if (!(error instanceof ApiError && error.status === 404)) throw error;
        }
        if (run !== this.run) return false;
        this.session = null;
        if (state) discardSession(session.id);
      }
      if (state && state.status === "error") {
        throw new ApiError(state.error || "Transcription stopped. Your text so far is kept.", 200, "failed");
      }
      if (state && !wasPaused) this.show(state);
      else if (state && (state.text || "").length > this.shown) {
        // After a pause the box may have been edited, and nothing new was said; but words
        // that became final only after the pause (Whisper Base was busy) go in, at the caret.
        this.beginSegment(this.shown, true);
        this.show(state);
      }
      const speech = this.speech || !!(state && state.speech);
      const sound = this.sound || !!(state && state.sound);
      let message = this.panel ? PANEL_SAY.done : SAY.done;
      if (!this.heard) message = speech ? SAY.noWords : sound ? SAY.noiseOnly : SAY.silence;
      this.setState("done", message);
      if (!forSend) this.focusOutput();
      return true;
    }

    /** Stop's answer, then the session until its last words are final. */
    async finalState(session, run, render) {
      const deadline = Date.now() + FINISH_LIMIT_MS;
      let state = await api("POST", sessionUrl(session.id, "stop"), undefined, 60000);
      while (state.status === "finishing") {
        if (Date.now() > deadline) throw new ApiError(SAY.slow, 0, "slow");
        if (render) this.show(state);
        await sleep(POLL_MS);
        if (run !== this.run) return state;
        state = await api("GET", sessionUrl(session.id));
      }
      return state;
    }

    /** Calls off a start or resume still waiting for the microphone: nothing new was said. */
    cancel() {
      const hadText = this.heard;
      this.halt();
      if (hadText) this.setState("done", this.panel ? PANEL_SAY.done : SAY.done);
      else this.setState("idle", this.panel ? PANEL_SAY.idle : "");
      return true;
    }

    /** Ends everything at once: microphone, queue, timers and the session. Keeps the text. */
    halt() {
      this.run++;
      if (this.pendingCtx) closeContext(this.pendingCtx);
      this.pendingCtx = null;
      const capture = this.capture;
      this.capture = null;
      if (capture) closeCapture(capture);
      this.pending = [];
      this.pendingLength = 0;
      this.queue = [];
      this.drained();
      this.stopClock();
      this.stopKeepAlive();
      this.busy = false;
      if (this.session) {
        discardSession(this.session.id);
        this.session = null;
      }
    }

    /** Something went wrong: stop, keep the text, say what happened. `run`: the start or
     * resume that failed (ignored if it was called off meanwhile). */
    fail(error, run) {
      if (run !== undefined && run !== this.run) return;
      this.halt();
      if (error instanceof MicError) {
        this.showMicProblem(error.kind, false, error.message);
        return;
      }
      this.micProblem = null;
      if (error instanceof ApiError && error.code === "whisper_unavailable") {
        this.showWhisperProblem(error.message);
      } else if (error instanceof ApiError && error.code === "too_many") {
        this.setState("error", SAY.tooMany);
      } else if (error instanceof ApiError && error.code === "bad_rate") {
        this.setState("error", SAY.badRate);
      } else if (error instanceof NetworkError) {
        this.setState("error", SAY.offline);
      } else if (error instanceof ApiError) {
        this.setState("error", error.message);
      } else {
        const detail = error && error.message ? ` (${error.message})` : "";
        this.setState("error", `Voice input stopped because of an unexpected problem${detail}. Your text so far is kept.`);
      }
    }

    // -- Problems with the microphone or Whisper Base ------------------------------------------

    showMicProblem(kind, quiet, detail) {
      this.micProblem = kind;
      this.micProblemText = micProblemText(kind, this.panel, detail);
      this.setState("error", quiet ? "" : this.micProblemText);
    }

    clearMicProblem() {
      this.micProblem = null;
      this.setState("idle", this.panel ? PANEL_SAY.idle : "");
    }

    /** Whisper Base stopped being usable since the page was made (409 at start): the
     * microphone becomes the crossed-out one, as if the page had been made now, and the
     * area shows the error (state "error"; whisperProblem keeps it from recording). */
    showWhisperProblem(message) {
      this.whisperProblem = true;
      const mic = this.toggle;
      const help = document.getElementById("mic-help");
      if (mic) {
        mic.removeAttribute("data-dictation-action");
        mic.removeAttribute("aria-disabled");
        mic.classList.remove("mic-toggle", "mic-error");
        mic.classList.add("mic-unavailable");
        mic.setAttribute("data-dictation-unavailable", "");
        mic.title = message + " Click to see what to do.";
        setIcon(mic, "mic-off");
        setText(mic.querySelector("[data-mic-label]"), "Voice input unavailable");
        if (help) mic.setAttribute("popovertarget", "mic-help");
      }
      if (help) {
        setText(help.querySelector("[data-mic-reason]"), message);
        try {
          if (mic && !help.matches(":popover-open")) help.showPopover();
        } catch (error) {
          // No pop-up support: the status line says it.
        }
      }
      const problem = this.part("[data-dictation-problem]");
      if (problem) {
        setText(problem.querySelector("[data-dictation-problem-text]"), message);
        problem.hidden = false;
      }
      const start = this.part('[data-dictation-action="start"]');
      if (start) start.setAttribute("data-dictation-unavailable", "");
      this.setState("error", message);
    }

    // -- The live panel's tools ------------------------------------------------------------------

    copy() {
      const box = this.output;
      if (!box || !box.value) return;
      const fallback = () => this.say(copyBySelecting(box) ? SAY.copied : SAY.notCopied);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(box.value).then(() => this.say(SAY.copied), fallback);
      } else {
        fallback();
      }
    }

    /** Empties the box. A finished recording's "Completed" goes with its text. */
    clear() {
      const box = this.output;
      if (!box || LOCKED.has(this.state)) return;
      box.value = "";
      box.dispatchEvent(new Event("input", { bubbles: true }));
      this.selection = null;
      if (this.state === "done") {
        this.elapsed = 0;
        this.setState("idle", SAY.cleared);
      } else {
        this.say(SAY.cleared);
      }
      box.focus();
    }

    download() {
      const box = this.output;
      if (!box || !box.value) return;
      const name = transcriptName(new Date());
      const text = box.value.endsWith("\n") ? box.value : box.value + "\n";
      const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.hidden = true;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      this.say(`Downloading ${name}.`);
    }

    language() {
      const select = this.part("[data-dictation-language]");
      return (select && select.value) || "auto";
    }

    focusOutput() {
      const box = this.output;
      if (!box) return;
      box.focus();
      const length = box.value.length;
      const [start, end] = this.selection || [length, length];
      setSelection(box, Math.min(start, length), Math.min(end, length));
    }

    /** Points the keyboard at the way to stop: the panel's Stop, or the chat microphone. */
    focusStop() {
      const button = this.part('[data-dictation-action="stop"]') || this.toggle;
      if (button && button.getClientRects().length > 0) button.focus();
    }

    /** Leaving the page: the session ends with the page, and the microphone is released. */
    release() {
      if (this.session) discardSession(this.session.id, true);
      this.session = null;
      this.halt();
      if (this.active()) this.setState(this.heard ? "done" : "idle", "");
    }

    // -- Showing it ------------------------------------------------------------------------------

    setState(state, message) {
      this.state = state;
      this.notice = "";
      this.refresh();
      this.say(message === undefined ? this.defaultMessage() : message);
      syncLeaveGuard();
    }

    /** Why a message wasn't sent. It goes once it no longer applies: when the text changes,
     * or when the page has loaded completely (an answer being written has ended). */
    notify(message) {
      this.notice = message;
      this.say(message);
    }

    clearNotice() {
      if (!this.notice || this.message !== this.notice) return;
      this.notice = "";
      this.say("");
    }

    defaultMessage() {
      return (this.panel ? PANEL_SAY[this.state] : SAY[this.state]) || "";
    }

    setBusy(busy) {
      if (this.busy === busy) return;
      this.busy = busy;
      this.refresh();
    }

    /** The status message. Written a moment later, so a burst of changes is announced once;
     * partial text is never announced. */
    say(message) {
      this.message = message || "";
      if (this.messageTimer) return;
      this.messageTimer = setTimeout(() => {
        this.messageTimer = 0;
        this.writeMessage();
      }, MESSAGE_DELAY_MS);
    }

    /** The panel's state name is aria-hidden (it flips between Recording and Transcribing
     * all the time), so its message starts with the state for screen readers. */
    writeMessage() {
      const element = this.part("[data-dictation-message]");
      if (!element) return;
      const prefix = this.panel && this.message ? this.stateLabel() + ". " : "";
      if (element.textContent !== prefix + this.message) {
        element.textContent = this.message;
        if (prefix) {
          const hidden = document.createElement("span");
          hidden.className = "sr-only";
          hidden.textContent = prefix;
          element.prepend(hidden);
        }
      }
      this.refreshQuiet();
    }

    /** Everything the state decides, set again after each change. */
    refresh() {
      const state = this.state;
      const active = this.active();
      this.root.setAttribute("data-state", state);
      if (state === "error" && this.micProblem) this.root.setAttribute("data-mic-problem", this.micProblem);
      else this.root.removeAttribute("data-mic-problem");
      const busy = this.busy && state === "recording";
      this.root.toggleAttribute("data-busy", busy);
      this.refreshToggle(active);
      this.refreshPause();
      this.refreshPanel(active);
      const box = this.output;
      if (box) box.readOnly = LOCKED.has(state);
      const select = this.part("[data-dictation-language]");
      if (select) select.disabled = active;
      setText(this.part("[data-dictation-label]"), this.label());
      const busyNote = this.part("[data-dictation-busy]");
      if (busyNote) busyNote.hidden = !busy;
      this.renderClock();
      this.refreshQuiet();
      this.keepFocus();
    }

    /** A control that the new state hides or turns off mustn't take the keyboard focus with
     * it (to the top of the page): the control that fits the state gets it instead (Start →
     * Stop; Stop or an error → Start; else the text box). Runs right after the change,
     * while the browser still reports the old control as focused. */
    keepFocus() {
      const focused = document.activeElement;
      if (!focused || !this.root.contains(focused)) return;
      if (!focused.disabled && focused.getClientRects().length > 0) return;
      const next = this.active() ? ["stop", "toggle", "pause", "resume"] : ["toggle", "start"];
      for (const action of next) {
        const button = this.part(`[data-dictation-action="${action}"]`);
        if (button && !button.disabled && button.getClientRects().length > 0) {
          button.focus();
          return;
        }
      }
      if (this.output) this.output.focus();
    }

    /** The chat microphone: Start or Stop, or a warning mark for a microphone problem. Its
     * name says what a click does (no aria-pressed: a toggle's name must not change). */
    refreshToggle(active) {
      const mic = this.toggle;
      if (!mic) return;
      const problem = !active && this.state === "error" && !!this.micProblem;
      const name = active ? "Stop voice input" : "Start voice input";
      setAriaDisabled(mic, this.state === "finishing" && !this.pausingNow);
      mic.classList.toggle("mic-error", problem);
      setIcon(mic, problem ? "mic-alert" : "mic");
      setText(mic.querySelector("[data-mic-label]"), name);
      mic.title = problem ? this.micProblemText : `${name} (${MIC_KEY})`;
    }

    /** One button that pauses, and resumes once paused. */
    refreshPause() {
      const button = this.pauseButton;
      if (!button) return;
      const paused = this.state === "paused";
      button.setAttribute("data-dictation-action", paused ? "resume" : "pause");
      setAriaDisabled(button, !(this.state === "loading" || this.state === "recording" || paused));
      setIcon(button, paused ? "play" : "pause");
      const label = paused ? "Resume" : "Pause";
      setText(button.querySelector("[data-pause-label]"), this.panel ? label : label + " voice input");
      button.title = label + " voice input";
    }

    refreshPanel(active) {
      const start = this.part('[data-dictation-action="start"]');
      if (start) start.disabled = this.whisperProblem || active;
      setAriaDisabled(
        this.part('[data-dictation-action="stop"]'),
        !active || (this.state === "finishing" && !this.pausingNow)
      );
      const box = this.output;
      const empty = !(box && box.value);
      const tools = { copy: empty, download: empty, clear: empty || LOCKED.has(this.state) };
      for (const action of Object.keys(tools)) {
        const button = this.part(`[data-dictation-action="${action}"]`);
        if (button) button.disabled = tools[action];
      }
    }

    /** The live panel's state as shown: one of LABELS, or Transcribing while words are
     * being worked out. */
    label() {
      return this.state === "recording" && this.busy ? "Transcribing" : this.stateLabel();
    }

    /** The same without the Recording/Transcribing flicker: what screen readers hear. */
    stateLabel() {
      return LABELS[this.state] || "Inactive";
    }

    startClock() {
      this.since = Date.now();
      clearInterval(this.clockTimer);
      this.clockTimer = setInterval(() => this.renderClock(), 500);
      this.renderClock();
    }

    stopClock() {
      if (this.since) this.elapsed += Date.now() - this.since;
      this.since = 0;
      clearInterval(this.clockTimer);
      this.clockTimer = 0;
      this.renderClock();
    }

    /** Recorded time as m:ss (pauses don't count). The chat shows it once recording has
     * begun, until it ends; the panel always. */
    renderClock() {
      const started = this.state !== "requesting" || this.elapsed > 0;
      const show = this.panel || (this.active() && started);
      const ms = this.elapsed + (this.since ? Date.now() - this.since : 0);
      setText(this.part("[data-dictation-timer]"), show ? clockText(ms) : "");
    }

    /** The chat's status line takes no room when it has nothing to say. */
    refreshQuiet() {
      const status = this.part("[data-dictation-status]");
      if (!status || this.panel) return;
      const message = this.part("[data-dictation-message]");
      const timer = this.part("[data-dictation-timer]");
      const empty = !(message && message.textContent) && !(timer && timer.textContent);
      status.classList.toggle("is-empty", empty);
    }
  }

  // ---- Enter-to-send ------------------------------------------------------------------------------

  /** Forms on their way to the server (no second send), and forms waiting for voice input
   * to finish before sending. Both are forgotten when the page comes back from the
   * back/forward cache. */
  let sending = new WeakSet();
  let finishing = new WeakSet();
  /** The one submission the script itself starts lets the submit handler through. */
  const letThrough = new WeakSet();

  function messageBox(form) {
    return form.querySelector('textarea[name="text"]') || form.querySelector("textarea");
  }

  function sendButton(form) {
    const send = form.querySelector(".btn-send");
    if (send) return send;
    for (const element of form.elements) {
      if (element.type === "submit" && !element.hasAttribute("formaction")) return element;
    }
    return null;
  }

  function hasFiles(form) {
    for (const input of form.querySelectorAll('input[type="file"]')) {
      if (input.files && input.files.length > 0) return true;
    }
    return false;
  }

  /** The composer shows Stop instead of Send while an answer is written; the streamed page
   * shows Send again when it ends (the class stays). */
  function isGenerating(form) {
    const send = form.querySelector(".btn-send");
    return !!send && form.classList.contains("is-generating") && send.getClientRects().length === 0;
  }

  /** What happens to a send now: "send", "dictation" (voice input finishes first), or why not. */
  function decide(form) {
    if (sending.has(form) || finishing.has(form)) return "duplicate";
    if (form.getAttribute("data-can-send") === "false") return "model";
    if (isGenerating(form)) return "generating";
    if (form.hasAttribute("data-dictation") && recorderFor(form).active()) return "dictation";
    const box = messageBox(form);
    if (!(box && box.value.trim()) && !hasFiles(form)) return "empty";
    return "send";
  }

  function tell(form, problem) {
    if (form.hasAttribute("data-dictation") && SEND_PROBLEMS[problem]) recorderFor(form).notify(SEND_PROBLEMS[problem]);
  }

  function submitNow(form, submitter) {
    const button = submitter || sendButton(form);
    letThrough.add(form);
    try {
      if (typeof form.requestSubmit === "function") form.requestSubmit(button || undefined);
      else if (button) button.click();
      else form.submit();
    } finally {
      letThrough.delete(form);
    }
  }

  function markSending(form) {
    sending.add(form);
    const set = sending;
    // A navigation stopped by hand (Esc) would otherwise keep the form from sending again.
    setTimeout(() => set.delete(form), 60000);
  }

  /** Enter (or Send) while recording: stop, wait for the final text, then send it. */
  function sendAfterDictation(form, submitter) {
    if (finishing.has(form)) return;
    finishing.add(form);
    const set = finishing;
    recorderFor(form)
      .stop(true)
      .then((ok) => {
        set.delete(form);
        if (!ok) return; // the problem is shown and the text kept
        const decision = decide(form);
        if (decision === "send") submitNow(form, submitter);
        else if (decision === "model" || decision === "generating") tell(form, decision);
        // "empty": nothing was heard, and the recording's message says so.
      })
      .catch((error) => {
        set.delete(form);
        console.error(error);
      });
  }

  /** Keys that don't put the caret anywhere (leaving the box with Tab, for one). */
  const NOT_TOUCHING = new Set(["Tab", "Shift", "Control", "Alt", "Meta", "Escape"]);

  function onKeyDown(event) {
    const box = event.target;
    if (!(box instanceof HTMLTextAreaElement)) return;
    if (!NOT_TOUCHING.has(event.key)) box.lasTouched = true;
    if (event.key !== "Enter" || event.defaultPrevented) return;
    const form = box.form;
    if (!form || !form.hasAttribute("data-enter-send")) return;
    if (event.isComposing || event.keyCode === 229) return; // confirming a character in an input method
    if (event.altKey) return;
    if (event.shiftKey && !event.ctrlKey && !event.metaKey) return; // Shift+Enter: a new line
    event.preventDefault();
    if (event.repeat) return;
    try {
      const decision = decide(form);
      if (decision === "send") submitNow(form, null);
      else if (decision === "dictation") sendAfterDictation(form, null);
      else if (decision === "model" || decision === "generating") tell(form, decision);
      // "empty" and "duplicate": Enter does nothing.
    } catch (error) {
      console.error(error);
      // A bug above mustn't take Enter away: send what there is, as the Send button would.
      if (box.value.trim() || hasFiles(form)) submitNow(form, null);
    }
  }

  /** A form that would replace the page while a recording is open (a Whisper Base button, the
   * theme switch, Regenerate…): held back, with a word on why. The page, and the recording's
   * text with it, would be gone; the composer's own Send finishes the recording first instead.
   * Returns true when it was held back. */
  function keepRecording(event, form) {
    const recorder = recording();
    if (!recorder || event.defaultPrevented) return false;
    const submitter = event.submitter;
    const target = (submitter && submitter.getAttribute("formtarget")) || form.getAttribute("target") || "";
    if (target && !/^_(self|top|parent)$/i.test(target)) return false; // a hidden frame, a new tab
    const method = (submitter && submitter.getAttribute("formmethod")) || form.getAttribute("method") || "";
    if (method.toLowerCase() === "dialog") return false;
    event.preventDefault();
    const popup = form.closest("[popover]");
    if (popup) {
      try {
        popup.hidePopover();
      } catch (error) {
        // Not open, or no pop-up support.
      }
    }
    recorder.notify(recorder.panel ? SAY.leavePanel : SAY.leaveChat);
    recorder.focusStop();
    return true;
  }

  /** Send clicks and Alt+S follow the same rules as Enter. */
  function onSubmit(event) {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    if (!(form.hasAttribute("data-enter-send") && form.hasAttribute("data-dictation")) && keepRecording(event, form)) {
      return;
    }
    if (!form.hasAttribute("data-enter-send")) return;
    let decision = "send";
    try {
      if (!letThrough.has(form)) decision = decide(form);
    } catch (error) {
      console.error(error); // a bug here must not keep the message from being sent
    }
    if (decision === "send") {
      markSending(form);
      return;
    }
    event.preventDefault();
    if (decision === "dictation") sendAfterDictation(form, event.submitter || null);
    else if (decision !== "duplicate") tell(form, decision);
  }

  // ---- Wiring --------------------------------------------------------------------------------------

  const recorders = new Map();

  /** The area recording right now (a recording is open), if any. */
  function recording() {
    for (const recorder of recorders.values()) {
      if (recorder.active()) return recorder;
    }
    return null;
  }

  let leaveGuard = false;

  function onBeforeUnload(event) {
    if (!recording()) return;
    event.preventDefault();
    event.returnValue = ""; // older browsers ask only when this is set
  }

  /** Leaving the page ends a recording and loses its text, so the browser asks first; but
   * only while one is open (a page that might ask isn't kept for Back and Forward). */
  function syncLeaveGuard() {
    const need = !!recording();
    if (need === leaveGuard) return;
    leaveGuard = need;
    if (need) window.addEventListener("beforeunload", onBeforeUnload);
    else window.removeEventListener("beforeunload", onBeforeUnload);
  }

  /** What got the focus in this very task: an access key (Alt+M) focuses and clicks at once. */
  let justFocused = null;

  function onFocusIn(event) {
    const target = event.target;
    justFocused = target;
    setTimeout(() => {
      if (justFocused === target) justFocused = null;
    }, 0);
  }

  /** How a button was pressed: "mouse" (or a pen), "touch", "shortcut" (its access key), or
   * "keyboard" (Enter or Space while it had the focus). */
  function pressedHow(event, button) {
    if (event.detail > 0) {
      if (event.pointerType) return event.pointerType === "touch" ? "touch" : "mouse";
      const fine = window.matchMedia && window.matchMedia("(hover: hover)").matches;
      return fine ? "mouse" : "touch";
    }
    return justFocused === button ? "shortcut" : "keyboard";
  }

  /** Chat buttons after which the text box gets the focus back (see onClick). */
  const BACK_TO_TEXT = new Set(["toggle", "pause", "resume"]);

  function recorderFor(root) {
    let recorder = recorders.get(root);
    if (!recorder) {
      recorder = new Dictation(root);
      recorders.set(root, recorder);
      recorder.prepare();
    }
    return recorder;
  }

  /** Whether the parser has finished `element`: something after it already exists. */
  function isParsed(element) {
    if (document.readyState !== "loading") return true;
    for (let node = element; node && node !== document.body; node = node.parentElement) {
      if (node.nextElementSibling) return true;
    }
    return false;
  }

  function setUp() {
    for (const root of document.querySelectorAll("[data-dictation]")) {
      if (recorders.has(root) || !isParsed(root)) continue;
      try {
        recorderFor(root);
      } catch (error) {
        console.error(error); // one area's problem mustn't stop the others (or Enter-to-send)
      }
    }
  }

  function onClick(event) {
    const button = event.target instanceof Element ? event.target.closest("[data-dictation-action]") : null;
    const root = button && button.closest("[data-dictation]");
    if (!root) return;
    event.preventDefault(); // also keeps the "needs JavaScript" pop-up shut
    if (button.disabled || button.getAttribute("aria-disabled") === "true") return;
    const recorder = recorderFor(root);
    const action = button.getAttribute("data-dictation-action");
    try {
      recorder.act(action);
    } catch (error) {
      recorder.fail(error);
    }
    // A click leaves the focus on the microphone, where Enter would press it again (stopping
    // instead of sending). In the chat the text box gets it back, so Enter sends (finishing
    // the recording first). Not after a tap (the on-screen keyboard would pop up), and not
    // for Enter or Space on a focused button: keyboard users keep their place.
    const how = pressedHow(event, button);
    if (!recorder.panel && BACK_TO_TEXT.has(action) && (how === "mouse" || how === "shortcut")) {
      recorder.focusOutput();
    }
  }

  function onInput(event) {
    const box = event.target;
    if (!(box instanceof HTMLTextAreaElement)) return;
    if (event.isTrusted) box.lasTouched = true;
    const root = box.hasAttribute("data-dictation-output") && box.closest("[data-dictation]");
    const recorder = root && recorders.get(root);
    if (!recorder) return;
    recorder.refreshPanel(recorder.active());
    if (event.isTrusted) recorder.clearNotice(); // "Type a message first" is moot now
  }

  function onPointerDown(event) {
    if (event.target instanceof HTMLTextAreaElement) event.target.lasTouched = true;
  }

  document.addEventListener("click", onClick);
  document.addEventListener("focusin", onFocusIn);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("submit", onSubmit);
  document.addEventListener("input", onInput);
  document.addEventListener("pointerdown", onPointerDown, true);
  // Not on visibilitychange: switching tabs must not end a recording.
  window.addEventListener("pagehide", () => {
    for (const recorder of recorders.values()) recorder.release();
  });
  window.addEventListener("pageshow", () => {
    sending = new WeakSet();
    finishing = new WeakSet();
  });

  setUp();
  if (document.readyState === "loading") {
    // A streaming chat page reaches DOMContentLoaded only when the answer ends: pick up the
    // composer as soon as the parser has passed it.
    const observer = typeof MutationObserver === "function" ? new MutationObserver(setUp) : null;
    if (observer) observer.observe(document.documentElement, { childList: true, subtree: true });
    document.addEventListener("DOMContentLoaded", () => {
      if (observer) observer.disconnect();
      setUp();
      // The answer being written has ended: "An answer is still being written" is out of date.
      for (const recorder of recorders.values()) recorder.clearNotice();
    });
  }
}
