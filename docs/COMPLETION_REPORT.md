# Local AI Studio — completion report

*11 November 2026 · Bhavesh G and Claude*

Local AI Studio is one Rust program that finds AI models on Hugging Face and in the Ollama
library, downloads and verifies them, and runs them — with its built-in engine (Candle) or
through Ollama — behind a browser interface made only of server-rendered HTML and CSS. It has
chat with streaming answers, maths and code, image understanding through vision models, speech
to text, image generation, a model library, a hardware dashboard, six themes and a local
OpenAI-compatible API.

Everything below was observed, not assumed. The automated tests ran in a Linux container and
on the developer's Ubuntu laptop. The app was then used end to end on that laptop against the
real Hugging Face, ollama.com and Ollama, with real models. The problems that session found were
fixed, and each fix that can be tested has a regression test.

## 1. Status of the requirements

✅ verified on the laptop with the real services and models (in the live session or by the
opt-in real-model tests) · ☑ verified by automated tests with local fake servers · ⚠ partly
implemented or not verified · ❌ not implemented

| Area | Status | How it was verified |
|---|---|---|
| Hugging Face search: filters (task, format, family, author), sorting, pagination | ✅ | Live against huggingface.co and by the network test; tests |
| Model pages: “Can I run it?” verdict, versions with sizes and memory estimates, recommended version, model card | ✅ | Live (Qwen/Qwen2.5-0.5B-Instruct-GGUF); tests |
| Ollama library: search, versions with size, context and inputs, fit verdicts | ✅ | Live against ollama.com, after fixes 1–2 below, and by the network test; tests |
| Offline index, Hub outages, rate limits, no internet | ☑ | Tests |
| Downloads: progress, speed, time left, pause, resume from where it stopped, checksum verification | ✅ | Live: 469 MiB GGUF at ≈ 21 MB/s, paused at 75 %, resumed by HTTP range, verified, added to the library |
| Downloads: cancel, retry, corrupted or truncated files, disk-space check, duplicates, gated/private repositories with a token, recovery after a restart | ☑ | Tests |
| Ollama: detection and version, install instructions, pulls with progress | ✅ | Live: Ollama 0.32.14 detected, `qwen3:0.6b` pulled through the app |
| Built-in engine (Candle) on the CPU, GGUF chat models | ✅ | Live and by the real-model test: Qwen2.5-0.5B-Instruct Q4_K_M; a fixed seed repeats its answer; measurements in section 3 |
| Built-in engine on a GPU (Metal on Apple Silicon, CUDA via `scripts/build.sh`) | ⚠ | Metal is built into Apple Silicon builds and compiled in CI; CUDA builds are made by the build scripts when the CUDA toolkit is present. No CUDA toolkit or Mac was available here, so neither has run on real hardware yet. The device probe survives a Mac without a GPU, and a model that doesn't fit on the GPU falls back to the CPU |
| Ollama engine on the GPU | ✅ | Live: gemma4:e2b and qwen3:0.6b, “100 % on GPU” as reported by Ollama |
| Chat: streaming, stop, regenerate, continue, edit, reasoning sections, Markdown, code highlighting, maths, titles, settings per chat, export and import | ✅ / ☑ | Live: streaming, regenerate, reasoning, maths, titles, statistics; the rest by tests |
| OpenAI-compatible and JSON API | ✅ / ☑ | Real-model tests: the built-in engine and Ollama (`qwen3:0.6b`) answered through `/v1/chat/completions`; the JSON API and error cases by tests |
| Images sent to vision models | ☑ | Tests with a fake Ollama (the test tools could not drive the browser's native file picker); text-only models refuse images, also tested |
| Speech to text (Whisper) | ☑ | Unit tests; the live run was skipped to save disk space (the laptop had 1.8 GB free at the start) |
| Video understanding (frames + Whisper) | ☑ | Tests only |
| Image generation (Stable Diffusion 1.5/2.1/Turbo/XL) | ⚠ | Implemented with progress, cancel and history; an opt-in real-model test exists but was not run (several GB of downloads) |
| Video generation, text to speech | ❌ | No practical Rust-native model; the interface never offers them |
| Library: load, unload, rename, delete, inspect, import a `.gguf`, update | ✅ / ☑ | Live: listing, load state, sizes, sources; actions by tests |
| Hardware dashboard | ✅ | Live: CPU and features, RAM, swap, RTX 4060 (8 GB) and Intel iGPU, disks, models in memory, which engine can use which hardware |
| Six themes, remembered | ✅ | Live: switching saved to `config.toml`; tests |
| Settings | ✅ / ☑ | Live: cache clearing; validation by tests |
| Local server: `127.0.0.1` by default, network access only with explicit opt-in and an access token, port conflicts, graceful shutdown | ✅ / ☑ | Live: three Ctrl+C restarts, each logged as “stopped cleanly”; the rest by tests |
| Security: Host-header check, cross-site request blocking, CSP without scripts, path traversal, HTML escaping (including maths), token never written to disk, size limits | ☑ | Tests |

## 2. Problems found in the live session and on GitHub — all fixed

1. **Ollama library pages failed.** ollama.com now answers `/library/<name>/tags` with a bare JSON
   list of names whenever a client accepts JSON. The app asks for the HTML page, also understands
   the JSON, and the test suite's fake ollama.com behaves the same way, so the tests catch it.
2. **A cached unreadable page stayed in use** (“size unknown” for six hours). It is now fetched
   again. Regression test added.
3. **Ollama speed was understated.** The chat divided by the total time, including Ollama loading
   the model: 8.3 tokens/s for an answer Ollama generated at 99.1 tokens/s. The engine's own
   measurement is now stored (database migration v2) and shown. Test added.
4. **The built-in engine used all 22 threads,** which made generation three times slower on a
   processor with performance and efficiency cores. Prompts are now read with all threads and
   answers written with the performance cores (section 3).
5. **Maths was shown as TeX source.** TeX in answers is now converted to MathML on the server
   and drawn by the browser (no JavaScript); unknown commands stay visible as TeX.
6. **Maths written as `$ x $` (with spaces)** wasn't recognised; the first fix then misread a
   closing `$` followed by a space. Dollars are now paired in order. The answer that exposed
   it is a test case.
7. **Browsers kept an old stylesheet** after an update (the URL only changed with the version
   number). Stylesheet URLs now carry a hash of their content, and assets are revalidated.
8. **The chat header squeezed the token counts** into a narrow column; it is one line now.
9. Models without a task on the Hub showed the filter label **“Any task”**.
10. A relative **`--data-dir`** stayed relative; it is made absolute (test added).
11. **“(s)” plurals** (“1 download(s)”) replaced by proper singular and plural forms.

Also: a chat is named after its first message as soon as it is sent and never overwrites a name
given in the meantime; the tests ignore a Hugging Face token in the developer's environment; the
README's platform table only claims what was observed. The System page now also shows roughly
how much memory a loaded built-in model takes (the size of its weights); that change came after
the live session, so it has not been seen on a live page yet.

The first GitHub Actions run found two more:

12. **Windows downloads skipped the free-space check.** Windows gives folder paths as
    `\\?\C:\…`, which never matched the disk `C:\`, so free space was “unknown” (also on the
    Downloads, Settings and Home pages) and a download larger than the disk would have started.
    The prefix is now removed before matching. The download test caught it on Windows; a unit
    test now covers the conversion.
13. **The stated minimum Rust version, 1.88, was too old:** Candle 0.11.1 needs Rust 1.94 on
    x86-64. The minimum is now 1.94; the CI check reads it from `Cargo.toml`, and Clippy checks
    our own code against it.

The release workflow also stopped because a release for `v0.1.0` had already been made by hand.
It now attaches its files to an existing release instead of failing.

## 3. Measurements on the laptop

Ubuntu 24.04, Intel Core Ultra 9 185H (6 performance + 10 efficiency cores, 22 threads),
15.1 GB RAM, NVIDIA RTX 4060 Laptop GPU (8 GB, driver 595.99.02), Ollama 0.32.14.

| Engine, model | Setup | Result |
|---|---|---|
| Built-in, Qwen2.5-0.5B-Instruct Q4_K_M (CPU) | 6 threads (performance cores) | writing 18.8 tokens/s; reading a 1,182-token prompt ≈ 22 tokens/s |
| same | 22 threads | writing 6.2–6.4 tokens/s; reading the same prompt ≈ 41 tokens/s |
| same | final: read with 22, write with 6 | writing 18.7 tokens/s; reading the same prompt ≈ 41 tokens/s (28.6 s); writing after it 12.5 tokens/s (11.9 with 6 threads, 4.4 with 22) |
| Ollama, gemma4:e2b (GPU) | — | 99.1 tokens/s (735 tokens) |
| Ollama, qwen3:0.6b (GPU) | — | 38.4 tokens/s (344 tokens; one run, right after the model was pulled) |
| Download from Hugging Face | — | ≈ 21 MB/s |

The built-in engine is much slower than llama.cpp-based tools on the same CPU; Candle's CPU
kernels are the limit. Ollama is the fast path on this machine.

## 4. Automated test results

| Where | Toolchain | Result |
|---|---|---|
| Linux container (x86_64) | Rust 1.97 | **187 passed, 0 failed** with the fixes for items 12–13 (185 before them); 6 opt-in real-model tests not run (no models or internet there). `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`: clean |
| Ubuntu 24.04 laptop | Rust 1.99.1 | **185 passed, 0 failed** (before items 12–13); the same 6 opt-in tests skipped. Compiled in 37 s (dependencies already built), ran in ≈ 11 s |
| Ubuntu 24.04 laptop, opt-in real-model tests (release build) | Rust 1.99.1 | **3 passed, 0 failed** in 7.8 s. Built-in engine with the real Qwen2.5-0.5B-Instruct Q4_K_M: answered “Paris”, gave the same answer twice with a fixed seed, answered through the OpenAI-compatible API, unloaded. Ollama with `qwen3:0.6b`: answered “4” through the OpenAI-compatible API. Live huggingface.co and ollama.com: search, model pages, versions with sizes. Not run: download-then-chat (the same download was done by hand in the live session), Whisper and SD-Turbo (their downloads were skipped to save disk space) |
| Windows, macOS: GitHub Actions, first run | stable | Release builds compiled for Windows and both kinds of Mac. Windows tests: 149 passed, **1 failed** (item 12, fixed); the test files after it did not run, because Cargo stops at the first failing file. CI now uses `--no-fail-fast`, so the next run gives the first complete Windows result |

By suite (container, with the fixes): unit 118 · chat 11 · discovery 8 · downloads 15 · models 8 ·
pages 11 · security 9 · server 7.

## 5. Main decisions

- **No JavaScript at all**, as required. Actions are forms with redirects; menus use `<details>`;
  dialogs use the `popover` attribute; live answers are streamed HTML; progress pages refresh
  themselves; the Content-Security-Policy forbids scripts. Maths is converted to MathML on the
  server because KaTeX/MathJax would need JavaScript.
- **Two engines behind one trait.** Candle is pure Rust and runs GGUF quantisations on any CPU;
  Ollama brings the GPU and formats Candle can't run. Every model page says which engine can run
  it, and nothing is labelled as GPU-accelerated unless it is.
- **A finished download is not assumed to work**: files are verified by checksum and inspected
  (GGUF header, tokenizer, architecture) before a model is marked as runnable.
- **State in SQLite** (WAL, versioned migrations), settings in TOML, the Hugging Face token in the
  OS keychain (or memory only) — never in files or logs.
- **Apache-2.0** for the project (patent grant; the licence of Candle and most of the stack).
  Models keep their publishers' licences.

## 6. Supported platforms

Linux x86_64: built, tested and used (Ubuntu 24.04). Windows x64 and macOS (Apple Silicon and
Intel): the release builds compile on GitHub Actions and CI runs the tests there on every push;
on Windows the first run found one bug (item 12, fixed). The app has not yet been used
interactively on them. GPU acceleration in the built-in engine (Metal, built in on Apple
Silicon; CUDA from a build-script build) has not yet run on real hardware; Ollama provides GPU
use everywhere it supports the hardware.

## 7. Known limitations

- The built-in CPU engine is slow next to llama.cpp (18.8 tokens/s for a 0.5B model on this
  laptop); prefer Ollama for larger models or a GPU.
- IQ quantisations and split GGUF files run through Ollama, not the built-in engine.
- ollama.com has no public catalogue API; if its pages change, the app shows an error instead
  of wrong data until the reader is updated.
- The maths converter covers common notation; anything else is shown as TeX source.
- The OpenAI-compatible API returns a reasoning model's thinking inside `content`, in
  `<think>…</think>`, not in a separate field.
- Image understanding needs a vision model in Ollama; there is no text-to-speech or video
  generation.
- Release binaries are not code-signed (Windows SmartScreen / macOS Gatekeeper warnings).

## 8. Build and run

Needs Rust 1.94 or newer and a C compiler (see the README).

```bash
cargo build --release
./target/release/local-ai-studio            # opens http://127.0.0.1:8765
cargo test                                  # all automated tests (no internet or models needed)
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

Useful options: `--port 9000`, `--no-browser`, `--data-dir <folder>`, `--host` with
`--allow-network` (shares the app on the network, with an access token), `--system-info`.

## 9. Publishing on GitHub

From the project folder (the `.gitignore` already leaves out `target/`, `data/`, `vendor/`,
`vendor.tgz.part-*`, `env-report.txt` and model files):

```bash
git init -b main
git add .
git commit -m "Local AI Studio 0.1.0"
gh repo create local-ai-studio --public --source . --push
# or: create an empty repository on github.com, then
# git remote add origin https://github.com/<you>/local-ai-studio.git && git push -u origin main
```

The **CI** workflow then checks formatting and Clippy and builds and tests on Ubuntu, Windows,
macOS (Apple Silicon) and macOS (Intel). To publish downloadable programs, tag a version:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The **Release** workflow builds the four platforms and attaches archives with SHA-256 checksums to
a GitHub release.
