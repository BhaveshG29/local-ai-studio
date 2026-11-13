# Local AI Studio

**Author:** Bhavesh G and Claude (Opus 5.5)

[![Latest release](https://img.shields.io/github/v/release/BhaveshG29/local-ai-studio)](https://github.com/BhaveshG29/local-ai-studio/releases/latest)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust 1.94+](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://rustup.rs)

Find, download and run AI models on your own computer, privately and without the cloud.

## 1. Overview

Local AI Studio is a single program that turns your computer into a private AI workstation.
You can browse the model catalogs of Hugging Face and Ollama, download models, see whether
they will run on your hardware, and then use them: chat with language models, ask vision
models about pictures and videos, transcribe recordings with Whisper, and create images with
Stable Diffusion.

Everything happens on your machine. The program starts a small web server on `127.0.0.1` and
opens its interface in your browser; nothing you type, upload or generate is sent anywhere.
Models run in a built-in engine written in Rust on top of Hugging Face's
[Candle](https://github.com/huggingface/candle), or in [Ollama](https://ollama.com) if you
have it installed, which adds thousands more models.

The built-in engine uses your graphics card when it can. Apple Silicon Macs use it through
Metal out of the box, and NVIDIA cards through CUDA when you build the program with the
included build script (see [GPU acceleration](#5-gpu-acceleration)). The interface is plain
HTML and CSS without any JavaScript, so it works in every browser and stays fast.

## 2. Download and run

There are two ways to get Local AI Studio. Downloading a release is the quickest. Building
from source takes a few minutes more, and it is how you get NVIDIA GPU support on Linux and
Windows.

### Download a release

Open the [Releases page](https://github.com/BhaveshG29/local-ai-studio/releases) and download
the file for your computer:

| Your computer | File |
|---|---|
| Windows 10 or 11, 64-bit | `local-ai-studio-<version>-x86_64-pc-windows-msvc.zip` |
| Mac with Apple Silicon (M1 or newer) | `local-ai-studio-<version>-aarch64-apple-darwin.tar.gz` |
| Mac with an Intel processor | `local-ai-studio-<version>-x86_64-apple-darwin.tar.gz` |
| Linux, 64-bit (Ubuntu 22.04 or newer, or similar) | `local-ai-studio-<version>-x86_64-unknown-linux-gnu.tar.gz` |

Unpack it and start the program inside. On macOS and Linux:

```bash
tar xzf local-ai-studio-*.tar.gz
cd local-ai-studio-*/
./local-ai-studio
```

On Windows, unzip the file and double-click `local-ai-studio.exe`.

The release builds are not code-signed yet. If Windows SmartScreen stops the program, choose
**More info → Run anyway**. On a Mac, remove the download quarantine once before the first
start:

```bash
xattr -d com.apple.quarantine local-ai-studio
```

The Apple Silicon release runs models on the GPU. The Linux and Windows releases run the
built-in engine on the processor; to use an NVIDIA card, build from source as described next,
or install Ollama, which uses the card by itself.

### Build from source

You need [Rust](https://rustup.rs) 1.94 or newer and a C compiler:

| System | Install first |
|---|---|
| Ubuntu, Debian | `sudo apt install build-essential git` |
| Fedora | `sudo dnf groupinstall "Development Tools"` |
| macOS | `xcode-select --install` |
| Windows | [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++", and Git |

Then install Rust with the command from [rustup.rs](https://rustup.rs) (on Windows, keep the
default MSVC toolchain). If your computer has an NVIDIA graphics card, also install NVIDIA's
[CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

Get the code and build it with the build script, which checks your hardware and turns on GPU
support where it can:

```bash
git clone https://github.com/BhaveshG29/local-ai-studio
cd local-ai-studio
./scripts/build.sh
./target/release/local-ai-studio
```

On Windows, in PowerShell:

```powershell
git clone https://github.com/BhaveshG29/local-ai-studio
cd local-ai-studio
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
.\target\release\local-ai-studio.exe
```

| Your computer | The built-in engine runs on |
|---|---|
| Mac with Apple Silicon | the GPU, through Metal |
| Linux or Windows with an NVIDIA card and the CUDA toolkit | the GPU, through CUDA |
| Anything else | the processor |

The first build takes several minutes (longer with CUDA, whose kernels are compiled for your
card); later builds are quick. If you prefer plain Cargo, `cargo build --release` gives the
same result except that you add `--features cuda` yourself for an NVIDIA card.

### First steps

When the program starts, your browser opens at <http://127.0.0.1:8765>. If it doesn't, open
that address yourself.

1. Go to **Discover** and search for a model. A good first one is
   `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, about 1 GB. Each model page tells you whether it fits
   your computer.
2. Press **Download** and follow the progress on **Downloads**.
3. Open **Chat**, pick the model and write a message. The first answer takes a moment while
   the model loads.

Other good first downloads are `openai/whisper-base` for speech to text,
`stabilityai/sd-turbo` for images and, with Ollama, `gemma3`, which can also describe
pictures.

To stop the program, press **Ctrl+C** in its terminal or use **Settings → About → Stop Local
AI Studio**.

### Command-line options

`local-ai-studio --help` lists them all. The ones you are most likely to need:

| Option | What it does |
|---|---|
| `--port 9000` | Use another port than 8765 |
| `--no-browser` | Don't open the browser on start |
| `--data-dir PATH` | Keep settings, database, models and logs in another folder (also `LOCAL_AI_STUDIO_HOME`) |
| `--host 0.0.0.0 --allow-network` | Let other devices on your network connect; they need the access token from Settings → Server |
| `--no-keychain` | Keep the Hugging Face token in memory only |
| `--system-info` | Print the detected hardware as JSON and exit |

## 3. Features

### Finding and downloading models

- Search Hugging Face by task, format, model family and author, sorted by trending,
  downloads, likes or date. Browse the Ollama library too, including which models can see
  images, use tools or reason step by step.
- Every model page shows its files, license and parameter count, and estimates how much
  memory each version needs, so you know before downloading whether it will run.
- Downloads can be paused and resumed (also after a restart), are retried when the connection
  drops, and every file is checked against the checksum Hugging Face publishes. Gated models
  work with a Hugging Face token, which is kept in your system's keychain.
- Everything you browse is saved in a local search index that also works offline.

### Chat

- Several conversations with history and renaming, export as JSON or Markdown, and import
  from JSON.
- Answers appear as they are written. You can stop them, regenerate, continue, or edit an
  earlier message and ask again.
- Markdown with tables and highlighted code, and maths: TeX formulas are drawn as real
  formulas.
- Each chat has its own system prompt and generation settings (temperature, top-p, top-k,
  answer length, context length, seed).

### Pictures, audio and video

- Attach pictures for vision models such as Gemma 3, Llama 3.2 Vision or Qwen2.5-VL (these
  run through Ollama).
- Audio is transcribed with a local Whisper model and handed to the chat model. The **Speech
  to text** page also translates into English and exports TXT and SRT subtitles.
- For videos, evenly spaced frames go to a vision model and the sound is transcribed (this
  needs ffmpeg).

### Creating images

Stable Diffusion 1.5, 2.1, SD-Turbo, SDXL and SDXL-Turbo run in the built-in engine. You set
the prompt, negative prompt, size, steps, guidance and seed, create up to four images at a
time, and can reuse the settings of any earlier image.

### Your library and your hardware

- One library holds Hugging Face downloads, Ollama models, and GGUF files you already have
  (they are used where they are, not copied).
- Models can be loaded, unloaded, renamed, checked again, updated and deleted.
- The **System** page shows your processor, memory, graphics cards and disks, which engine
  uses which hardware, and which model sizes fit.

### Also included

- Six themes that change the whole look, not just a colour: Midnight, Graphite, Neon
  Cyberpunk, Pure Minimal, Warm Studio and Modern Glass.
- An OpenAI-compatible API for other programs on your computer (see
  [Configuration](#6-configuration)).
- Optional access from other devices on your network, protected by an access token.
- Keyboard shortcuts, screen-reader labels, reduced-motion support, and a layout that works
  on a phone.

### Which models run where

The built-in engine runs GGUF chat models of the Llama, Mistral, Qwen 2/2.5/3,
Phi-3/3.5/4, Gemma 1/2/3, GLM-4 and LFM2 families (quantisations Q2_K to Q8_0, F16 and F32),
and safetensors chat models of the Llama, Mistral, Qwen 2/3 and Phi-3 families. Other GGUF
files, such as IQ quantisations or models split into several files, run through Ollama; model
pages say so and offer it. Text-to-speech and video generation are not available yet.

## 4. System requirements

| | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10 or 11 (64-bit), macOS 13 or newer, 64-bit Linux (the release build needs glibc 2.35 or newer, as in Ubuntu 22.04) | Current versions |
| Processor | 64-bit, 4 cores; AVX2 (Intel, AMD) or NEON (ARM) strongly recommended | 8 cores or more |
| Memory | 8 GB, for small models (0.5 to 3 billion parameters) | 16 GB for 7 to 8 billion, 32 GB or more for larger ones |
| Graphics card | Not required | Apple Silicon, or an NVIDIA card (see [GPU acceleration](#5-gpu-acceleration)) |
| Disk | 200 MB for the app | An SSD with 20 GB or more free for models (0.5 to 10 GB each) |
| Browser | Any current one: Firefox, Chrome, Edge, Safari | |

Optional software:

- [Ollama](https://ollama.com/download) for many more models, vision models, and automatic GPU
  use on NVIDIA and AMD cards.
- [ffmpeg](https://ffmpeg.org/download.html) for videos in chats and video transcription.

## 5. GPU acceleration

The built-in engine chooses its hardware with **Settings → Inference → Hardware**, which is
set to *Automatic*: it uses a graphics card when this copy of the program can, and the
processor otherwise.

- **Apple Silicon Macs:** Metal is always included, so models run on the GPU without any
  setup.
- **NVIDIA cards on Linux and Windows:** CUDA is included when you build the program with
  `scripts/build.sh` or `scripts/build.ps1` on a computer with the CUDA toolkit. Its kernels
  are compiled for the card in that computer, which is also why the downloadable Linux and
  Windows releases don't include CUDA.
- **Other graphics cards and Intel Macs:** the built-in engine uses the processor. Ollama can
  use many AMD cards.

If a model doesn't fit in the graphics card's memory, *Automatic* loads it on the processor
instead, and the System page shows where each model runs. Ollama manages its own GPU use. When
the built-in engine can't use your card but Ollama can, model pages offer the Ollama version
first (**Settings → Inference → When a model can run in both engines**).

On the processor, prompts are read with all threads while answers are written with the
fastest cores only, because the slower efficiency cores hold the others back. On an Intel
Core Ultra 9 185H this triples the writing speed of Qwen2.5-0.5B (18.8 instead of 6.3 tokens
per second). **Settings → Inference → Processor threads** sets one number for both if you want
to choose yourself. For the fastest CPU build on your own machine (not to share with others),
build with `RUSTFLAGS="-C target-cpu=native"`.

## 6. Configuration

Everything can be changed on the **Settings** page. The settings are stored in a
`config.toml` file in the data folder:

| System | Data folder |
|---|---|
| Linux | `~/.local/share/local-ai-studio` |
| macOS | `~/Library/Application Support/dev.LocalAIStudio.local-ai-studio` |
| Windows | `%APPDATA%\LocalAIStudio\local-ai-studio\data` |

The annotated [`config/local-ai-studio.example.toml`](config/local-ai-studio.example.toml)
lists every option with its default.

### Ollama (optional)

Install Ollama with `curl -fsSL https://ollama.com/install.sh | sh` on Linux, or with the
installer from <https://ollama.com/download> on macOS and Windows, and make sure it is
running (the Ollama app, or `ollama serve`). Local AI Studio finds it at
`http://127.0.0.1:11434`; another address can be set in **Settings → Ollama**.

### Hugging Face token (optional)

Most models download without an account. For gated models (Llama and Gemma, for example) or
your private repositories:

1. Open the model on huggingface.co and accept its license.
2. Create a token with *Read* access at <https://huggingface.co/settings/tokens>.
3. Paste it into **Settings → Hugging Face**.

The token is stored in your system's keychain (Keychain on macOS, Credential Manager on
Windows, Secret Service on Linux), never in a file. If you are logged in with the Hugging Face
command-line tool, **Use my Hugging Face CLI login** imports that token. The `HF_TOKEN`
environment variable works too.

### Using it from other programs

Local AI Studio has an OpenAI-compatible API, so tools that speak to OpenAI can use your local
models:

```bash
curl http://127.0.0.1:8765/v1/models
curl http://127.0.0.1:8765/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "ollama:gemma3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Model names are `local:<id>` for the built-in engine and `ollama:<name>` for Ollama;
`/v1/models` lists them. There is also a JSON API under `/api` (`/api/health`,
`/api/v1/system`, `/api/v1/models`, `/api/v1/downloads`, `/api/v1/chat`).

## 7. Project structure

```text
local-ai-studio/
├── src/
│   ├── main.rs             Command line, start-up and shutdown
│   ├── app.rs              Shared state and background jobs
│   ├── config.rs           Data folders and settings
│   ├── db/                 SQLite storage: chats, library, downloads, offline index
│   ├── hub.rs              Hugging Face client and "Can I run it?" analysis
│   ├── ollama.rs           Ollama client and library reader
│   ├── downloads.rs        Download manager
│   ├── library.rs          Model files: inspection, import, delete
│   ├── inference/          Engines: device choice, LLMs, Whisper, Stable Diffusion
│   ├── render.rs, math.rs  Markdown and TeX to safe HTML
│   ├── system.rs           Hardware detection and memory estimates
│   └── web/                HTTP server, security checks, pages and API
├── templates/              HTML templates, one per page
├── assets/                 CSS, themes, icons and fonts (built into the program)
├── tests/                  Integration tests with fake Hugging Face and Ollama servers
├── scripts/                build.sh and build.ps1, which pick GPU support automatically
├── config/                 Example configuration
├── docs/                   The completion report
└── tools/                  Generator for THIRD_PARTY_LICENSES.md
```

## 8. How it works

Local AI Studio is one Rust program that contains the web server, the inference engines, the
database and all assets. Requests are handled by [Axum](https://github.com/tokio-rs/axum) on
Tokio, and pages are rendered on the server with [Askama](https://github.com/askama-rs/askama)
templates, so the browser receives finished HTML and CSS.

**An interface without JavaScript.** Actions are ordinary forms followed by a redirect.
Menus are `<details>` elements, dialogs use the HTML `popover` attribute, and progress pages
refresh themselves. Chat answers are streamed: the server keeps the page open and writes the
answer into it as it is generated. The Content-Security-Policy forbids scripts entirely.

**Model analysis.** A Hugging Face repository is analysed from its file list and metadata
(GGUF headers, `config.json`, diffusers layouts) to decide what it can do, which engine can run
it and how much memory each version needs. After a download the files are inspected again, so
a finished download isn't simply assumed to work.

**Engines.** Both engines implement one small trait. The built-in engine runs GGUF models with
Candle's quantized kernels and safetensors models in full precision. It builds prompts with
each model's own chat template, drops the oldest messages when a conversation outgrows the
context window, and reuses its cache when a conversation continues. Models load on first use
and are unloaded after a configurable idle time. Ollama requests go to its `/api/chat`.

**Storage.** One SQLite database holds conversations, the library, download jobs (so they
survive restarts), the offline index, and image and transcription history. Settings are a
TOML file; uploads and generated images are plain files in the data folder.

**Security.** The server listens on `127.0.0.1` only, unless you explicitly allow your
network, and then every other device needs an access token. Host and Origin checks block DNS
rebinding and cross-site requests. Model output is rendered with raw HTML escaped and only
`http`, `https` and `mailto` links allowed. File paths from requests are never allowed to leave
their folders. The Hugging Face token stays in the keychain and is only sent to Hugging Face.

## 9. Testing

```bash
cargo test                                   # unit and integration tests, no internet or models needed
cargo fmt --all -- --check                   # formatting
cargo clippy --all-targets -- -D warnings    # lints
```

| Tests | What they cover |
|---|---|
| Unit tests in `src/` | Parsing Hub and Ollama data, compatibility rules, tokenizers and chat templates, memory estimates, settings, Markdown safety, maths |
| `tests/downloads.rs` | Real downloads from a fake Hugging Face server: checksums, resuming, corrupted files, gated repositories, disk space, pause and cancel, restarts |
| `tests/chat.rs` | Chats through the real routes with a fake Ollama: streaming, stop, regenerate, edit, export and import, images, the OpenAI-compatible API |
| `tests/pages.rs` | Every page renders without scripts, and every link, form and button leads somewhere that exists |
| `tests/security.rs` | Host and Origin checks, the network token, path traversal, HTML injection, size limits |
| `tests/models.rs`, `tests/discovery.rs`, `tests/server.rs` | Crafted GGUF files, search and the offline index, port conflicts and shutdown |

Tests that need real models, a GPU, Ollama or internet access are skipped unless you ask for
them:

```bash
LAS_TEST_GGUF=/path/to/model.gguf  cargo test --release --test real_models -- --ignored builtin
LAS_TEST_OLLAMA_MODEL=qwen3:0.6b   cargo test --release --test real_models -- --ignored ollama
LAS_TEST_NETWORK=1                 cargo test --release --test real_models -- --ignored network
```

`tests/real_models.rs` lists the others (Whisper and image generation). The results of the
last full run are in [`docs/COMPLETION_REPORT.md`](docs/COMPLETION_REPORT.md).

GitHub Actions runs only when you publish a version. Pushing a tag such as `v0.3.0` first
checks formatting and lints and runs the tests on Ubuntu, Windows, macOS on Apple Silicon (where
it also checks that Metal is built in) and macOS on Intel. Only if all of that passes are the four
downloads built and published on the Releases page. To run the checks without releasing, start
the CI workflow by hand from the repository's Actions tab.

| Platform | Automated tests | Built-in engine | GPU in the built-in engine | Ollama |
|---|---|---|---|---|
| Linux x86_64 | Ubuntu 24.04 laptop and CI | Tested | NVIDIA with a CUDA build | Tested, on the GPU |
| Windows x64 | CI | Expected to work | NVIDIA with a CUDA build | Expected to work |
| macOS, Apple Silicon | CI | Expected to work | Metal, built in | Expected to work |
| macOS, Intel | CI | Expected to work | Processor only | Expected to work |

The program has so far been used day to day only on Linux, and the GPU paths of the built-in
engine have not yet been run on real hardware by the authors. Reports from Windows and Mac
users are very welcome.

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| "port 8765 is already used" | Another program uses the port. Start with `--port 9000` or change it in Settings → Server. If Local AI Studio is already running, a second start just opens the first one. |
| The browser doesn't open | Open <http://127.0.0.1:8765> yourself; the address is also printed in the terminal. |
| "Blocked: this request used an unexpected host name" | Use `http://127.0.0.1:8765` or `http://localhost:8765`. Other host names are blocked to protect against DNS-rebinding attacks. |
| Ollama shows "Not running" | Start Ollama (the app, or `ollama serve`). If it runs on another address, set it in Settings → Ollama. |
| A model says "Can't run" | The model page gives the reason (unsupported architecture or quantisation, missing tokenizer, split file). Most such GGUF files run through Ollama: use **Get it with Ollama instead**. |
| A gated model fails with 403 | Accept the license on huggingface.co, then add a token in Settings → Hugging Face. |
| Downloads stop or fail | Each file is tried up to four times and resumes where it stopped. After that, press **Retry**. Check your free disk space and connection. |
| Answers are slow | Choose a smaller or more quantised model (Q4_K_M), close other programs, or use the GPU (see [GPU acceleration](#5-gpu-acceleration)). |
| My NVIDIA card isn't used | The Linux and Windows releases run the built-in engine on the processor. Install the CUDA toolkit and build with `scripts/build.sh`; the System page shows whether CUDA is available. Ollama uses the card either way. |
| The CUDA build fails with "unsupported GNU version" | Your CUDA toolkit is older than your system's compiler. Install a newer toolkit, or point it at an older compiler: `NVCC_CCBIN=g++-12 ./scripts/build.sh`. |
| "error while loading shared libraries: libcublas…" | The CUDA libraries aren't on the library path, for example `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`. |
| "needs about X but this computer has Y" | The model is too big for your memory. Choose a smaller size or quantisation. |
| The token can't be saved in the keychain (Linux) | No Secret Service is running, which is common on minimal systems. The token then lasts for the session only; set `HF_TOKEN` to keep it. |
| Videos or video transcription fail | Install ffmpeg. Settings → Media shows whether it was found, and lets you set its path. |
| Something else | Settings → Logging → **Open logs folder**. Raise the level to `debug`, try again and look at the newest log. Settings → Data & cache can clear caches or reset settings without touching models or chats. |

## 11. License

Local AI Studio is licensed under the [Apache License 2.0](LICENSE) (see also
[`NOTICE`](NOTICE)). It is a permissive license like MIT, and it also includes an explicit
patent grant, which matters for software in the AI space. It is the license of Candle,
Tokenizers and much of the Rust ecosystem this project builds on.

The Rust libraries compiled into the program keep their own licenses, listed in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) and in the app under Settings → About.
Almost all are permissive (MIT, Apache-2.0, BSD, ISC and similar). A few use MPL-2.0 (the
Symphonia audio decoders and the CSS-selector crates behind `scraper`), a file-level copyleft
that is satisfied because their unmodified source is available on crates.io. Nothing in the
dependency tree is GPL. The bundled fonts use the SIL Open Font License.

AI models are not part of this project. The models you download belong to their publishers,
and their licenses may restrict commercial use or require attribution. Checking and following
them is up to you; Local AI Studio shows each model's license where the publisher provides one.

## 12. Future improvements

- Vision models in the built-in engine, so image questions don't need Ollama.
- Text-to-speech for spoken answers.
- Video generation, once a practical model exists in Rust.
- Support for IQ quantisations and split GGUF files without Ollama.
- Questions over your own documents (local embeddings and search), and tool calling.
- HTTPS for network mode, and user accounts for shared computers.
- Installers (MSI, DMG, AppImage or Flatpak) with code signing, and update checks.
- Running part of a model on the GPU when it doesn't fit completely.
