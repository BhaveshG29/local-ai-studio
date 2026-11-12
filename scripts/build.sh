#!/usr/bin/env bash
# Builds Local AI Studio for the fastest hardware this computer has.
#
#   Apple Silicon Mac                  graphics card, through Metal (always included)
#   Linux, NVIDIA card + CUDA toolkit  graphics card, through CUDA
#   anything else                      processor (CPU)
#
# Usage: scripts/build.sh [--cpu] [more cargo build options]
#   --cpu   leave CUDA out even when an NVIDIA card is found
#
# On Windows, use scripts/build.ps1 instead.
set -euo pipefail

cd "$(dirname "$0")/.."

use_gpu=1
extra=()
for arg in "$@"; do
  case "$arg" in
    --cpu) use_gpu=0 ;;
    -h | --help)
      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) extra+=("$arg") ;;
  esac
done

if ! command -v cargo > /dev/null 2>&1; then
  echo "Rust isn't installed. Install it from https://rustup.rs, open a new terminal and run this again." >&2
  exit 1
fi

features=()
target="the processor (CPU)"
advice=""

case "$(uname -s)" in
  Darwin)
    if [ "$(uname -m)" = "arm64" ]; then
      target="the Apple GPU (Metal)"
      if [ "$use_gpu" = 0 ]; then
        advice="Metal is always part of the Apple Silicon build. To keep the GPU idle, choose
\"CPU only\" in Settings > Inference instead."
      fi
    fi
    ;;
  *)
    if [ "$use_gpu" = 1 ] && command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
      # The CUDA toolkit often isn't on PATH after installation.
      if ! command -v nvcc > /dev/null 2>&1; then
        for dir in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /opt/cuda; do
          if [ -n "$dir" ] && [ -x "$dir/bin/nvcc" ]; then
            export PATH="$dir/bin:$PATH"
            break
          fi
        done
      fi
      if command -v nvcc > /dev/null 2>&1; then
        features=(--features cuda)
        target="the NVIDIA GPU (CUDA)"
      else
        advice="This computer has an NVIDIA graphics card, but the CUDA toolkit isn't installed,
so the built-in engine will use the processor. To use the card, install the toolkit
from https://developer.nvidia.com/cuda-downloads and run this script again.
(Ollama, if you use it, runs on the graphics card either way.)"
      fi
    fi
    ;;
esac

echo "Building Local AI Studio. The built-in engine will run on $target."
if [ ${#features[@]} -gt 0 ]; then
  echo "CUDA kernels are compiled for this computer's graphics card; the first build takes a while."
fi
echo

cargo build --release ${features[@]+"${features[@]}"} ${extra[@]+"${extra[@]}"}

echo
echo "Done. Start it with:"
echo
echo "  ./target/release/local-ai-studio"
if [ -n "$advice" ]; then
  echo
  echo "$advice"
fi
