# assay batch image -- NVFP4 quantize + lm-eval on Blackwell (sm_120).
# Latest-stable coherent set, pinned exactly and bumped as a unit
# (this is a 0.x ecosystem -- vLLM/llmcompressor/compressed-tensors pin each other
# with == because "patch" releases can carry format/API changes; never float them):
#   vLLM 0.25.0+cu129  (installed from vLLM's GitHub release asset, NOT PyPI -- PyPI's
#                 default 0.25.0 wheel is CUDA-13; the +cu129 asset links libcudart.so.12,
#                 torch 2.11, Blackwell/sm_120, and stays off the cu130/r580 driver floor)
#   torch 2.11.0+cu129
#   llmcompressor 0.12.0  (the writer; first-class NVFP4)
#   compressed-tensors 0.17.1  (the on-disk NVFP4 FORMAT CONTRACT -- shared by writer
#                 and reader; see the deliberate-mismatch note at the build gate)
# This is a BATCH image: it KEEPS the build tools + full toolchain (llm-compressor,
# lm-eval, vllm) that a slim serving image would strip -- as a SINGLE stage on bare UBI9.
#
# BASE = bare ubi9, deliberately NOT nvidia/cuda:12.9.x (verified on metal). The
# nvidia/cuda images bundle the `cuda-compat` package, and the NVIDIA container
# runtime engages it whenever the host driver is OLDER than the compat lib (RunPod's
# 5090 fleet skews driver 570.x < the base's compat 575.57.08). CUDA forward
# compatibility is supported ONLY on data-center GPUs, so on our consumer RTX 5090
# the compat libcuda raises CUDA error 804 (CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE)
# -> torch sees zero devices -> llm-compressor SILENTLY falls back to CPU and grinds
# (observed: 35 min of paid CPU calibration on a rented pod). Bare ubi9 carries no
# cuda-compat; torch's bundled site-packages/nvidia/* provides the CUDA runtime and
# RunPod injects the host libcuda. Mirrors a serving base validated on live 5090s
# while assay stays its own specialized image. The job also asserts a
# usable GPU at startup (assay.job.assert_gpu_available) so a dead GPU fails loud
# instead of silently CPU-grinding again.
FROM registry.access.redhat.com/ubi9/ubi

# GPU runtime contract. Bare ubi9 sets none of the vars the nvidia/cuda base provided,
# so restore the functional ones the NVIDIA container runtime consumes:
#  - NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES: request all GPUs WITH the
#    compute+utility caps (utility-only = no libcuda -> engine dies).
#  - LD_LIBRARY_PATH: where the host injects the driver libs (libcuda) in legacy mode.
#  - NVIDIA_REQUIRE_CUDA: fail-fast on a too-old driver. cuda>=12.9, NOT 12.8:
#    we first tried 12.8 reasoning the cu129 stack runs on r570 via CUDA
#    minor-version compat -- TRUE for torch's cubin kernels, but vLLM's FA2 sm_120 kernel
#    ships PTX the r570 driver's JIT is too old to load, dying at eval-engine start with
#    cudaErrorUnsupportedPtxVersion. r575+/CUDA-12.9 is the real floor. This ENV is the
#    secondary guard; runpod_ctl.ALLOWED_CUDA_VERSIONS=[12.9,13.0] is the primary
#    (scheduler-level) filter -- see its RCA comment.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.9" \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV PIP_NO_CACHE_DIR=1 PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=300
# gcc/gcc-c++/python3.12-devel: vLLM + Triton JIT need a host CC + Python.h at runtime
# (a documented vLLM/Triton runtime requirement); llm-compressor/lm-eval may JIT too.
RUN dnf install -y python3.12 python3.12-pip python3.12-devel gcc gcc-c++ git \
    && dnf clean all && rm -rf /var/cache/dnf

# torch cu129 FIRST (pinned, cu129 index), THEN vLLM's +cu129 wheel from its GitHub
# RELEASE ASSETS -- NOT PyPI. vLLM 0.25.0's default PyPI wheel is a CUDA-13 build (its
# _C ext links libcudart.so.13, absent on a cu129 image -> eval ImportError). vLLM
# publishes per-CUDA wheels only as GitHub release assets; the +cu129 asset links
# libcudart.so.12 and matches torch 2.11.0+cu129 -- keeping us off the CUDA-13/r580
# host-driver floor. (An `--extra-index-url .../cu129` on the plain `vllm==` spec does
# NOT work: that torch index has no vllm, so pip falls through to PyPI's cu13 wheel --
# the exact bug that shipped in 0.1.2. The build gate below now asserts the +cu129 tag.)
#
# SPLIT into separate RUN layers, deliberately (verified on metal): chaining torch + vLLM
# in one RUN produced a single ~6.6GB layer (94% of the image) that repeatedly failed
# to cold-pull to EUR-IS-1 -- 60-90 min then `unexpected EOF`. A dropped pull of one
# huge blob fails/restarts the WHOLE blob; smaller per-layer blobs are pulled and
# retried independently, so a drop only re-fetches that chunk. Install ORDER is
# preserved (torch cu129 first, then vLLM) -- the split changes layering, not behavior.
RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
RUN python3.12 -m pip install \
         "https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
         --extra-index-url https://download.pytorch.org/whl/cu129

WORKDIR /app
COPY . /app
# Install assay + quant/eval toolchain (llmcompressor 0.12 -> compressed-tensors 0.17.1,
# lm-eval, datasets, hf-hub, runpod) against the already-present cu129 torch/vLLM.
RUN python3.12 -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
# Deliberate, documented pin: run compressed-tensors 0.17.1 across BOTH write and read.
# llmcompressor 0.12 requires ==0.17.1; vLLM 0.25 pins ==0.17.0, but no released
# llmcompressor pins 0.17.0 (it skips 0.16.0->0.17.1), so 0.17.1 is the closest coherent
# pairing, and 0.17.1's diff over 0.17.0 is perf/validation, NOT the NVFP4 on-disk format.
# Making it explicit here so it is a decision, not an install-order accident.
RUN python3.12 -m pip install "compressed-tensors==0.17.1"
# transformers is part of the coherent set: llmcompressor 0.12 declares
# transformers<=5.10.1,>=5.9.0, so 5.10.1 is the intended ceiling (verified loading
# Qwen2ForCausalLM on metal). Pin it explicitly so a future dep bump can't silently
# drift transformers out from under the tested quantize path.
RUN python3.12 -m pip install "transformers==5.10.1"

# Remove vLLM-optional deps our FA2-pinned / FlashInfer-sampler-off, TEXT-ONLY eval
# path never uses. Two rationales:
#  (a) never-loaded backends (verified unreachable in vLLM 0.24/0.25 source): the
#      FlashInfer sampler short-circuits on VLLM_USE_FLASHINFER_SAMPLER=0 BEFORE import;
#      cutlass-dsl/tilelang/tokenspeed are mamba/DeepSeek-MoE backends, not Qwen dense;
#      opencv/av are the multimodal try/except path; z3 is an off-path constraint solver.
#  (b) torchcodec -- CRITICAL, not just cleanup (verified on metal). vLLM eagerly imports it
#      via vllm.multimodal.video during `from vllm import LLM`, guarded by
#      `try: from torchcodec.decoders import VideoDecoder; except ImportError: <placeholder>`.
#      torchcodec dlopens FFmpeg's shared libs at import, and bare ubi9 has NO FFmpeg, so
#      the PRESENT-but-broken package raises RuntimeError (NOT ImportError) -> the guard
#      does NOT catch it -> the whole eval import crashes (verified on metal). Making it
#      ABSENT turns that into ModuleNotFoundError (a subclass of ImportError) which the
#      guard DOES catch -> placeholder -> clean import. We do text-only eval, so dropping
#      video decode is a no-op; installing FFmpeg+EPEL to satisfy a feature we never use
#      would be the wrong fix. Verified locally: uninstall torchcodec -> `from vllm import
#      LLM, SamplingParams, TokensPrompt` succeeds.
# llm-compressor and lm-eval import none of these. Each `|| true` so a dep this build
# didn't pull can't fail the prune. NOTE: placed in its own RUN, so this removes the
# packages from the RUNTIME filesystem (which is all torchcodec's fix needs) but does NOT
# shrink the pull -- the bytes stay in the lower install layers under a whiteout; real
# size reduction is the deferred multi-stage slim. The build gate below (import vllm/
# llmcompressor/lm_eval) is the LOUD net if any pruned pkg were actually load-bearing.
RUN for pkg in torchcodec flashinfer-python flashinfer-cubin nvidia-cutlass-dsl tokenspeed-triton tilelang z3-solver opencv-python-headless ; do \
        echo "prune: $pkg" ; python3.12 -m pip uninstall -y "$pkg" || true ; \
    done

# Build gate. We CANNOT GPU-test locally, so this asserts the load-bearing invariants and
# surfaces (without falsely failing on) the one KNOWN, deliberate advisory mismatch.
# Also serves as the post-prune safety net: these imports must still resolve.
#
# CRITICAL (verified on metal): import `from vllm import LLM, SamplingParams, TokensPrompt`
# -- the EXACT import lm-eval's vllm backend does -- NOT a bare `import vllm`. Bare
# `import vllm` is lazy and does NOT pull vllm.multimodal.video, so it passed green while
# eval crashed on a rented GPU when that fuller chain hit a present-but-broken torchcodec.
# Exercising the real import here turns that class of failure into a BUILD failure (free,
# local) instead of a metal failure ($ + a 6min pull + 13min quantize to discover). This
# import needs no GPU (module load only), so it does not falsely fail on the GPU-less
# build host.
RUN python3.12 -c "\
import torch, llmcompressor, lm_eval, compressed_tensors as ct; \
from vllm import LLM, SamplingParams, TokensPrompt; \
import vllm; \
from importlib.metadata import version as _dv; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| vllm', _dv('vllm'), \
'| llmcompressor', llmcompressor.__version__, '| compressed-tensors', ct.__version__); \
assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'torch drifted off cu129: '+str(torch.version.cuda); \
assert '+cu129' in _dv('vllm'), 'vllm is not a cu129 build (PyPI default is CUDA 13; install the +cu129 GitHub wheel): '+_dv('vllm'); \
assert ct.__version__ == '0.17.1', 'compressed-tensors must be 0.17.1 (write/read NVFP4 format contract): '+ct.__version__"
# pip check WILL flag vLLM 0.25's advisory ct==0.17.0 (and a harmless fastapi range -- assay
# uses lm-eval's OFFLINE vLLM engine, not vLLM's fastapi server). That mismatch is EXPECTED
# and deliberate; NVFP4 load-correctness on ct 0.17.1 is validated on metal, not here.
RUN python3.12 -m pip check || echo "NOTE: pip-check mismatch above is EXPECTED (deliberate ct 0.17.1 over vLLM's advisory 0.17.0; fastapi range irrelevant to the offline eval path)."

ENV HF_HOME=/runpod-volume/hf-cache
# Force FA2 + disable the FlashInfer sampler for the eval engine. The cu129 wheel pulls
# flashinfer-python as a dep, and on Blackwell vLLM auto-selects FlashInfer, whose JIT
# paths need nvcc -- absent in this runtime base. FA2 is precompiled in the wheel and
# validated on 5090/sm_120 (a proven serving config); without this the import fix
# just trades the ImportError for a FlashInfer JIT failure at EngineCore start.
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
RUN chmod +x /app/scripts/pod_entry.sh
ENTRYPOINT ["/app/scripts/pod_entry.sh"]
