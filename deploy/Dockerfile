# assay batch image - NVFP4 quantize + lm-eval on Blackwell (sm_120).
#
# TWO-STAGE. The `builder` stage installs the toolchain exactly as the old single-stage
# image did, then the `runtime` stage rebuilds a clean tree from it. Two things motivate
# the split, in this order of importance:
#
#   1. BLOB GRANULARITY (the primary goal). The 2026-07-26 run pulled for ~50 min at an
#      observed ~80 Mbps and died on `unexpected EOF`. At 80 Mbps a 7.13 GB pull is ~12
#      min, so 50 min of not-finishing is RETRY AMPLIFICATION: the torch layer was a single
#      4.81 GB compressed blob, and a drop anywhere in it re-fetches the WHOLE blob. This
#      file therefore stages the tree into ~9 disjoint pieces, one COPY each, so a drop
#      re-fetches only that piece. A smaller image with one huge blob would have failed the
#      same way - splitting is the fix, shrinking is the bonus.
#   2. Dead-byte removal (secondary). `pip uninstall` in a later layer removes packages from
#      the filesystem while every byte stays shipped underneath a whiteout - which is why the
#      prune step's own layer reads as ~18 kB and looks free. Whiteouts do not survive a
#      stage boundary, so the runtime tree carries only live content. Measured on 0.6.0 that
#      is ~3 GB of ~15.1 GB uncompressed.
#
#      MEASUREMENT WARNING, learned the hard way: `du -sh` inside the image reports COMPRESSED
#      BLOCKS when the container store is on lz4 ZFS (this box), which understated the live
#      tree as 8.0 GB when it is really 12.0 GB. Always `du --apparent-size`, and for pull
#      cost read the COMPRESSED manifest total (7.13 GB on 0.6.0), not `podman images`.
#
# Latest-stable coherent set, pinned exactly and bumped as a unit
# (this is a 0.x ecosystem - vLLM/llmcompressor/compressed-tensors pin each other
# with == because "patch" releases can carry format/API changes; never float them).
# EVERY member, torch and vLLM included, is asserted exactly by the build gate
# reading deploy/constraints.txt (F-030 closed the torch/vLLM gap - they install
# outside the constraints resolve, marked `# image-only` in that file):
#   vLLM 0.25.0+cu129  (installed from vLLM's GitHub release asset, NOT PyPI - PyPI's
#                 default 0.25.0 wheel is CUDA-13; the +cu129 asset links libcudart.so.12,
#                 torch 2.11, Blackwell/sm_120, and stays off the cu130/r580 driver floor)
#   torch 2.11.0+cu129
#   llmcompressor 0.12.0  (the writer; first-class NVFP4)
#   compressed-tensors 0.17.1  (the on-disk NVFP4 FORMAT CONTRACT - shared by writer
#                 and reader; see the deliberate-mismatch note at the build gate)
# This is a BATCH image: it KEEPS the build tools + full toolchain (llm-compressor,
# lm-eval, vllm) that a slim serving image would strip.
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
# RunPod injects the host libcuda. BOTH stages use this base, so the runtime tree
# inherits the same property. The job also asserts a usable GPU at startup
# (assay.job.assert_gpu_available) so a dead GPU fails loud instead of CPU-grinding.

# =============================================================================
# STAGE 1: builder
# =============================================================================
# CACHE CONTRACT: every instruction from here down to the vLLM install is kept
# BYTE-IDENTICAL to the old single-stage image on purpose. Those two pip layers are
# ~13.8 GB of work; keeping them cache-valid is what makes this rewrite a minutes-long
# rebuild instead of an hours-long one. Do not "tidy" them - reordering, reflowing, or
# merging any of them silently throws away both layers. The NVIDIA_* ENV below is inert
# during a build (nothing here reads it) and is carried only to preserve that cache;
# the runtime stage sets it again for real.
FROM registry.access.redhat.com/ubi9/ubi AS builder

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
# RELEASE ASSETS - NOT PyPI. vLLM 0.25.0's default PyPI wheel is a CUDA-13 build (its
# _C ext links libcudart.so.13, absent on a cu129 image -> eval ImportError). vLLM
# publishes per-CUDA wheels only as GitHub release assets; the +cu129 asset links
# libcudart.so.12 and matches torch 2.11.0+cu129 - keeping us off the CUDA-13/r580
# host-driver floor. (An `--extra-index-url .../cu129` on the plain `vllm==` spec does
# NOT work: that torch index has no vllm, so pip falls through to PyPI's cu13 wheel -
# the exact bug that shipped in 0.1.2. The build gate below now asserts the +cu129 tag.)
#
# These stay SEPARATE RUN layers for the cache contract above. Their old
# pull-resilience rationale is now served by the blob staging further down, which is
# strictly better: it splits the CONTENT, not just the install steps.
RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
RUN python3.12 -m pip install \
         "https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
         --extra-index-url https://download.pytorch.org/whl/cu129

# ---- end of the cache-critical prefix; safe to edit below this line ----

WORKDIR /app
COPY . /app
# NON-EDITABLE install, deliberately (changed from `pip install -e .` in the multi-stage
# rewrite). An editable install writes a .pth pointing at an absolute /app/src, which
# would couple the copied site-packages to a /app that must exist at the identical path
# forever - a standing tripwire for a production image, and one no build-time check
# catches directly. A normal install puts the package in site-packages where it travels
# with the rest of the staged tree, and assay_task_dir() resolves against the installed
# location. /app still exists in the runtime stage, but only to carry pod_entry.sh.
# -c constraints.txt: the pinned coherent set (llmcompressor / compressed-tensors /
# transformers / lm-eval / scipy, plus torch/vllm as image-only pins, F-030) resolved
# in this ONE pass - per-pin rationale lives
# in that file (F-006/F-011). The old shape force-pinned compressed-tensors and
# transformers with LATER `pip install pkg==X` lines - the F-007 trap: a future
# llmcompressor needing a different compressed-tensors would be silently down-forced
# into an incoherent pair while the assert still passed. Under constraints, that
# future is a LOUD resolve failure at build time. (The known, deliberate vLLM
# ct==0.17.0 mismatch still surfaces only as a resolver warning - vLLM is
# pre-installed above, outside this resolve - and the build gate asserts 0.17.1 won.)
RUN python3.12 -m pip install . -c deploy/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu129

# lm-eval[math] extra. minerva_math500 (MATH-500, in the R1 reasoning battery) does a
# HARD import of math_verify + sympy + antlr4-python3-runtime==4.11 at task-module load
# time and RAISES if any is missing - bare `lm-eval` does NOT install them. Without this,
# loading the task crashes AFTER a paid quantize+eval on the rented GPU. The [math]
# extra needs its own line; the version is constrained by the same file as above
# (tests/test_build_pins.py asserts the == here matches the constraint).
RUN python3.12 -m pip install "lm-eval[math]==0.4.12" -c deploy/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu129

# Prune vLLM-optional deps our FA2-pinned / FlashInfer-sampler-off, TEXT-ONLY eval path
# never uses. Two rationales:
#  (a) never-loaded backends (verified unreachable in vLLM 0.24/0.25 source): the
#      FlashInfer sampler short-circuits on VLLM_USE_FLASHINFER_SAMPLER=0 BEFORE import;
#      cutlass-dsl/tilelang/tokenspeed are mamba/DeepSeek-MoE backends, not Qwen dense;
#      opencv/av are the multimodal try/except path; z3 is an off-path constraint solver.
#  (b) torchcodec - CRITICAL, not just cleanup (verified on metal). vLLM eagerly imports it
#      via vllm.multimodal.video during `from vllm import LLM`, guarded by
#      `try: from torchcodec.decoders import VideoDecoder; except ImportError: <placeholder>`.
#      torchcodec dlopens FFmpeg's shared libs at import, and bare ubi9 has NO FFmpeg, so
#      the PRESENT-but-broken package raises RuntimeError (NOT ImportError) -> the guard
#      does NOT catch it -> the whole eval import crashes. Making it ABSENT turns that into
#      ModuleNotFoundError (an ImportError subclass) which the guard DOES catch.
#
# THE PRUNE USED TO LIE. Every uninstall ended in `|| true`, so a name matching nothing
# exited silently - and three packages the list believed it removed were still shipping in
# 0.6.0: nvidia-cutlass-dsl-libs-base (list said nvidia-cutlass-dsl), tokenspeed-mla (list
# said tokenspeed-triton), and pynvvideocodec (never listed at all - a VIDEO CODEC in a
# text-only eval image, the exact sibling of the torchcodec problem this prune exists for).
#
# So the fix is NOT a longer list - the list drifts on every dep bump. It is the FAMILY
# SCAN below: after uninstalling, walk every installed distribution and fail the build if
# any name matches a banned family token. That catches a renamed or newly-introduced
# sibling (`-libs-base`, `-mla`, a future `flashinfer-jit`) without anyone updating a list.
# `|| true` stays on the individual uninstalls - a name that matches nothing is benign
# list drift; the family scan is what decides whether the prune actually worked.
# The verifier lives in scripts/verify_prune.py (unit-tested in tests/test_verify_prune.py)
# rather than inline here: an inline multi-line Python block cannot survive Dockerfile line
# continuation, and more importantly a guard nobody can run or test locally is a guard
# nobody trusts.
RUN set -eu; \
    PRUNE="torchcodec flashinfer-python flashinfer-cubin nvidia-cutlass-dsl \
           nvidia-cutlass-dsl-libs-base tokenspeed-triton tokenspeed-mla tilelang \
           z3-solver opencv-python-headless pynvvideocodec"; \
    for pkg in $PRUNE; do echo "prune: $pkg"; python3.12 -m pip uninstall -y "$pkg" || true; done; \
    python3.12 /app/scripts/verify_prune.py

# -----------------------------------------------------------------------------
# Blob staging - the point of this whole file
# -----------------------------------------------------------------------------
# Each /stage/NN becomes exactly ONE `COPY --from=builder` below, which becomes one layer,
# which becomes one independently-retried blob on pull.
#
# WHY MOVES AND NOT CLEVER COPIES: Dockerfile COPY has no exclude, and a glob flattens
# directory contents rather than preserving the tree. "Everything except torch" is simply
# not expressible. Physically partitioning the tree here is the only way to get balanced
# blobs while keeping paths intact.
#
# WHY COMPUTED AND NOT A HAND-WRITTEN LIST. The obvious version transcribes a measurement
# taken on one particular day:
#     stage 01 nvidia/cudnn ; stage 02 nvidia/cublas ; ...
# That works exactly once. Every torch or vLLM bump renames, resizes or splits those
# directories, and then someone has to re-measure the tree by hand before the image will
# build - with no guidance on how. Dependency bumps are the most common maintenance event
# this file will ever see, so the split is DERIVED from the tree instead: plan_blobs.py
# walks it, descends into anything oversized, and greedily bin-packs into BUCKETS bins.
# Bump a dependency and the packing re-balances itself.
#
# COMPLETENESS IS STRUCTURAL. The packer partitions ALL of /usr/local into disjoint units
# that provably cover it (asserted in tests/test_plan_blobs.py), so nothing can be omitted
# by a stale list - the failure mode that would otherwise be silent. dnf writes only under
# /usr, never /usr/local, so treating /usr/local as wholly pip-owned is exhaustive and
# stays exhaustive. That matters: /usr/local/bin holds 89 console scripts including a real
# `ninja` binary that torch.utils.cpp_extension probes on PATH at runtime, and NOTHING in
# the build gate or smoke executes a console script - so omitting it would fail silently,
# on metal. The runtime stage checks for ninja explicitly as a backstop.
#
# BUCKETS must equal the number of COPY --from=builder instructions in the runtime stage.
# A mismatch is caught by the reassembly check right after the COPYs, not by silence.
#
# --max-bucket-mib is the ENFORCED acceptance criterion, and it is the reason this file
# exists: no blob may be big enough that a dropped pull throws away a ruinous transfer.
# 2048 sits comfortably under the 4.81 GB blob that failed on 2026-07-26 while leaving
# the packer room. The first build after this rewrite produced a 5.0 GiB bucket because
# the unit-splitting depth guard fired too early - it looked completely green. Without
# this check that would have shipped, and the run would have failed exactly as before.
RUN python3.12 /app/scripts/plan_blobs.py --root /usr/local --buckets 9 --stage /stage \
      --max-bucket-mib 2048

# =============================================================================
# STAGE 2: runtime - the image that actually ships
# =============================================================================
FROM registry.access.redhat.com/ubi9/ubi

# GPU runtime contract. Bare ubi9 sets none of the vars the nvidia/cuda base provided,
# so restore the functional ones the NVIDIA container runtime consumes:
#  - NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES: request all GPUs WITH the
#    compute+utility caps (utility-only = no libcuda -> engine dies).
#  - LD_LIBRARY_PATH: where the host injects the driver libs (libcuda) in legacy mode.
#  - NVIDIA_REQUIRE_CUDA: fail-fast on a too-old driver. cuda>=12.9, NOT 12.8:
#    we first tried 12.8 reasoning the cu129 stack runs on r570 via CUDA
#    minor-version compat - TRUE for torch's cubin kernels, but vLLM's FA2 sm_120 kernel
#    ships PTX the r570 driver's JIT is too old to load, dying at eval-engine start with
#    cudaErrorUnsupportedPtxVersion. r575+/CUDA-12.9 is the real floor. This ENV is the
#    secondary guard; runpod_ctl.ALLOWED_CUDA_VERSIONS=[12.9,13.0] is the primary
#    (scheduler-level) filter - see its RCA comment.
# THIS BLOCK IS METAL-ONLY IN ITS FAILURE MODE: nothing in the build gate, the smoke
# step, or the test suite exercises it. Dropping it produces a green build and a dead pod.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.9" \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Same dnf line as the builder, and it MUST stay. gcc/gcc-c++/python3.12-devel are a
# RUNTIME requirement, not a build convenience: vLLM and Triton JIT on the pod and need a
# host CC and Python.h. This is the obvious thing to "slim" away and it is the wrong
# thing to slim - a sibling vLLM image already paid for that lesson (missing Python.h for
# Triton JIT). python3.12-pip is kept so the build gate's `pip check` works and so a 2 AM
# operator can introspect the environment on a live pod.
RUN dnf install -y python3.12 python3.12-pip python3.12-devel gcc gcc-c++ git \
    && dnf clean all && rm -rf /var/cache/dnf

# /app carries pod_entry.sh (the ENTRYPOINT) plus the build-time helper scripts, and
# nothing load-bearing for imports - assay itself is installed into site-packages by the
# non-editable install in the builder. Copied from the build context rather than from the
# builder: it is ~230 kB, and sourcing it here keeps this high-churn content in its own
# tiny blob instead of dirtying a multi-GB one.
#
# It lands BEFORE the staged COPYs because the reassembly check below runs from it.
WORKDIR /app
COPY . /app

# The staged tree, ONE COPY PER BUCKET. This count and the --buckets argument in the
# builder are a two-sided contract: a bucket with no COPY is silently absent from the
# image, and no other gate here would notice (the build gate imports a handful of modules;
# a missing bucket might take out none of them). The manifest check below is what makes
# that loud. If you add or remove a COPY, change --buckets to match and let the check
# confirm it - do not reason about it.
COPY --from=builder /stage/01/ /
COPY --from=builder /stage/02/ /
COPY --from=builder /stage/03/ /
COPY --from=builder /stage/04/ /
COPY --from=builder /stage/05/ /
COPY --from=builder /stage/06/ /
COPY --from=builder /stage/07/ /
COPY --from=builder /stage/08/ /
COPY --from=builder /stage/09/ /
COPY --from=builder /stage/manifest.txt /tmp/blob-manifest.txt

# Reassembly check: the runtime tree must hold exactly the bytes the builder staged.
# Runs before every other gate so a truncated tree fails on its real cause rather than as
# a confusing ImportError three steps later.
RUN python3.12 /app/scripts/plan_blobs.py --root /usr/local --verify /tmp/blob-manifest.txt

ENV HF_HOME=/runpod-volume/hf-cache
# Force FA2 + disable the FlashInfer sampler for the eval engine. The cu129 wheel pulls
# flashinfer-python as a dep, and on Blackwell vLLM auto-selects FlashInfer, whose JIT
# paths need nvcc - absent in this runtime base. FA2 is precompiled in the wheel and
# validated on 5090/sm_120 (a proven serving config); without this the import fix just
# trades the ImportError for a FlashInfer JIT failure at EngineCore start. flashinfer is
# PRUNED above, so losing these two vars trades one metal failure for another.
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0

# Build gate, run in the RUNTIME stage on purpose: it is what proves the staged COPYs
# reconstructed a complete, working tree. We CANNOT GPU-test locally, so this asserts the
# load-bearing invariants and surfaces (without falsely failing on) the one KNOWN,
# deliberate advisory mismatch.
#
# CRITICAL (verified on metal): import `from vllm import LLM, SamplingParams, TokensPrompt`
# - the EXACT import lm-eval's vllm backend does - NOT a bare `import vllm`. Bare
# `import vllm` is lazy and does NOT pull vllm.multimodal.video, so it passed green while
# eval crashed on a rented GPU when that fuller chain hit a present-but-broken torchcodec.
# This import needs no GPU (module load only), so it does not falsely fail on the GPU-less
# build host.
RUN python3.12 -c "\
import torch, llmcompressor, lm_eval, compressed_tensors as ct; \
from vllm import LLM, SamplingParams, TokensPrompt; \
import vllm; \
from math_verify import parse, verify; \
import antlr4; \
from scipy.stats import sem; \
from importlib.metadata import version as _dv; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| vllm', _dv('vllm'), \
'| llmcompressor', llmcompressor.__version__, '| compressed-tensors', ct.__version__, \
'| antlr4-runtime', _dv('antlr4-python3-runtime')); \
assert torch.version.cuda and torch.version.cuda.startswith('12.9'), 'torch drifted off cu129: '+str(torch.version.cuda); \
assert '+cu129' in _dv('vllm'), 'vllm is not a cu129 build (PyPI default is CUDA 13; install the +cu129 GitHub wheel): '+_dv('vllm'); \
assert _dv('antlr4-python3-runtime').startswith('4.11'), 'antlr4-python3-runtime must be 4.11.x - minerva_math (lm-eval[math]) asserts this at task import: '+_dv('antlr4-python3-runtime'); \
pins = [l.split('#')[0].strip() for l in open('/app/deploy/constraints.txt')]; \
pins = dict(p.split('==') for p in pins if p); \
assert pins, 'constraints.txt parsed empty - the pin gate would be a no-op'; \
bad = {n: (_dv(n), w) for n, w in pins.items() if _dv(n) != w}; \
assert not bad, 'drifted off the pinned coherent set (installed vs constraints.txt): '+repr(bad); \
print('pin gate OK:', ' '.join(f'{n}=={w}' for n, w in sorted(pins.items())))"

# Console-script check. /usr/local/bin arrives via stage 99, and NOTHING else in this file
# executes a console script - so without this, losing it would be invisible until Triton
# tried to JIT on a paid pod and could not find `ninja` on PATH. Cheap, direct, catches the
# one COPY-completeness failure the import gate structurally cannot see.
RUN set -eu; \
    command -v ninja >/dev/null || { echo "FATAL: ninja missing from PATH - /usr/local/bin did not survive the stage COPY (torch.utils.cpp_extension needs it to JIT)" >&2; exit 1; }; \
    ninja --version; \
    python3.12 -c "import shutil,sys; sys.exit(0 if shutil.which('lm_eval') else 'FATAL: lm_eval console script missing - /usr/local/bin incomplete')"

# `pip check` WILL flag several EXPECTED mismatches, and this is deliberately NOT asserted
# to be empty - a correct build produces new lines here:
#   - vLLM 0.25's advisory compressed-tensors==0.17.0 vs our deliberate 0.17.1
#   - a harmless fastapi range (assay uses lm-eval's OFFLINE vLLM engine, not vLLM's server)
#   - `vllm requires pynvvideocodec / tokenspeed-mla, which is not installed` - both are
#     declared deps of the vLLM we KEEP, so pruning them necessarily shows up here. That is
#     the visible price of dropping a kept package's optional deps, not a defect.
RUN python3.12 -m pip check || echo "NOTE: pip-check mismatches above are EXPECTED (deliberate ct 0.17.1 over vLLM's advisory 0.17.0; fastapi range irrelevant to the offline eval path; pruned vLLM optionals reported as missing deps)."

# Tier-1 structural smoke in the PROD IMAGE: resolves every recipe's tasks, binds the
# external filter_fn, runs process_results on a synthetic sample, and crosses a REAL
# spawn boundary. Would have caught the v0.4.1 avg@K crash at build time, ~2s, no GPU.
# Also the end-to-end proof that the non-editable install resolves package data (the
# lm_eval_tasks/*.yaml files) from site-packages rather than from /app/src.
RUN python3.12 -m assay.smoke

# Baked build identity: the git short SHA of the built tree, surfaced by the in-pod
# identity print so a stale/wrong image is obvious in minute one. Pass at build:
#   podman build --build-arg ASSAY_BUILD_SHA="$(git rev-parse --short HEAD)" ...
# DELIBERATELY placed LAST: this ARG changes on every commit (it IS the git SHA), and an
# ARG/ENV invalidates the build cache from its own instruction downward. In the old
# single-stage file, placing it high busted the ~13.8 GB install layers on every release.
# Here the penalty would be smaller (re-running cached COPYs reproduces identical digests,
# since COPY preserves mtimes) but it is still pure waste. It is read only at RUNTIME
# (assay.job.run_identity_lines), never at build, so the low placement is behavior-identical.
ARG ASSAY_BUILD_SHA=unknown
ENV ASSAY_BUILD_SHA=${ASSAY_BUILD_SHA}

RUN chmod +x /app/scripts/pod_entry.sh
ENTRYPOINT ["/app/scripts/pod_entry.sh"]
