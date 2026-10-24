# LLaDA forward-pass bring-up

The initial target is single-request BF16 inference with full recomputation.
The ModuleV3 model is in `max/python/max/pipelines/architectures/llada`.
It is not registered with `max serve` or the autoregressive text pipeline.

## Frozen reference

- HF weights, tokenizer, configuration, and modeling code:
  `GSAI-ML/LLaDA-8B-Instruct@6059b30c9531767ce47e0ca6a8161a6bb0c67693`.
- Generation algorithm:
  `ML-GSAI/LLaDA@9182493720ed723ef8031210d85959364e51cbe0`.

In a Python environment with PyTorch, Transformers, Accelerate and safetensors,
run on a CUDA GPU:

```sh
git clone https://github.com/ML-GSAI/LLaDA.git /tmp/LLaDA
python max/examples/llada/reference.py --reference-repo /tmp/LLaDA \
  --output /tmp/llada-reference.json --dump-first-logits /tmp/llada-first.pt
```

The runner reads `generate.py` directly from the pinned Git blob, regardless
of the reference checkout's working tree. It uses `AutoModel`, the pinned
instruction template, one unpadded request, 64 output tokens, 64 steps,
zero temperature, no guidance, and confidence-based selection. It records
the canvas before every forward call and the final canvas. Adjacent snapshots
identify the committed positions. The optional first-logits fixture supports
forward parity checks independently of generation.

No reference output or full-checkpoint parity result has been established yet.
The pinned config records Transformers 4.46.3; compatibility with the installed
Transformers version must be checked when running the custom model code.

The small CPU parity suite can be run with:

```sh
./bazelw test //max/tests/integration/architectures/llada:test_forward
```

On Xcode 27, the bundled LLVM linker rejects the SDK's `arm64e.x1` TAPI
targets. The local retry uses Apple's linker and disables LLVM-style library
grouping without changing repository toolchain files:

```sh
./bazelw test --linkopt=--ld-path=/usr/bin/ld \
  --host_linkopt=--ld-path=/usr/bin/ld \
  --features=-supports_start_end_lib --host_features=-supports_start_end_lib \
  //max/tests/integration/architectures/llada:test_forward
```

It compares a two-layer model against independent PyTorch SDPA and RMSNorm,
and checks that future tokens affect earlier logits. This suite has not yet
executed: the local Xcode setup is fixed, and the first Bazel build of native
dependencies is in progress. Formatting, lint, Python syntax, and reference
CLI help were checked.

## Architecture differences from the MAX Llama ModuleV3 path

| Component | Required LLaDA behavior | Initial implementation |
| --- | --- | --- |
| Attention | 32 Q/K/V heads, bidirectional, no KV cache | Explicit float32 scores/softmax; no causal mask |
| RoPE | Contiguous half-head rotation in float32 | External float32 cos/sin inputs, cast rotated Q/K back |
| MLP | `ff_out(silu(ff_proj(x)) * up_proj(x))` | Separate checkpoint-named projections |
| Norm/residual | Pre-RMSNorm before attention and MLP | Existing RMSNorm, epsilon from config |
| Output | Untied head; logits at every position | `[batch, sequence, vocabulary]` float32 logits |
| Weights | `model.transformer.*` | Module names match after stripping that prefix |

The forward model takes tokens `[B, S]` and RoPE cos/sin `[1, 1, S, D]`.
Construct frequencies using `1 / theta ** (arange(0, D, 2) / D)`, multiply
by positions, duplicate the half-head frequencies, then take cos/sin.
Reject canvases longer than the configured maximum before execution.

Explicit attention is a correctness baseline with quadratic workspace;
short-sequence feasibility does not establish 4096-token feasibility on 24 GB.
The checkpoint weights occupy about 14.93 GiB. Run HF and MAX sequentially.

## Remaining work

1. Run the pinned reference and save the trace and initial logits.
2. Validate the small forward model against PyTorch, then add checkpoint loading
   with complete key/shape coverage and compare full-checkpoint logits.
3. Implement the token update loop and compare committed positions per step.
4. Add the local MAX inference entry point and final text decoding.
5. Replace explicit attention with a supported noncausal fused operation after
   establishing numerical parity.
