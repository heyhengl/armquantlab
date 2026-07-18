# ArmQuantLab

ArmQuantLab is a reproducible, privacy-scoped ONNX optimization evidence harness for Arm64.
It generates a deterministic FP32 workload, performs ONNX shape inference and graph
preprocessing, applies dynamic per-channel INT8 weight quantization, benchmarks the
preprocessed FP32 baseline and INT8 model with the same inputs, and emits machine-readable size,
latency, throughput, and numerical-parity evidence.

The included workload is synthetic and requires no model download, cloud account, API key,
customer data, or paid hardware. It runs natively on an Apple Silicon laptop through ONNX
Runtime's CPU execution provider.

## Why it exists

Optimization claims are often reduced to one favorable latency number without a model digest,
repeat protocol, output-parity check, or disclosure of slower cases. ArmQuantLab keeps the two
models, benchmark protocol, hashes, and every measured batch together so another developer can
inspect or repeat the result.

The same evidence workflow can also process a compatible ONNX model that the operator is
authorized to use. That path is local-only and does not add the supplied model or derivative
artifacts to this repository.

## Run the demo

```bash
uv run --python 3.12 --no-project --no-cache --with '.' \
  armquantlab demo --out dist/demo
```

Generated artifacts:

- `dist/demo/models/fixture-source.onnx`
- `dist/demo/models/fixture-fp32.onnx`
- `dist/demo/models/fixture-int8.onnx`
- `dist/demo/evidence/benchmark-report.json`
- `dist/demo/evidence/benchmark-summary.md`

The benchmark fixes ONNX Runtime to one CPU thread, uses the same deterministic inputs for both
models, performs warm-up runs, and records p50/p95 latency across batch sizes 1, 8, and 32. Each
latency sample averages 200 consecutive model invocations to reduce timer noise.

Symbolic shape inference is intentionally skipped because the fixture is a non-transformer graph
with complete ONNX shapes; graph optimization and ONNX shape inference remain enabled.

## Optimize a rights-cleared ONNX model locally

```bash
uv run --python 3.12 --no-project --no-cache --with '.' \
  armquantlab optimize \
  --model /path/to/authorized-model.onnx \
  --out dist/local-candidate \
  --ops MatMul,Gemm \
  --batch-sizes 1,8 \
  --acknowledge-model-rights
```

This path intentionally has a narrow contract:

- the model must be a regular, non-symlink, self-contained `.onnx` file under 1 GB;
- external tensor files are rejected;
- inputs must be float32 tensors with only the leading batch dimension dynamic;
- quantization ops must be an explicit subset of `Conv`, `Gemm`, and `MatMul`;
- generated inputs are deterministic synthetic values and are never persisted; and
- the report records hashes and shape contracts, but not the source path or filename.

The rights acknowledgement is mandatory. Local candidate outputs remain outside the checked-in
public evidence set; review model licensing, privacy, task accuracy, and publication rights
separately before sharing any derivative.

## Test and verify

```bash
uv run --python 3.12 --no-project --no-cache --with '.[dev]' pytest
uv run --python 3.12 --no-project --no-cache --with '.[dev]' ruff check .
uv run --python 3.12 --no-project --no-cache --with '.[dev]' python scripts/verify_release.py
uv run --python 3.12 --no-project --no-cache --with '.[dev]' python scripts/package_release.py
```

## Build the demo video

The macOS video pipeline renders a 1920x1080 benchmark-notebook storyboard from the checked-in
frozen report. It creates English narration and sentence-level SRT captions without rerecording
the benchmark, desktop, account, shell history, or a private model. The renderer rejects a
timeline of three minutes or longer.

See [`submission/VIDEO_BUILD.md`](submission/VIDEO_BUILD.md) for the reproducible commands and
[`submission/VIDEO_COMPOSITION.md`](submission/VIDEO_COMPOSITION.md) for the selected visual
direction and anti-AI-template gate.

The GitHub Pages player is a public preview, not a rule-compliant final video host. The official
rules allow an optional demo only when it is publicly visible on YouTube, Vimeo, or Youku. The
verified assets, paste-ready metadata, chapters, captions, disclosures, and financial stop
conditions are in
[`submission/YOUTUBE_PUBLICATION_PACKET.md`](submission/YOUTUBE_PUBLICATION_PACKET.md) and
[`submission/YOUKU_PUBLICATION_PACKET_zh.md`](submission/YOUKU_PUBLICATION_PACKET_zh.md).

## Arm Create track

The project targets the Mobile AI track: local inference on an Arm-powered laptop with
measurable model-size, responsiveness, privacy, and offline benefits. It also creates a reusable
developer workflow, which the official judging criteria explicitly value.

Official references:

- [Arm Create challenge](https://arm-ai-optimization-challenge.devpost.com/)
- [Arm Create track details](https://arm-ai-optimization-challenge.devpost.com/details/trackdetails)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime Python on Arm/macOS](https://onnxruntime.ai/docs/get-started/with-python.html)

## Truth and privacy boundaries

- The fixture proves the optimization and evidence workflow, not production model accuracy.
- The compatible-model path proves reusable tooling, not that any third-party or private model is licensed, accurate, or publishable.
- Latency is local evidence for the recorded hardware/runtime and can vary on other systems.
- A smaller model is not automatically a faster model; all measured batches remain visible.
- The report does not collect hostname, username, serial number, account data, or environment
  variables.
- No customer data, credentials, cookies, payment details, or private models are included.
- No claim in this repository is evidence of a hackathon submission or prize.

## License

MIT. See `LICENSE`.
