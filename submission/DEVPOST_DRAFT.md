# ArmQuantLab — Devpost submission draft

> Draft only. The project has not been registered or submitted. The repository URL is published
> and anonymously verified; the optional video and Devpost project URLs remain pending.

## Project name

ArmQuantLab

## Tagline

Reproducible FP32-to-INT8 ONNX optimization evidence on Arm64.

## Track

Track 3: Mobile AI — local inference and developer tooling on an Arm-powered laptop.

## Project overview

ArmQuantLab turns an optimization claim into a reproducible evidence bundle. It generates a
deterministic FP32 ONNX workload, applies ONNX Runtime's recommended shape and graph
preprocessing followed by dynamic per-channel INT8 weight quantization, runs the preprocessed
FP32 baseline and INT8 model with identical inputs on native Arm64, and records model hashes,
size reduction, p50/p95 latency, throughput, and numerical parity for every measured batch.

It also exposes the same pipeline for a compatible, rights-cleared ONNX model. The reusable path
rejects symlinks, external tensor files, oversized inputs, unresolved non-batch dimensions, and
unsupported quantization ops. It generates deterministic inputs and does not persist source
paths, filenames, or input values.

The public fixture requires no cloud account, API key, model download, customer data, or paid
hardware. It runs offline on Apple Silicon through the ONNX Runtime CPU execution provider.

The submitted project was created during the challenge submission period. It uses ONNX and ONNX
Runtime under their open-source licenses and contains a deterministic synthetic reference model,
not a downloaded or customer-owned model. AI coding assistance was used during development, and
every submitted source file, benchmark claim, visual, and artifact was reviewed against tests,
frozen hashes, and the release verifier. The demo uses code-generated visuals and a synthetic
English voice.

## Why it should win

Optimization examples often highlight a favorable number while omitting output drift, slow
batches, model identity, or the protocol needed to reproduce it. ArmQuantLab keeps the two model
artifacts, SHA-256 digests, benchmark configuration, and all measured batches together. It is a
small, reusable developer workflow rather than a one-off screenshot. The compatible-model path
lets another developer apply the same evidence contract without replacing the public fixture or
silently publishing their model.

## Functionality and output

One command:

```bash
uv run --python 3.12 --no-project --no-cache --with '.' \
  armquantlab demo --out dist/demo
```

produces:

- a deterministic FP32 ONNX model;
- a per-channel INT8 weight-quantized ONNX model;
- a machine-readable benchmark report; and
- a reviewer-friendly Markdown summary.

The report also states what it does not prove: the fixture measures runtime optimization and
numerical parity, not production task accuracy or general model quality.

For a model the operator is authorized to use:

```bash
uv run --python 3.12 --no-project --no-cache --with '.' \
  armquantlab optimize --model /path/model.onnx --out dist/local-candidate \
  --ops MatMul,Gemm --acknowledge-model-rights
```

The command keeps all derivatives local and requires a separate license/privacy/accuracy review
before any publication.

## Arm-specific implementation

- Native `arm64` Python and ONNX Runtime CPU execution.
- Offline local inference on an Arm-powered laptop.
- Dynamic per-channel INT8 quantization for MatMul weights.
- One-thread paired benchmark protocol across batch sizes 1, 8, and 32, with 200 invocations
  averaged into each latency sample.
- Explicit model-size and output-parity gates.
- A bounded compatible-model input contract with mandatory rights acknowledgement and no
  persisted source path or input values.
- No hostname, username, serial number, account information, or environment-variable capture.

## Setup and validation

```bash
uv run --python 3.12 --no-project --no-cache --with '.' \
  armquantlab demo --out dist/demo
uv run --python 3.12 --no-project --no-cache --with '.[dev]' pytest
uv run --python 3.12 --no-project --no-cache --with '.[dev]' python scripts/verify_release.py
```

## Measured result

The frozen native Arm64 run used ONNX Runtime 1.27.0 with one CPU thread. Dynamic INT8
quantization reduced the preprocessed model from 1,446,989 bytes to 372,905 bytes, a 74.229%
reduction. Output cosine similarity remained between 0.999891129 and 0.999906593 across all
measured batches.

| Batch | FP32 p50 ms | INT8 p50 ms | INT8 speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.008125 | 0.010221 | 0.794915x |
| 8 | 0.024307 | 0.018966 | 1.281611x |
| 32 | 0.034320 | 0.053216 | 0.644923x |

INT8 improved p50 latency for batch 8 but was slower for batches 1 and 32. ArmQuantLab keeps
those negative results visible instead of presenting quantization as a universal speedup.

## Public links

- Source repository: `https://github.com/heyhengl/armquantlab`
- Project/testing URL: `https://github.com/heyhengl/armquantlab`
- Browser preview: `https://heyhengl.github.io/armquantlab/`
- Optional rule-compliant demo video: pending public YouTube, Vimeo, or Youku upload

The public repository is the unrestricted test build. It contains the MIT-licensed source,
deterministic model artifacts, frozen benchmark evidence, and setup/validation commands. It
requires no login, paid service, cloud account, API key, or private test credential.

## Truth-boundary checklist

- [x] Public repository and public-site commit `b614f64f41e0d18ff069858365db16a2dfc55f4d` are verified in a signed-out browser.
- [ ] Live form track is exactly `Track 3: Mobile AI`.
- [x] MIT license is visible at the top of the repository.
- [x] Project/testing URL is free, unrestricted, and opens without login.
- [x] Submission-period creation and third-party open-source dependencies are disclosed.
- [x] AI coding assistance, code-generated visuals, and synthetic narration are disclosed truthfully.
- [x] Measured numbers match the checked-in benchmark report exactly.
- [x] No statement generalizes synthetic workload performance to production models.
- [ ] No local candidate model or derivative is published without a separate license and privacy
  review.
- [ ] Arm Developer and Devpost personal fields are filled only by the user.
- [ ] If the optional video is submitted, its YouTube, Vimeo, or Youku watch page matches the
  locally verified MP4/SRT hashes and plays signed out with captions. GitHub Pages is a preview
  and cannot be used as the submission video URL.
- [ ] Final submission is confirmed free and causes no financial movement.
