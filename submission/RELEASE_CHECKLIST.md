# Release and submission checklist

A checked item must correspond to a reproducible artifact or public receipt.

## Local technical release

- [x] Frozen environment records native `arm64` and ONNX Runtime CPU execution.
- [x] Three model files match the freeze receipt.
- [x] FP32 and INT8 hashes match the benchmark report.
- [x] Model footprint changes from 1,446,989 to 372,905 bytes, a 74.229% reduction.
- [x] All three batches preserve cosine similarity above 0.99989.
- [x] Batch 8 improves p50 latency; batches 1 and 32 remain visible as regressions.
- [x] Compatible-model processing requires explicit rights acknowledgement.
- [x] Compatible-model reports omit source paths, filenames, and generated input values.
- [x] Six automated tests pass.
- [x] Ruff formatting and static checks pass.
- [x] Release verifier passes with zero public-text privacy findings.
- [x] Deterministic archive uses an explicit allowlist and excludes `dist/`.
- [x] Video-build Python and Swift sources are included in the allowlist.

## Local demo

- [x] Twelve 1920x1080 scenes use the frozen report without rerunning metrics.
- [x] Visual direction passes the written anti-AI-template gate.
- [x] Final MP4 is 171.777 seconds with one H.264 video and one AAC audio track.
- [x] English SRT contains 32 sentence-level cues with a 54-character line limit.
- [x] Decode the finished MP4 cover frame and inspect orientation and crop.
- [x] Confirm author, creator, and download-origin metadata are absent.
- [x] Targeted binary-string scan finds no local path, username, or common email domain.

## Public release

- [x] Create public MIT repository `https://github.com/heyhengl/armquantlab`.
- [x] Record public commit `e270c9f0cad0a74b822ee0c3d866e1fd1c922d38`.
- [x] Verify the repository, license, frozen evidence, and installation in a signed-out browser.
- [ ] Publish the MP4 with public or unlisted judge-accessible visibility.
- [ ] Verify hosted playback and captions in a signed-out browser.
- [ ] Add public repository and video URLs to the Devpost draft.

## Devpost and Arm Developer

- [ ] User fills required personal fields only inside the platform.
- [ ] Select the live label `Track 3: Mobile AI`; do not use `Track 1`.
- [x] Confirm the project/testing URL is free, unrestricted, and requires no login.
- [ ] Disclose submission-period creation, open-source dependencies, AI coding assistance, code-generated visuals, and synthetic narration.
- [ ] Confirm eligibility and live final fields on the submission date.
- [ ] Confirm final submission is completely free and causes no financial movement.
- [ ] Submit only verified public URLs and exact frozen metrics.
- [ ] Record the Devpost project URL and submission receipt.

## Financial evidence

- Settled or withdrawable net income: RMB 0.
- Startup spend: RMB 0.
- Prize pool, local files, public URLs, submission, judging, and award notices are not income.
- Only settled or withdrawable evidence may change the income total.
