# ArmQuantLab YouTube publication packet

Status: ready for a free public upload after the user signs in to YouTube Studio. The Arm video is optional, but the official rules require any submitted video to be publicly visible on YouTube, Vimeo, or Youku. The GitHub Pages player is a verified preview, not the final rule-compliant video URL.

## Upload assets

- Video: `dist/demo-video/ArmQuantLab_Demo_1080p_en.mp4`
- Video SHA-256: `b9e59b54a4dc830de72e65b7b4f5b2c5b7f286869940318a486aade1e70b3682`
- Duration: 171.777 seconds
- Format: MP4, 1920x1080, H.264 video, AAC audio
- Captions: `dist/demo-video/ArmQuantLab_Demo_en.srt`
- Captions SHA-256: `460f655b6036ab28773c11dbe7049a8c3dfc141384ae89601e032b6acfa043fb`
- Captions: plain UTF-8 SubRip, 32 cues, maximum caption-line length 54 characters
- Thumbnail: `dist/demo-video/frames/scene-01.png`
- Thumbnail SHA-256: `dfd68a2bc3f41b7df927eff40ca659b4c52eda1d3bd537850d8ee8f1f5ec56db`
- Thumbnail dimensions: 1920x1080

## Paste-ready fields

### Title

`ArmQuantLab — FP32-to-INT8 evidence on Arm64`

### Description

```text
ArmQuantLab is an evidence-first ONNX quantization workflow for native Arm64. It compares a preprocessed FP32 model with dynamic per-channel INT8 using the same deterministic inputs and batch protocol, then freezes model hashes, size, latency, and numerical-parity evidence.

The frozen synthetic result reduces model size from 1,446,989 bytes to 372,905 bytes, a 74.229% reduction. Output cosine similarity remains above 0.99989 for batches 1, 8, and 32. Latency is not uniformly better: batch 8 reaches a 1.281611x p50 speedup, while batches 1 and 32 regress to 0.794915x and 0.644923x. The project keeps those negative results visible.

Source code and frozen evidence (MIT):
https://github.com/heyhengl/armquantlab

Browser preview:
https://heyhengl.github.io/armquantlab/

00:00 Measure the trade-off, not the hype
00:13 Three claims must agree
00:26 A fixed Arm64 protocol
00:42 From source graph to paired evidence
00:56 INT8 is 74.229% smaller
01:10 Every batch stays above 0.99989 cosine similarity
01:24 One faster batch, two regressions
01:39 The curve crosses break-even once
01:52 Compatible models stay rights-gated
02:07 What the evidence deliberately does not claim
02:22 Models, hashes, batches, and privacy checked together
02:39 Smaller: yes. Faster everywhere: no. Reproducible: yes.

The video uses code-generated visuals and a synthetic English voice. It contains no third-party music, human footage, customer data, private models, credentials, account information, or hardware identifiers.

Built independently for the Arm Create: AI Optimization Challenge, Mobile AI category. No Arm, Devpost, or sponsor endorsement is implied.
```

## Frozen claim boundary

- FP32 model: 1,446,989 bytes; SHA-256 `65b2eadbdb8d4077d92303f21b60cbbc02a6b12426991cd10a106e3761a94b15`
- INT8 model: 372,905 bytes; SHA-256 `87adfb162f98aa799c2386baedab3d92f78abfca4bb35a34e469fa97e2ca26af`
- Model size reduction: 74.229%
- Batch 1 p50 ratio: 0.794915x, slower
- Batch 8 p50 ratio: 1.281611x, faster
- Batch 32 p50 ratio: 0.644923x, slower
- Cosine similarity: above 0.99989 for every measured batch
- Do not summarize this result as a universal speedup or production-model accuracy claim.

## YouTube details

- Video language: English
- Caption language: English
- Category: Science & Technology
- Tags: `Arm64`, `ONNX`, `ONNX Runtime`, `INT8`, `Quantization`, `Mobile AI`, `AI Optimization`, `Benchmark`, `Open Source`
- Thumbnail: upload the verified `scene-01.png` if available without additional account verification; otherwise select the clearest generated frame and record the limitation
- Paid promotion: No
- Altered or synthetic content: Yes, as a conservative transparency choice
- License: Standard YouTube License; do not grant broader video reuse rights merely because the code is MIT-licensed
- Allow embedding: On
- Recording date and location: leave blank
- Monetization, shopping, memberships, donations, and paid promotion: Off or not enabled

## Audience and visibility

- Audience: `No, it's not made for kids.` This is a professional developer-tool and benchmark demonstration.
- Age restriction: No
- Visibility: Public
- Schedule: Off
- Premiere: Off
- The watch page must play without sign-in, a password, a subscription, or payment.

## Caption upload

Upload `ArmQuantLab_Demo_en.srt` as English captions **with timing**. YouTube supports plain UTF-8 SubRip files. After processing, verify all 32 cues and sample the first, middle, and final captions against the narration.

## Publication checks

1. Recompute the MP4, SRT, and thumbnail SHA-256 values before upload.
2. Confirm no fee, purchase, paid promotion, payment profile, monetization enrollment, or revenue feature is required.
3. Keep the video public and embeddable; do not use private, unlisted, members-only, scheduled, or premiere-only visibility for the Devpost URL.
4. Disclose the synthetic narration and code-generated visuals.
5. Wait for 1080p processing.
6. Open the watch URL signed out and verify title, full 171.777-second playback, 1080p, public visibility, description links, 12 chapters, disclosure, and English captions.
7. Record the public watch URL and verification timestamp before updating Devpost.

## Stop conditions

- Any fee, purchase, recharge, deposit, paid storage, paid promotion, or payment-profile requirement.
- Any monetization, revenue sharing, shopping, memberships, donations, collection, settlement, or withdrawal setup.
- Any request for unrelated identity documents, payment details, contacts, browser data, cookies, credentials, private models, customer data, or device identifiers.
- Any supervised-account, strike, age/identity verification, or creator gate requiring user-only information or an unreviewed legal declaration.

## Official references rechecked 2026-07-19

- Arm challenge overview and Mobile AI category: <https://arm-ai-optimization-challenge.devpost.com/>
- Arm official rules and allowed video hosts: <https://arm-ai-optimization-challenge.devpost.com/rules>
- YouTube upload fields: <https://support.google.com/youtube/answer/57407?hl=en>
- Altered or synthetic content disclosure: <https://support.google.com/youtube/answer/14328491>
- Caption upload flow: <https://support.google.com/youtube/answer/2734796?hl=en>
- Supported plain UTF-8 SRT captions: <https://support.google.com/youtube/answer/2734698?hl=en-GB>
- Audience setting: <https://support.google.com/youtube/answer/9527654?co=GENIE.Platform%3DDesktop&hl=en>
- Recommended MP4, H.264, AAC encoding: <https://support.google.com/youtube/answer/1722171>
