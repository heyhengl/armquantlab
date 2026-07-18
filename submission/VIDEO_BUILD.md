# Reproducible demo-video build

The video is generated from `evidence/benchmark-report.json`, `evidence/freeze-receipt.json`, and
the actual release verifier. It does not rerun the frozen benchmark, record a desktop or browser,
or process a private model.

## Build on macOS

```bash
uv run --python 3.12 --no-project --no-cache \
  --with '.[dev]' --with pillow \
  python scripts/build_demo_frames.py

swiftc -parse-as-library scripts/render_demo.swift \
  -o dist/demo-video/render_demo

dist/demo-video/render_demo \
  dist/demo-video/scene-manifest.json \
  dist/demo-video/frames \
  dist/demo-video/audio \
  dist/demo-video/ArmQuantLab_Demo_1080p_en.mp4 \
  dist/demo-video/timeline.json
```

The Python builder records `metrics_rerun: false` in its manifest. The Swift renderer refuses a
timeline of three minutes or longer and emits `ArmQuantLab_Demo_en.srt` with sentence-level
English cues.

## Verified local artifact

- Render date: 2026-07-18 Asia/Shanghai.
- Video: `dist/demo-video/ArmQuantLab_Demo_1080p_en.mp4`.
- Duration: 171.777 seconds.
- Tracks: one H.264 video track and one AAC audio track.
- Canvas: 1920x1080.
- Video SHA-256: `b9e59b54a4dc830de72e65b7b4f5b2c5b7f286869940318a486aade1e70b3682`.
- Captions: 32 cues, no line longer than 54 characters.
- SRT SHA-256: `460f655b6036ab28773c11dbe7049a8c3dfc141384ae89601e032b6acfa043fb`.
- Quick Look decoded the final cover frame without rotation or crop errors.
- Spotlight reports no author, creator, or download-origin metadata.
- A targeted binary-string scan found no local user path, username, or common email domain.

## Publication boundary

- `dist/demo-video/` is generated and excluded from the source-and-evidence archive.
- The archive includes both video-build source files and the written design rationale.
- Upload only the finished MP4 and SRT to a free judge-accessible video host.
- After upload, verify duration, playback, captions, and access in a signed-out browser.
- Do not replace the frozen benchmark metrics with a later run or a more favorable subset.
