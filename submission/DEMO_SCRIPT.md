# ArmQuantLab demo script

Target: a judge-readable English demo below three minutes. It uses only checked-in frozen native
Arm64 evidence; the video build must not rerun or replace the benchmark metrics.

## Storyboard

1. **Claim** — Optimization needs size, parity, protocol, and every batch rather than one
   favorable number.
2. **Evidence contract** — Tie footprint, output parity, and per-batch latency to one frozen model
   pair and report.
3. **Protocol** — Show native Arm64, CPU provider, one thread, ten warm-ups, forty measured
   samples, and two hundred invocations per sample.
4. **Pipeline** — Explain deterministic source, preprocessing, dynamic per-channel INT8, and
   paired inputs.
5. **Footprint** — Show 1,446,989 bytes versus 372,905 bytes and the 74.229% reduction.
6. **Parity** — Show all three cosine-similarity values above 0.99989 without turning that into a
   production-accuracy claim.
7. **Batch table** — Keep batch 8's 1.281611x improvement and the batch 1 and 32 regressions
   visible together.
8. **Speedup profile** — Plot the three results against the 1.0x break-even line.
9. **Reusable path** — Explain the single-file, explicit-op, bounded-input, rights-acknowledged,
   local-only compatible-model gate.
10. **Privacy and truth** — State what is not collected, persisted, or proven.
11. **Release verification** — Show the actual verifier result for architecture, models, batches,
    parity, and privacy.
12. **Decision** — Smaller: yes. Faster everywhere: no. Reproducible: yes.

The exact narration lives in `scripts/build_demo_frames.py`. Scene timing is derived from the
generated AIFF narration and written to `dist/demo-video/timeline.json`; sentence-level captions
are written to `dist/demo-video/ArmQuantLab_Demo_en.srt`.

## Recording boundary

- Use only the generated 1920x1080 synthetic storyboard.
- Do not show a desktop, menu bar, notification, local username, account, or shell history.
- Do not display a private model, customer data, credential, environment variable, or account
  page.
- Do not rerun the frozen benchmark merely to obtain a more favorable result.
- Use no third-party music, stock footage, generated hardware image, or logo treatment.
- Follow `VIDEO_BUILD.md`, then verify the hosted video and captions in a signed-out browser.
