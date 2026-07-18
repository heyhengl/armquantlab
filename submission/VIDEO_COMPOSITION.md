# ArmQuantLab video composition brief

## Product decision

- Viewer: an Arm hackathon judge or developer evaluating an optimization claim.
- Primary task: decide where the INT8 trade-off is useful and whether the evidence is repeatable.
- Focal evidence: model footprint, per-batch p50 latency, output parity, and fixed protocol.
- Supporting evidence: native Arm64 environment, model hashes, privacy boundary, and the
  rights-cleared compatible-model gate.
- On demand: implementation detail that does not change the measured decision.

## Structural directions considered

### A. Benchmark notebook + protocol rail — selected

- Weight: 72% benchmark evidence, 28% quiet protocol and truth-boundary rail.
- Reading order: claim, measured comparison, protocol, decision.
- Interaction path: footprint first, parity second, then the three-batch latency decision.
- Density: one dominant evidence object per scene; no card grid.
- Type: compact sans for claims, monospaced numerals for measurements and hashes.
- Color: teal means measured benefit; amber means measured regression; blue-grey identifies
  protocol. Color never substitutes for the numeric result.

This direction makes the non-universal result the product's focal object. It fits a video because
each scene can advance one experimental claim while the stable rail preserves context.

### B. FP32 / INT8 change-diff workspace

- Weight: 55% FP32, 45% INT8 with a narrow decision bar.
- Reading order: before, after, then difference.
- Strength: direct model comparison.
- Rejected for the video: equal halves overstate symmetry and make the three batch profiles feel
  secondary even though they determine the latency decision.

### C. Optimization pipeline canvas + evidence inspector

- Weight: 68% pipeline, 32% inspector.
- Reading order: source model through preprocessing, quantization, benchmark, and freeze.
- Strength: explains implementation sequence.
- Rejected for the video: the pipeline risks becoming decorative flow-chart theater; the measured
  trade-off should own the frame.

## Anti-AI gate

- No dark neon base, glow, gradient, glass surface, or cyberpunk styling.
- No repeated rounded metric cards, generic AI claims, decorative chart, or equal-density columns.
- No metric appears unless it changes the optimization decision or proves the protocol.
- No stock image, generated hardware illustration, third-party logo, or background music.
- Whitespace is kept around the active measurement instead of being filled with feature inventory.
- Negative results receive the same typographic weight as the favorable batch.

## Visual system

- 1920x1080 open warm-neutral canvas.
- Twelve-column alignment with a stable 72/28 evidence-and-protocol split.
- Deep ink body text, teal measured benefit, amber measured regression, blue-grey protocol.
- Straight dividers and square evidence surfaces; minimal containers and no decorative radius.
- A fixed bottom evidence sentence and progress rule create continuity without framing every region.
