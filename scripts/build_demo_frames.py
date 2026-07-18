#!/usr/bin/env python3
"""Build an evidence-grounded ArmQuantLab video storyboard and narration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080
BACKGROUND = "#F3F2ED"
INK = "#151B20"
MUTED = "#667079"
DIVIDER = "#CECEC7"
TEAL = "#176D66"
TEAL_PALE = "#DCE9E5"
AMBER = "#A65A30"
AMBER_PALE = "#F0E1D6"
PROTOCOL = "#475F70"
SURFACE = "#E7E5DE"
TERMINAL = "#13191D"
TERMINAL_TEXT = "#E6ECEB"
TERMINAL_MUTED = "#96A1A5"

FONT_DISPLAY = "/System/Library/Fonts/SFCompact.ttf"
FONT_SANS = "/System/Library/Fonts/SFNS.ttf"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"


@dataclass(frozen=True)
class Scene:
    scene_id: str
    eyebrow: str
    title: str
    narration: str
    caption: str
    mode: str


SCENES = (
    Scene(
        "01",
        "ARMQUANTLAB / MOBILE AI",
        "Measure the trade-off, not the hype",
        (
            "A useful optimization claim needs more than one favorable number. ArmQuantLab keeps "
            "model size, numerical parity, protocol, and every measured batch together, including "
            "the two cases where INT8 is slower."
        ),
        "One model pair. Three batches. No cherry-picked conclusion.",
        "cover",
    ),
    Scene(
        "02",
        "EVIDENCE CONTRACT",
        "Three claims must agree",
        (
            "The decision has three parts. Is the model smaller? Are outputs still numerically "
            "aligned? And for which batch shapes is latency actually better? Each answer is tied "
            "to the same frozen model hashes and benchmark report."
        ),
        "Footprint, parity, and response time belong to one evidence bundle.",
        "contract",
    ),
    Scene(
        "03",
        "FROZEN NATIVE RUN",
        "A fixed Arm64 protocol",
        (
            "The frozen run uses native Arm64, ONNX Runtime's CPU provider, and one CPU thread. "
            "Each batch gets ten warm-ups, forty measured samples, and two hundred invocations per "
            "sample. The video does not rerun or replace those metrics."
        ),
        "Hardware and protocol context stay attached to the result.",
        "protocol",
    ),
    Scene(
        "04",
        "OPTIMIZATION PATH",
        "From source graph to paired evidence",
        (
            "The workflow creates a deterministic ONNX graph, runs shape and graph preprocessing, "
            "then applies dynamic per-channel INT8 weight quantization. FP32 and INT8 receive the "
            "same deterministic inputs before the evidence is frozen."
        ),
        "The comparison changes representation, not the benchmark input.",
        "pipeline",
    ),
    Scene(
        "05",
        "MODEL FOOTPRINT",
        "INT8 is 74.229% smaller",
        (
            "The preprocessed FP32 model is one million four hundred forty-six thousand nine "
            "hundred eighty-nine bytes. INT8 is three hundred seventy-two thousand nine hundred "
            "five bytes, a measured reduction of seventy-four point two two nine percent."
        ),
        "The strongest result is footprint, not universal latency.",
        "size",
    ),
    Scene(
        "06",
        "NUMERICAL PARITY",
        "Every batch stays above 0.99989 cosine similarity",
        (
            "Compression is not useful if the output diverges. Across batches one, eight, and "
            "thirty-two, cosine similarity remains above zero point nine nine nine eight nine. "
            "The report also keeps maximum and mean absolute error."
        ),
        "Parity is measured beside performance, not assumed from model size.",
        "parity",
    ),
    Scene(
        "07",
        "BATCH DECISION TABLE",
        "One faster batch. Two regressions.",
        (
            "Batch eight improves p50 latency by one point two eight two times. Batch one slows to "
            "zero point seven nine five times, and batch thirty-two slows to zero point six four "
            "five times. ArmQuantLab keeps all three rows visible."
        ),
        "The decision depends on workload shape and quantization overhead.",
        "table",
    ),
    Scene(
        "08",
        "SPEEDUP PROFILE",
        "The curve crosses the break-even line once",
        (
            "The speedup profile is non-monotonic. INT8 wins only at batch eight in this fixture. "
            "That makes the next engineering question concrete: match optimization to the real "
            "batch profile instead of treating INT8 as automatically faster."
        ),
        "A smaller model can still lose the latency comparison.",
        "chart",
    ),
    Scene(
        "09",
        "REUSABLE LOCAL PATH",
        "Compatible models stay rights-gated",
        (
            "A separate command can process a compatible model the operator is authorized to use. "
            "It rejects symlinks, external tensor files, oversized models, unresolved shapes, and "
            "unsupported ops, while keeping the model and derivatives local."
        ),
        "Rights acknowledgement is a gate, not a substitute for license review.",
        "gate",
    ),
    Scene(
        "10",
        "PRIVACY + TRUTH BOUNDARY",
        "What the evidence deliberately does not claim",
        (
            "The public fixture collects no hostname, username, account data, credential, source "
            "path, or input value. It proves reproducible runtime optimization and numerical "
            "parity, not production task accuracy or universal performance."
        ),
        "Scope is part of the evidence, not fine print after the result.",
        "boundary",
    ),
    Scene(
        "11",
        "RELEASE VERIFICATION",
        "Models, hashes, batches, and privacy are checked together",
        (
            "The release verifier confirms native Arm64, three measured batches, a smaller INT8 "
            "model, output parity, model hashes, and a zero-finding public-text privacy scan. Six "
            "tests also cover deterministic generation and the compatible-model gates."
        ),
        "The public archive is allowlisted and excludes generated local candidates.",
        "verify",
    ),
    Scene(
        "12",
        "DECISION",
        "Smaller: yes. Faster everywhere: no. Reproducible: yes.",
        (
            "ArmQuantLab makes the trade-off reviewable. The model is substantially smaller, "
            "output parity remains high, one batch is faster, and two are slower. That honest "
            "profile is more useful than a universal optimization claim."
        ),
        "Evidence first, including the result that does not flatter the optimization.",
        "close",
    ),
)


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


DISPLAY_82 = font(FONT_DISPLAY, 82)
DISPLAY_62 = font(FONT_DISPLAY, 62)
DISPLAY_46 = font(FONT_DISPLAY, 46)
SANS_34 = font(FONT_SANS, 34)
SANS_29 = font(FONT_SANS, 29)
SANS_25 = font(FONT_SANS, 25)
SANS_21 = font(FONT_SANS, 21)
SANS_18 = font(FONT_SANS, 18)
MONO_42 = font(FONT_MONO, 42)
MONO_31 = font(FONT_MONO, 31)
MONO_26 = font(FONT_MONO, 26)
MONO_22 = font(FONT_MONO, 22)
MONO_19 = font(FONT_MONO, 19)


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def header(draw: ImageDraw.ImageDraw, scene: Scene, index: int) -> None:
    draw.text((78, 58), scene.eyebrow, font=SANS_18, fill=PROTOCOL)
    draw.text((1842, 58), f"{index:02d} / {len(SCENES):02d}", font=MONO_19, fill=MUTED, anchor="ra")
    draw.line((78, 110, 1842, 110), fill=DIVIDER, width=2)


def footer(draw: ImageDraw.ImageDraw, scene: Scene, index: int) -> None:
    draw.line((78, 973, 1842, 973), fill=DIVIDER, width=2)
    draw.text((78, 998), scene.caption, font=SANS_21, fill=INK)
    draw.rectangle((78, 1053, 1842, 1057), fill=DIVIDER)
    progress = 78 + int(1764 * index / len(SCENES))
    draw.rectangle((78, 1053, progress, 1057), fill=PROTOCOL)


def title(draw: ImageDraw.ImageDraw, scene: Scene) -> None:
    draw.text((78, 154), wrap(scene.title, 39), font=DISPLAY_62, fill=INK, spacing=5)


def rail(draw: ImageDraw.ImageDraw, heading: str, rows: list[tuple[str, str, str]]) -> None:
    draw.line((1396, 188, 1396, 920), fill=DIVIDER, width=2)
    draw.text((1434, 203), heading.upper(), font=SANS_18, fill=MUTED)
    y = 256
    for label, value, tone in rows:
        color = TEAL if tone == "good" else AMBER if tone == "tradeoff" else INK
        draw.text((1434, y), label.upper(), font=SANS_18, fill=MUTED)
        y += 34
        draw.text((1434, y), wrap(value, 23), font=SANS_29, fill=color, spacing=7)
        line_count = len(textwrap.wrap(value, width=23))
        y += 62 + max(0, line_count - 1) * 34
        draw.line((1434, y, 1842, y), fill=DIVIDER, width=1)
        y += 36


def metric_bar(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    value: str,
    ratio: float,
    color: str,
    maximum_width: int = 1120,
) -> None:
    draw.text((100, y), label, font=SANS_25, fill=MUTED)
    draw.text((1315, y), value, font=MONO_26, fill=INK, anchor="ra")
    draw.rectangle((100, y + 48, 100 + maximum_width, y + 86), fill=SURFACE)
    draw.rectangle((100, y + 48, 100 + int(maximum_width * ratio), y + 86), fill=color)


def terminal(draw: ImageDraw.ImageDraw, lines: list[tuple[str, str]], y: int = 324) -> None:
    draw.rectangle((78, y, 1346, 914), fill=TERMINAL)
    draw.rectangle((78, y, 1346, y + 42), fill="#1D252A")
    draw.rectangle((104, y + 15, 116, y + 27), fill="#B85C54")
    draw.rectangle((126, y + 15, 138, y + 27), fill="#C29643")
    draw.rectangle((148, y + 15, 160, y + 27), fill="#5A956B")
    ty = y + 76
    for kind, text in lines:
        color = TERMINAL_TEXT if kind == "out" else TERMINAL_MUTED
        if kind == "cmd":
            color = "#82B9B1"
        prefix = "$ " if kind == "cmd" else "  "
        for line in text.splitlines():
            draw.text((112, ty), prefix + line, font=MONO_22, fill=color)
            prefix = "  "
            ty += 36


def scene_image(scene: Scene, index: int, context: dict[str, Any]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    header(draw, scene, index)
    footer(draw, scene, index)
    report = context["report"]
    batches = report["batches"]

    if scene.mode == "cover":
        draw.text((106, 220), "ArmQuantLab", font=DISPLAY_82, fill=INK)
        draw.text((106, 356), wrap(scene.title, 28), font=DISPLAY_62, fill=INK, spacing=8)
        draw.line((106, 572, 986, 572), fill=PROTOCOL, width=5)
        draw.text((106, 624), "FP32  →  dynamic INT8", font=MONO_31, fill=PROTOCOL)
        draw.text(
            (106, 704),
            wrap("Frozen Arm64 evidence · same inputs · every measured batch", 48),
            font=SANS_34,
            fill=MUTED,
            spacing=10,
        )
        draw.text((1450, 254), "FOOTPRINT", font=SANS_18, fill=MUTED)
        draw.text((1450, 292), "−74.229%", font=MONO_42, fill=TEAL)
        draw.line((1450, 370, 1842, 370), fill=DIVIDER, width=1)
        draw.text((1450, 420), "LATENCY", font=SANS_18, fill=MUTED)
        draw.text((1450, 458), "1 / 3 faster", font=MONO_31, fill=AMBER)
        return image

    title(draw, scene)
    if scene.mode == "contract":
        rows = [
            ("01", "FOOTPRINT", "Are the bytes materially lower?"),
            ("02", "PARITY", "Do paired outputs remain aligned?"),
            ("03", "LATENCY", "Which batch profiles actually improve?"),
        ]
        y = 326
        for number, label, question in rows:
            draw.text((100, y), number, font=MONO_31, fill=PROTOCOL)
            draw.text((230, y), label, font=SANS_29, fill=INK)
            draw.text((560, y), question, font=SANS_29, fill=MUTED)
            draw.line((100, y + 66, 1330, y + 66), fill=DIVIDER, width=1)
            y += 146
        rail(
            draw,
            "Evidence binding",
            [
                ("Models", "FP32 + INT8 hashes", "neutral"),
                ("Inputs", "Same deterministic values", "neutral"),
                ("Report", "One frozen bundle", "good"),
            ],
        )
    elif scene.mode == "protocol":
        env = report["environment"]
        protocol = report["protocol"]
        draw.text((98, 330), env["architecture"], font=DISPLAY_82, fill=PROTOCOL)
        draw.text((98, 438), "native architecture", font=SANS_25, fill=MUTED)
        draw.line((98, 510, 1325, 510), fill=DIVIDER, width=2)
        items = [
            ("CPU provider", env["provider"]),
            ("Threads", str(env["thread_count"])),
            ("Warm-ups", str(protocol["warmup_runs"])),
            ("Measured samples", str(protocol["measured_runs"])),
            ("Invocations / sample", str(protocol["iterations_per_sample"])),
        ]
        y = 560
        for label, value in items:
            draw.text((100, y), label, font=SANS_25, fill=MUTED)
            draw.text((1310, y), value, font=MONO_26, fill=INK, anchor="ra")
            draw.line((100, y + 44, 1325, y + 44), fill=DIVIDER, width=1)
            y += 66
        rail(
            draw,
            "Frozen result",
            [
                ("Runtime", f"ONNX {env['onnxruntime']}", "neutral"),
                ("Batches", "1 · 8 · 32", "neutral"),
                ("Video", "No metric rerun", "good"),
            ],
        )
    elif scene.mode == "pipeline":
        optimization = report["optimization"]
        stages = [
            ("01", "SOURCE GRAPH", "deterministic fixture", PROTOCOL),
            ("02", "PREPROCESSED FP32", f"{optimization['fp32_bytes']:,} bytes", INK),
            ("03", "DYNAMIC INT8", f"{optimization['int8_bytes']:,} bytes", TEAL),
            ("04", "PAIRED BENCHMARK", "same inputs · all batches", AMBER),
        ]
        y = 332
        for number, label, detail, color in stages:
            draw.text((100, y), number, font=MONO_31, fill=color)
            draw.text((220, y), label, font=SANS_29, fill=INK)
            draw.text((220, y + 46), detail, font=MONO_22, fill=MUTED)
            if number != "04":
                draw.line((115, y + 88, 115, y + 136), fill=DIVIDER, width=3)
            y += 145
        rail(
            draw,
            "Quantization",
            [
                ("Method", "Dynamic per-channel", "neutral"),
                ("Weights", "INT8", "neutral"),
                ("Inputs", "Deterministic FP32", "good"),
            ],
        )
    elif scene.mode == "size":
        optimization = report["optimization"]
        metric_bar(
            draw,
            y=352,
            label="PREPROCESSED FP32",
            value=f"{optimization['fp32_bytes']:,} B",
            ratio=1.0,
            color=PROTOCOL,
        )
        metric_bar(
            draw,
            y=520,
            label="DYNAMIC INT8",
            value=f"{optimization['int8_bytes']:,} B",
            ratio=optimization["int8_bytes"] / optimization["fp32_bytes"],
            color=TEAL,
        )
        draw.text((100, 704), "−74.229%", font=DISPLAY_82, fill=TEAL)
        draw.text((100, 810), "measured footprint reduction", font=SANS_29, fill=MUTED)
        rail(
            draw,
            "Decision",
            [
                ("Smaller", "Yes", "good"),
                ("Digest", optimization["int8_sha256"][:16] + "…", "neutral"),
                ("Faster", "Evaluate by batch", "tradeoff"),
            ],
        )
    elif scene.mode == "parity":
        draw.text((100, 334), "> 0.99989", font=DISPLAY_82, fill=TEAL)
        draw.text((100, 440), "cosine similarity floor", font=SANS_29, fill=MUTED)
        draw.line((100, 510, 1326, 510), fill=DIVIDER, width=2)
        y = 568
        for batch in batches:
            similarity = batch["output_parity"]["cosine_similarity"]
            draw.text((100, y), f"BATCH {batch['batch_size']}", font=SANS_25, fill=MUTED)
            draw.text((1316, y), f"{similarity:.9f}", font=MONO_31, fill=INK, anchor="ra")
            draw.line((100, y + 52, 1326, y + 52), fill=DIVIDER, width=1)
            y += 92
        rail(
            draw,
            "Parity gate",
            [
                ("Threshold", "At least 0.999", "neutral"),
                ("All batches", "Pass", "good"),
                ("Task accuracy", "Not claimed", "tradeoff"),
            ],
        )
    elif scene.mode == "table":
        columns = [(100, "BATCH"), (340, "FP32 p50"), (650, "INT8 p50"), (960, "SPEEDUP")]
        for x, label in columns:
            draw.text((x, 334), label, font=SANS_18, fill=MUTED)
        draw.text((1308, 334), "DECISION", font=SANS_18, fill=MUTED, anchor="ra")
        y = 392
        for batch in batches:
            speedup = batch["p50_speedup"]
            good = speedup > 1
            color = TEAL if good else AMBER
            verdict = "FASTER" if good else "SLOWER"
            draw.text((100, y), str(batch["batch_size"]), font=MONO_31, fill=INK)
            draw.text((340, y), f"{batch['fp32_latency_ms']['p50']:.6f} ms", font=MONO_26, fill=INK)
            draw.text((650, y), f"{batch['int8_latency_ms']['p50']:.6f} ms", font=MONO_26, fill=INK)
            draw.text((960, y), f"{speedup:.6f}x", font=MONO_26, fill=color)
            draw.text((1308, y), verdict, font=SANS_25, fill=color, anchor="ra")
            draw.line((100, y + 62, 1328, y + 62), fill=DIVIDER, width=1)
            y += 126
        rail(
            draw,
            "Measured outcome",
            [
                ("Faster", "Batch 8", "good"),
                ("Slower", "Batch 1 + 32", "tradeoff"),
                ("Universal win", "No", "tradeoff"),
            ],
        )
    elif scene.mode == "chart":
        left, top, right, bottom = 150, 340, 1290, 840
        minimum, maximum = 0.55, 1.35

        def chart_y(value: float) -> int:
            return int(bottom - (value - minimum) / (maximum - minimum) * (bottom - top))

        baseline = chart_y(1.0)
        draw.line((left, baseline, right, baseline), fill=PROTOCOL, width=3)
        draw.text((left, baseline - 36), "1.0× BREAK-EVEN", font=SANS_18, fill=PROTOCOL)
        xs = [left + 120, left + 560, left + 1000]
        points = [(x, chart_y(batch["p50_speedup"])) for x, batch in zip(xs, batches, strict=True)]
        draw.line(points, fill=INK, width=5, joint="curve")
        for (x, y), batch in zip(points, batches, strict=True):
            good = batch["p50_speedup"] > 1
            color = TEAL if good else AMBER
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=color)
            draw.text(
                (x, bottom + 34),
                f"BATCH {batch['batch_size']}",
                font=SANS_21,
                fill=MUTED,
                anchor="ma",
            )
            draw.text(
                (x, y - 34), f"{batch['p50_speedup']:.3f}×", font=MONO_22, fill=color, anchor="ms"
            )
        rail(
            draw,
            "Profile",
            [
                ("Crossings", "One", "neutral"),
                ("Best", "Batch 8", "good"),
                ("Shape", "Non-monotonic", "tradeoff"),
            ],
        )
    elif scene.mode == "gate":
        steps = [
            ("01", "REGULAR ONNX FILE", "non-symlink · self-contained · under 1 GB"),
            ("02", "EXPLICIT OPS", "Conv · Gemm · MatMul subset"),
            ("03", "BOUNDED INPUTS", "float32 · batch-only dynamic · 16M elements"),
            ("04", "RIGHTS ACKNOWLEDGEMENT", "required before model processing"),
            ("05", "LOCAL OUTPUT", "candidate and derivative stay outside release"),
        ]
        y = 308
        for number, label, detail in steps:
            draw.text((96, y), number, font=MONO_26, fill=PROTOCOL)
            draw.text((210, y), label, font=SANS_25, fill=INK)
            draw.text((210, y + 40), detail, font=SANS_21, fill=MUTED)
            draw.line((96, y + 84, 1328, y + 84), fill=DIVIDER, width=1)
            y += 116
        rail(
            draw,
            "Rejected before work",
            [
                ("External tensors", "Reject", "tradeoff"),
                ("Missing rights ack", "Reject", "tradeoff"),
                ("Source path saved", "False", "good"),
            ],
        )
    elif scene.mode == "boundary":
        left_rows = [
            ("HOSTNAME", "not collected"),
            ("USER IDENTITY", "not collected"),
            ("CREDENTIALS", "not collected"),
            ("INPUT VALUES", "not persisted"),
            ("SOURCE PATH", "not persisted"),
        ]
        right_rows = [
            ("PRODUCTION ACCURACY", "not proven"),
            ("UNIVERSAL SPEEDUP", "not proven"),
            ("THIRD-PARTY RIGHTS", "not inferred"),
            ("HACKATHON ENTRY", "not yet submitted"),
        ]
        draw.text((98, 322), "PRIVACY", font=SANS_18, fill=PROTOCOL)
        draw.text((742, 322), "TRUTH", font=SANS_18, fill=PROTOCOL)
        y = 370
        for label, value in left_rows:
            draw.text((98, y), label, font=SANS_21, fill=MUTED)
            draw.text((600, y), value, font=MONO_22, fill=TEAL, anchor="ra")
            draw.line((98, y + 46, 600, y + 46), fill=DIVIDER, width=1)
            y += 94
        y = 370
        for label, value in right_rows:
            draw.text((742, y), label, font=SANS_21, fill=MUTED)
            draw.text((1320, y), value, font=MONO_22, fill=AMBER, anchor="ra")
            draw.line((742, y + 46, 1320, y + 46), fill=DIVIDER, width=1)
            y += 108
        rail(
            draw,
            "Scope",
            [
                ("Proves", "Runtime + parity", "good"),
                ("Does not prove", "Task quality", "tradeoff"),
                ("Fixture", "Self-owned synthetic", "neutral"),
            ],
        )
    elif scene.mode == "verify":
        verifier = context["verifier"]
        terminal(
            draw,
            [
                ("cmd", "python scripts/verify_release.py"),
                ("out", f"status: {verifier['status']}"),
                ("out", f"architecture: {verifier['architecture']}"),
                ("out", f"measured batches: {verifier['batches']}"),
                ("out", "int8 model smaller: true"),
                ("out", "output parity: true"),
                ("out", f"privacy files scanned: {verifier['privacy_files_scanned']}"),
                ("out", f"privacy findings: {verifier['privacy_findings']}"),
            ],
        )
        rail(
            draw,
            "Release",
            [
                ("Tests", "6 passed", "good"),
                ("Models", "Frozen hashes", "neutral"),
                ("Source", "Allowlisted only", "good"),
            ],
        )
    elif scene.mode == "close":
        conclusions = [
            ("SMALLER", "YES", TEAL),
            ("FASTER EVERYWHERE", "NO", AMBER),
            ("OUTPUT PARITY", "PASS", TEAL),
            ("REPRODUCIBLE", "YES", TEAL),
        ]
        y = 318
        for label, value, color in conclusions:
            draw.text((100, y), label, font=SANS_21, fill=MUTED)
            draw.text((1310, y), value, font=DISPLAY_46, fill=color, anchor="ra")
            draw.line((100, y + 64, 1328, y + 64), fill=DIVIDER, width=1)
            y += 126
        rail(
            draw,
            "Next review",
            [
                ("Use", "Match real batch profile", "neutral"),
                ("Publish", "Fixture evidence only", "good"),
                ("Claim", "Keep regressions visible", "tradeoff"),
            ],
        )
    return image


def run_json(command: list[str], *, cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def build_context(root: Path) -> dict[str, Any]:
    report = json.loads((root / "evidence" / "benchmark-report.json").read_text(encoding="utf-8"))
    freeze = json.loads((root / "evidence" / "freeze-receipt.json").read_text(encoding="utf-8"))
    verifier = run_json([sys.executable, "scripts/verify_release.py"], cwd=root)
    return {"report": report, "freeze": freeze, "verifier": verifier}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dist/demo-video"))
    parser.add_argument("--voice", default="Daniel")
    parser.add_argument("--rate", type=int, default=190)
    parser.add_argument("--skip-audio", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    frames = output / "frames"
    audio = output / "audio"
    frames.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)
    context = build_context(root)

    manifest: list[dict[str, Any]] = []
    for index, scene in enumerate(SCENES, 1):
        frame_path = frames / f"scene-{scene.scene_id}.png"
        scene_image(scene, index, context).save(frame_path, format="PNG", optimize=True)
        audio_path = audio / f"scene-{scene.scene_id}.aiff"
        if not args.skip_audio:
            subprocess.run(
                [
                    "/usr/bin/say",
                    "-v",
                    args.voice,
                    "-r",
                    str(args.rate),
                    "-o",
                    str(audio_path),
                    scene.narration,
                ],
                check=True,
            )
        item = asdict(scene)
        item.update({"frame": frame_path.name, "audio": audio_path.name})
        manifest.append(item)

    payload = {
        "schema_version": 1,
        "project": "ArmQuantLab",
        "width": WIDTH,
        "height": HEIGHT,
        "voice": None if args.skip_audio else args.voice,
        "speech_rate": None if args.skip_audio else args.rate,
        "third_party_music": False,
        "customer_data": False,
        "credentials": False,
        "metrics_source": "checked_in_frozen_native_arm64_evidence",
        "metrics_rerun": False,
        "scenes": manifest,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", serialized):
        raise AssertionError("demo manifest contains an email-like string")
    manifest_path = output / "scene-manifest.json"
    manifest_path.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "audio_generated": not args.skip_audio,
                "frames": len(SCENES),
                "height": HEIGHT,
                "manifest": str(manifest_path),
                "metrics_rerun": False,
                "status": "pass",
                "width": WIDTH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
