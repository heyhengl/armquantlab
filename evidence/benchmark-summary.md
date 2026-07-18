# ArmQuantLab benchmark summary

- Architecture: `arm64`
- Provider: `CPUExecutionProvider`
- FP32 model: `1446989` bytes
- INT8 model: `372905` bytes
- Size reduction: `74.229%`

| Batch | FP32 p50 ms | INT8 p50 ms | Speedup | Cosine similarity |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.008125 | 0.010221 | 0.794915x | 0.999891129 |
| 8 | 0.024307 | 0.018966 | 1.281611x | 0.999906593 |
| 32 | 0.03432 | 0.053216 | 0.644923x | 0.999898529 |

## Truth boundary

Measures runtime optimization and numerical parity only; it does not prove task accuracy or production model quality.
Latency is local evidence for this run and can vary with hardware, load, and runtime version.
