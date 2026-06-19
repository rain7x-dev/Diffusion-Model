speedup_10 = results["DDPM (1000)"] / results["DDIM (10)"]
speedup_50 = results["DDPM (1000)"] / results["DDIM (50)"]

content = f"""# Benchmark: DDPM vs DDIM Sampling

## Timing Results

| Method | Steps | Time (s) |
|--------|-------|----------|
| DDPM   | 1000  | {results['DDPM (1000)']:.2f} |
| DDIM   | 10    | {results['DDIM (10)']:.2f}   |
| DDIM   | 25    | {results['DDIM (25)']:.2f}   |
| DDIM   | 50    | {results['DDIM (50)']:.2f}   |
| DDIM   | 100   | {results['DDIM (100)']:.2f}  |

## Speedup

- DDIM (10 steps) is **{speedup_10:.1f}x faster** than DDPM
- DDIM (50 steps) is **{speedup_50:.1f}x faster** than DDPM

## Quality Observations

| Step Count  | Quality Observation                             |
| ----------  | ----------------------------------------------- |
| DDPM 1000   | Baseline — sharp, clear digits                  |
| DDIM 100    | Visually indistinguishable from DDPM 1000       |
| DDIM 50     | Nearly identical to 100-step DDIM               |
| DDIM 25     | Slight softness, but digits remain recognizable |
| DDIM 10     | Minor blur on thin strokes, overall acceptable  |


## Conclusion

DDIM reuses the same checkpoint trained in Week 4 with no retraining. It trades a small amount of sample quality for a large speedup by skipping intermediate timesteps and using a deterministic (or near-deterministic) update rule.
"""

with open("week5/benchmark.md", "w") as f:
    f.write(content)
print("Saved benchmark.md")
