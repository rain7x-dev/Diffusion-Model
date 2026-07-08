# Week 7 Training Log — Butterfly Diffusion (64×64)

## Dataset
- Smithsonian Butterflies subset (HF: huggan/smithsonian_butterflies_subset), 1000 RGB images
- Chosen for: small size (T4-friendly), tutorial support, visually strong output
- Preprocessing: resize 64×64, random horizontal flip, normalize to [-1, 1]
- Augmentations deliberately excluded: color jitter (would distort the natural color
  distribution the model must learn), vertical flip (butterflies have a canonical orientation)

## Training runs

### v1 — baseline (100 → 250 epochs)
- UNet 6.2M params (base=64), linear schedule, Adam lr 2e-4, batch 64, fp32
- Result: ~10/16 recognizable butterflies; pale/ghost failures; quality plateaued by ~epoch 100
- Problem discovered: only `latest.pt` kept → a subjectively better epoch-210 model was
  unrecoverable; grids used fresh random seeds → epoch-to-epoch comparisons unreliable

### v2 — over-tuned (FAILED)
- Added: base=96 (13M), EMA 0.999, cosine schedule, fp16 (amp), warmup 500, grad clip 1.0
- Result: raw final weights collapsed (near-all-black samples); EMA masked it partially
- Lessons: EMA decay 0.999 averages over ~1000 steps — half of this run's 2,250 total steps;
  hyperparameters from large-scale recipes must be rescaled to run length. Prime divergence
  suspects: cosine schedule's near-zero alpha_bar terms + fp16 interaction.

### v3 — final (150 epochs)
- Kept from v2: base=96, grad clipping, LR warmup, fixed-seed eval, milestone checkpoints
- Reverted: linear schedule, fp32
- Rescaled: EMA decay 0.995 (~200-step window), warmup 100 steps
- Result: clean butterflies across the fixed-seed grid; epoch 150 selected as best milestone
- Final sampling: DDIM, 50 steps, eta=0, from EMA weights

## Key decisions & lessons
1. Fixed-seed evaluation grids (from HF tutorial) made checkpoint comparison meaningful —
   without it, v1's "epoch 210 vs 250" question was unanswerable.
2. Keeping every 10th checkpoint costs ~2.5 GB Drive but eliminates unrecoverable-best-model risk.
3. Checkpoint/resume to Drive survived multiple Colab disconnects with zero loss.
4. Diffusion loss correlates weakly with sample quality — decisions were made from grids, not curves.

## With 10× compute
Larger model (base 128+) at 128×128, longer runs with properly-scaled EMA, an A/B of
cosine-vs-linear in fp32 to isolate v2's true failure cause, and FID tracking instead of
eyeballing grids.
