# Resuming the ch32u training run

Paused at **step 22000 / 60000** on 2026-09-03.

## Resume

```
cd ~/Documents/Code/lucid
nohup .venv-convert/bin/python Tools/train_span.py \
  --corpus .build/corpus4 --bank-dir .build/bank \
  --channels 32 --steps 60000 --batch 32 \
  --resume Model/weights/span_ch32u.pth \
  > /tmp/train.log 2>&1 &
```

It restores the model, the optimiser state and the cosine LR schedule
position, so it continues rather than restarting the decay. ~12.5 it/s on
an M4 Pro, so the remaining 38000 steps are about 50 minutes.

Watch it with `grep "──" /tmp/train.log`.

## State on disk

| path | what it is |
|---|---|
| `Model/weights/span_ch32u.pth` | live checkpoint, step 22000 — resume from this |
| `Model/weights/span_ch32u_step22000.pth` | identical copy, in case the live one is overwritten by a bad run |
| `Model/weights/span_ch32u_animation_only.pth` | the control: trained on the animation-only corpus, step 24000 |
| `Model/weights/train-diversified.log` | this run's log so far |
| `.build/corpus4` | 1800 pairs, 64% live action / 35% animation, 6 tiers, 21 sources |
| `.build/bank` | 17800 pre-cut patches, content-gated |
| `.build/eval-corpus` | 120 disjoint pairs from crowd_run, shares no footage with training |

None of these are in git — `Model/` and `.build/` are ignored. They live only
on this machine.

## Where it was

| step | PSNR | fine |
|---|---|---|
| 14000 | 31.78 | 0.8783 |
| 18000 | 32.09 | 0.8822 |
| 20000 | 32.14 | 0.8840 |

Against the animation-only control at step 20000 (30.42 / 0.8437). Those two
numbers are **not** directly comparable — each model is validated on held-out
patches of its own corpus — which is what `.build/eval-corpus` exists to settle.

## What happens after 60000 steps

1. `Tools/convert_trained.py --weights Model/weights/span_ch32u.pth` — converts
   the whole ladder including 864×480.
2. `.build/modelbench Model/SPAN_x4_ch32u_<size>.mlpackage` for each — confirm
   the ladder still lands where the untrained timing said (12.6 ms at 640×360).
3. `Tools/eval_checkpoint.py --corpus .build/eval-corpus <both checkpoints>` —
   the honest comparison, on footage neither model has seen, against a Lanczos
   anchor.
4. `Tools/model_gallery.py` — visual comparison against the shipping ch28.
5. `Tools/ablate.py` — what each detail stage is worth on the new base, and
   whether cutting any of them buys the milliseconds that 480p needs.
