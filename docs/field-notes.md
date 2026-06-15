# Project Halide Field Notes

Project Halide is an edge-native diagnostic workbench for analog film scans. It
combines a fine-tuned MiniCPM-V 4.6 vision extractor with Nemotron-Mini-4B
diagnostic reasoning, then presents the result in a Gradio light-table
interface.

## Design Goal

The goal is not to classify a photograph as good or bad. The goal is to inspect
the scan as physical evidence: surface scratches, dust, dirt, lifted emulsion,
chemical stains, light leaks, and scanner artifacts. Film metadata is useful,
but it is treated as lower-priority context unless the user marks it as
verified.

## What Changed During Evaluation

The first working pipelines could detect obvious dust and scratches, but they
were unreliable on real analog negatives. In particular, broad transparent crack
networks over a portrait were sometimes returned as `{"defects": []}`. Visual
inspection showed that the model could recognize the same damage on crops, so
the failure was mainly scale and attention, not a complete lack of learned
semantics.

The final runtime therefore uses a vision strategy with model output plus a
small validator assist:

1. Run full-frame MiniCPM-V extraction.
2. If a large scan returns too few validated defects, run overlapping 960 px
   tiles.
3. Remap tile boxes back to full-image coordinates.
4. Validate, confidence-filter, and remove repeated sprocket or frame-edge
   artifacts.
5. Add conservative image-analysis scratch candidates when clear bright linear
   evidence is visible.
6. Dedupe the combined boxes.

This keeps clean scans inexpensive while recovering small or transparent damage
that disappears in a full-frame view.

## Real Web-Sourced Negative Check

Additional public examples were collected from photography forums and
restoration references, including a real color negative strip with emulsion
dirt, annotated Photrio negative strips, static-discharge examples, and damaged
positive scans from Analog.Cafe. These were kept in `.nottracked` for
inspection, not committed as public assets.

The check exposed two useful limits:

1. Raw strip photos can be harder than flatbed scan frames because sprocket
   holes and film borders look like repeated dust boxes.
2. Positive scans with obvious scratches can still confuse the vision model
   when subject content, sky gradients, or aircraft silhouettes dominate the
   frame.

The validator changes specifically target those failures by filtering repeated
edge artifacts and by preserving high-precision scratch evidence that is
visible in local contrast.

## Final Held-Out Smoke Test

The five private negatives supplied for evaluation stayed in `.nottracked` and
were not used for training. The final v7 checkpoint with tiled fallback produced
the following result:

| Image | Expected condition | Output summary |
| --- | --- | --- |
| `negative1.png` | Long scratches across a portrait | 8 defects, including scratch and emulsion damage |
| `negative2.png` | Abraded emulsion and dirt patches | 9 defects, including scratch and emulsion damage |
| `negative3.png` | Severe emulsion damage and debris | 6 defects, including scratch and emulsion damage |
| `negative4.png` | Near-clean hard negative | 0 defects |
| `negative5.png` | Broad lifted crack network over a portrait | 45 defects, including 17 scratch and 14 emulsion damage |

The key regression test is `negative5.png`: full-frame inference produced zero
validated defects, while tiled inference recovered the visible crack network.

## Public Demo Run

The public demo uses a real 35mm negative strip with residue, glare, sprocket
holes, and stain-like damage. The final launch video shows the live Space flow:
upload, metadata, evidence extraction, validated overlay, diagnostic report,
and history recall.

Demo video:
<https://youtube.com/watch?si=apzCiBZcIZWC1nFt&v=DGJ2M1aQCrE&feature=youtu.be>

Technical blog:
<https://lonelyguy.vercel.app/articles/2026-06-16-project-halide>

## Current Limits

Halide is an inspection aid, not an archival authority. It can over-box broad
damage regions, and it may assign multiple labels to the same physical
artifact when a crack also looks like chemical haze or peeled emulsion. The UI
therefore shows the overlay, evidence counts, metadata confidence, raw JSON, and
diagnosis history so the user can judge the evidence directly.

## Reproducibility

The exact private evaluation command used for the final smoke test was:

```bash
python scripts/run_private_negative_eval.py --image .nottracked/negative1.png --image .nottracked/negative2.png --image .nottracked/negative3.png --image .nottracked/negative4.png --image .nottracked/negative5.png --model /checkpoints/minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625 --out-dir .nottracked/v7_ckpt625_tiled960_private_eval_exact5
```

The output directory contains raw JSON and overlay images for local inspection.

The merged checkpoint was also converted on Modal to a llama.cpp Q4_K_M GGUF
artifact and uploaded to the model repository:

```text
minicpm-v-4.6-merged-v7-crack-curriculum-r1-ckpt625-q4_k_m.gguf
```
