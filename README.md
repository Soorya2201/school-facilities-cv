# School Facilities from Imagery

Measures four attributes for 25 US public schools from overhead imagery, with a
**calibrated confidence on every field**, and reports how well those confidences
track the actual error rate.

Everything is free and keyless. No paid API is used anywhere in the pipeline.

---

## Quick start — reproduce the results (no GPU, no keys, no network)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run --offline stage-c
```

This reads the committed `cache/` and `preds/` and regenerates
`out/measurements.csv`, `out/metrics.json` and `out/reliability.png`.
It needs no API key, no GPU and no network access, and takes a few seconds.

**This is the command to run first.** Everything below reproduces the inputs it
consumes.

---

## On USC CARC

```bash
git clone https://github.com/Soorya2201/school-facilities-cv.git
cd school-facilities-cv
bash slurm/setup_carc.sh          # conda env + deps + verifies the offline path
source .env.carc                  # in every new shell afterwards
```

`setup_carc.sh` puts the HuggingFace cache and the imagery cache under
`/scratch1/$USER` rather than `$HOME`, because the home quota is small and the
model weights are tens of gigabytes. It finishes by running the offline
reproduction and the test suite, so a successful run means the checkout is good.

Then either stage separately, or chain all three:

```bash
sbatch slurm/stage_a.sh           # CPU + internet: imagery, OSM, weight staging
sbatch slurm/stage_b.sh           # GPU, offline: array over model families
sbatch slurm/stage_c.sh           # CPU: fuse, calibrate, write outputs
# or
bash slurm/submit_all.sh          # all three with dependencies
```

---

## Full pipeline

### Stage A — fetch and deterministic extraction (CPU, needs internet)

```bash
python -m src.run stage-a
```

Resolves each campus, fetches and caches imagery, and runs the two
deterministic extractors (OSM vector geometry, classical CV). Roughly 45 s per
school, almost all of it network wait.

Optional free-tier API keys. The pipeline runs without any of them; each one
present adds an independent opinion and therefore a better-informed confidence:

```bash
export GEMINI_API_KEY=...      # ai.google.dev            (best judgment)
export NVIDIA_API_KEY=...      # build.nvidia.com         (40 RPM, no daily cap)
export GROQ_API_KEY=...        # console.groq.com
export OPENROUTER_API_KEY=...  # openrouter.ai
```

### Stage B — open-weight models (GPU)

```bash
sbatch slurm/stage_b.sh          # on CARC
# or locally, if you have a GPU:
python -m src.tier1_gpu --task solar --device cuda
python -m src.tier1_gpu --task owl   --device cuda
```

### Stage C — fuse, calibrate, write outputs

```bash
python -m src.run stage-c
```

### On CARC, all three chained

```bash
bash slurm/submit_all.sh
```

---

## Ground truth

Labels are produced **before** any model output is visible, so they cannot be
anchored to what the pipeline predicted:

```bash
python -m src.run label                 # builds labels/label_tool.html
# label all 25 in the browser, then "download truth.csv" -> labels/truth.csv
python -m src.run stage-c

# later, re-label a few schools into labels/truth_relabel.csv and run:
python -m src.run label --agreement     # intra-rater agreement
```

The re-labelling step measures how self-consistent the ground truth itself is.
That sets a ceiling: the pipeline cannot be shown to be more accurate than the
labels are reproducible.

---

## Design

**Free APIs do judgment. Open weights do measurement. Vector data does geometry.**

| Layer | What it is | What only it can do |
|---|---|---|
| **L0 vector** | OpenStreetMap via Overpass | campus polygon, building geometry, tagged pitches/pools/barriers — deterministic and instant |
| **L0 CV** | OpenCV colour/shape | an opinion whose failure modes are uncorrelated with a neural network's |
| **L1 GPU** | BDAPPV DeepLabV3, OWLv2, SAM2, Qwen3-VL via vLLM | segmentation masks, detector logits, token logprobs, 8× test-time augmentation |
| **L2 API** | Gemini / NIM / Groq / OpenRouter free tiers | frontier semantic judgment on the ambiguous cases |

L0 and L1 run on everything. **L2 is called only where the other layers
disagree** — which is where the information is, and which keeps API volume
inside every free tier.

### Confidence

Confidence is never a number a model asserts. Verbalised confidences cluster at
0.9 and carry little information. Instead it is a **learned function of how much
independent evidence agrees**, fitted on the blind hand labels from 14 features:
cross-source agreement, detector logits, segmentation probabilities, TTA
variance, token logprob margin, campus-resolution confidence, imagery
resolution, and OSM presence/absence.

Two properties worth noting:

- **No leakage.** The calibrator is fitted and evaluated under
  **leave-one-school-out cross-validation**, grouped by school rather than by
  observation, because fields within one school share a campus polygon and an
  image. Every number in `metrics.json` is out-of-fold.
- **Honest degradation.** `n_sources` is itself a feature, so if a free tier
  throttles or a GPU job fails, confidence falls automatically rather than the
  pipeline silently pretending it still had the same evidence.

`out/metrics.json` reports ECE, Brier score, a reliability diagram, per-field
accuracy with Wilson intervals, and a coverage–accuracy curve.

---

## Layout

```
src/
  config.py      every threshold that affects a measurement, in one place
  geo.py         projections, areas, and the metres-per-pixel conversion
  cache.py       content-addressed disk cache (committed, so runs reproduce)
  osm.py         Overpass client with mirror fallback and response validation
  resolve.py     campus resolution ladder + campus_confidence
  imagery.py     Esri World Imagery, Esri Wayback, NAIP
  prompts.py     the measurement codebook, shared by labeller and models
  tier0_vector.py  measurement from OSM geometry
  tier0_cv.py    classical CV detectors
  tier1_gpu.py   open-weight models (CARC)
  tier2_api.py   free-tier hosted models
  fuse.py        signal combination + calibration features
  calibrate.py   LOSO calibration, ECE/Brier/reliability/coverage
  labels.py      blind labelling tool + intra-rater agreement
  report.py      measurements.csv, metrics.json, reliability.png
  run.py         CLI
slurm/           stage_a.sh, stage_b.sh, stage_c.sh, submit_all.sh
cache/           imagery + API responses (committed — this is what makes it reproduce)
preds/           per-source raw predictions (committed)
```

---

## Data sources

| Source | Licence / terms | Used for |
|---|---|---|
| Esri World Imagery | Esri terms; keyless `export` endpoint | primary imagery |
| Esri Wayback | as above; 196 dated releases 2014–2026 | imagery vintage, two-date measurement |
| USGS NAIP | public domain | third view, exact acquisition dates, NIR |
| OpenStreetMap | ODbL | campus polygons, buildings, pitches, barriers |
| BDAPPV models | Google-subset checkpoints derive from CC-BY-NC-4.0 imagery | rooftop PV segmentation (non-commercial research use) |

---

## Known limitations

These are measured, not hypothetical — see `memo.pdf` for the numbers.

- **Perimeter fencing is the weakest field, for a physical reason.** Chain-link
  is ~2 mm wire at 0.3–0.6 m/px: it is below the resolution limit of nadir
  imagery at any model quality. OSM is the only source that sees fence
  *material*, and its fence-mapping recall is low. Fencing confidence is
  systematically the lowest of the four fields, and that is correct.
- **Imagery resolution varies by location.** Esri enforces a per-location
  resolution ceiling and hard-fails below it: measured 0.26 m/px in rural
  Alabama versus 0.156 m/px in Los Angeles. The pipeline steps down its
  requested pixel count until the request succeeds, and records what it got.
- **The solar segmenter is out of distribution.** BDAPPV reports IoU 0.884
  training and testing on Google imagery, but 0.561 across providers — and ours
  is Esri, a third provider, often at 2–3× coarser resolution than its 0.1 m/px
  training data. Its confidence is re-calibrated against our own labels rather
  than inherited from the model card.
- **OSM absence is not absence.** A missing tag mostly means nobody mapped it.
  Presence is treated as strong evidence and absence as weak, and the
  calibrator learns how weak.
