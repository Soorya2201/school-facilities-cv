# School Facilities CV — Blended Pipeline (free APIs + open weights on CARC)

**Target:** 25 schools → campus resolution + 4 attributes + a *calibrated* confidence per field + 1-page memo.
**Compute:** free API tiers (no credit card, $0) **+** open weights on USC CARC (A100-80 / A40-48, SLURM).
**Money spent: $0.**

---

## 0. The organising principle: assign by comparative advantage

This is not "some of each to hedge." Each layer does what the other physically cannot.

| Capability | Free API | Open weights on your GPU |
|---|---|---|
| Frontier-grade semantic judgment ("portable or storage shed?") | Yes, **best** | No: a 32B model is weaker here |
| **Segmentation masks / per-pixel probabilities** | No, impossible | Yes, **only source** |
| **Detector logits** (OWLv2 / G-DINO scores) | No | Yes, **only source** |
| **High-volume TTA** (8 dihedral views × 25 × 6 = 1,200 passes) | No: blows the daily quota | Yes: free and unlimited |
| **Token logprobs** → `P("yes")` not "0.9" | Partly, *sometimes* — see §2c | Yes: always |
| Zero setup for a grader with no GPU | Yes | No |
| Instant iteration (no SLURM queue) | Yes, **decisive during the 5-hour box** | No: 20–40 min queue per test |
| Determinism / pinned versions | No: model silently updates | Yes |

**The split that falls out of this:**

> **Free APIs do judgment. Open weights do measurement. Vector data does geometry.**

Two consequences worth stating in the memo:

1. **Confidence comes from mechanism, not self-report.** Verbalized "confidence: 0.9" clusters at 0.9 and carries almost no information. Logits, mask probabilities, and ensemble variance are real signals — and open weights are what make them available.
2. **Overhead imagery has no canonical orientation.** That makes the 8 dihedral transforms (4 rotations × 2 flips) a *fully valid* test-time augmentation, unlike natural images where upside-down is out-of-distribution. Variance across the 8 views is free, principled epistemic uncertainty — and it needs the unlimited-volume layer.

---

## 1. Architecture

```
 L0  VECTOR (free, keyless, deterministic, instant)
     OSM/Overpass · Overture · MS Building Footprints
     → campus polygon, pitches, barriers, building geometry
                    │
 L1  MEASUREMENT — open weights, CARC GPU  (things no API can do)
     BDAPPV DeepLab ── solar mask + per-pixel probability → area m²
     OWLv2 + G-DINO ── detector logits
     SAM2.1 ────────── box → precise mask
     RemoteCLIP ────── zero-shot softmax
     Qwen3-VL-32B ──── token logprobs, via vLLM
     8× dihedral TTA ─ variance = epistemic uncertainty
                    │
 L2  JUDGMENT — free APIs  (low volume, high value, no GPU)
     Gemini 2.5 Flash ─────────── frontier reasoning, adjudication
     NVIDIA NIM llama-3.2-90b-v ─ 40 RPM, NO daily cap
     Groq / OpenRouter ────────── fallback + extra ensemble diversity
                    │
 L3  FUSION + CALIBRATION (CPU, seconds)
     agreement features → isotonic/Platt → LOSO CV
     → measurements.csv · reliability diagram · ECE/Brier · flag_for_review
```

**Escalation policy:** L0+L1 run on everything. **L2 is called only where L0/L1 disagree** — which is exactly where the information is. That keeps API volume at ~60–120 calls total, comfortably inside every free tier even with reruns.

---

## 2. The model stack (every entry verified live)

### 2a. Vector — L0 (free, keyless)

| Source | Verified this session |
|---|---|
| **Overpass / OSM** | Yes: returned campus polygon **48,076 m²**, 2× `leisure=pitch`, a `barrier=fence` way, 207 buildings incl. **6 at 128–139 m²** (portable-sized) |
| **Overture Maps** via DuckDB spatial | bbox-filtered remote GeoParquet — no bulk download |
| **MS US Building Footprints** | 129 M polygons, ODbL — rural gap-fill where OSM is empty |

### 2b. Open weights — L1

**The highest-value find:**

| Model | Size | Why |
|---|---|---|
| **`gabrielkasmi/bdappv-models`** → `deeplab_google_best.pth` | ~0.2 GB | DeepLabv3+ **already pretrained for rooftop PV segmentation** on BDAPPV (28k installations, 13k masks). The checkpoint trained on the **Google-imagery** subset — closest match to Esri World Imagery, so domain shift is small. **No training required.** Ships `deeplab_ign_best.pth` + `inception_google_best.pth` as a built-in second opinion. License: Google-subset is CC-BY-NC-4.0 — non-commercial research is fine, **state it in the README**. |

**Small specialists — under 10 GB combined, all fit on one A40:**

| Model | Size | Role |
|---|---|---|
| `google/owlv2-large-patch14-ensemble` | 3.5 GB | Open-vocab detection; pure `transformers`, **no CUDA extensions** |
| `IDEA-Research/grounding-dino-base` | 1.9 GB | Second detector. **Use the HF port, not the original repo** (CUDA-extension build trap) |
| `facebook/sam2.1-hiera-large` | 1.8 GB | Box → precise mask → accurate m² |
| `chendelong/RemoteCLIP` (ViT-L-14) | ~1 GB | RS-tuned CLIP; real softmax probability |
| `facebook/dinov3-vitl16-...` | ~1.1 GB | Features for retrieval-based confidence |

**Large VLM (bf16, downloads/mo from the HF API):**

| Model | Size | Fits | Note |
|---|---|---|---|
| **`Qwen/Qwen3-VL-32B-Instruct`** | 66.7 GB | 1×A100-80 | **Primary.** 768k dl/mo, strong grounding + counting |
| `OpenGVLab/InternVL3-38B` | 76.8 GB | 1×A100-80 | Different family — ensemble diversity |
| `allenai/Molmo-7B-D-0924` | 32.1 GB | any | **Pointing-supervised → counting-by-pointing beats counting-by-language.** Ideal for portables |
| *fast alternates* | | | `Qwen3-VL-30B-A3B` (MoE, 3B active), `Qwen3-VL-8B` (17.5 GB, 6.7M dl/mo) |

**Serving: vLLM** — required, because it exposes `logprobs`. That is what turns "yes" into `P(yes)=0.83`.

### 2c. Free APIs — L2 (verified; **quotas shift, re-check live**)

| Provider | Free limits | Vision model | Role |
|---|---|---|---|
| **Google AI Studio** | ~250–1,500 RPD (sources conflict — **check your live AI Studio quota**), no card | Gemini 2.5 Flash | **Primary adjudicator.** Best semantic judgment in the stack |
| **NVIDIA NIM** | **40 RPM, no daily cap**; catalog is publicly listable (84 models, verified keyless) | `meta/llama-3.2-90b-vision-instruct` | Ensemble member; **no daily cap makes it the volume-safe API** |
| **Groq** | 30 RPM / 14,400 RPD | vision-capable models | Fastest; fallback |
| **OpenRouter** | 35+ free models, one key | several | Router/fallback, extra diversity |

> **The bridge finding:** NIM, Groq, and OpenRouter serve **open-weight** models over **OpenAI-compatible** endpoints — which often expose **`logprobs`**. That means a grader with **no GPU at all** can still obtain calibration-grade probabilities. This is what lets the repo run everywhere without degrading the core claim. **Probe `logprobs` support at runtime and record which providers returned it** — do not assume.

### 2d. Explicitly rejected

- **Original GroundingDINO repo / Grounded-SAM-1** — CUDA-extension builds fail against mismatched `nvcc`/torch on shared clusters. The HF port is the same model without the pain.
- **Prithvi-EO / SatMAE / Scale-MAE** — pretrained at 10–30 m multispectral. We need **0.3 m RGB**. Wrong resolution regime; they would actively hurt.
- **GeoChat** — RS-specific but LLaVA-1.5-7B-era; Qwen3-VL-32B beats it. Optional 4th member only if time is free.

### 2e. Imagery (all keyless — verified)

| Source | Verified | Role |
|---|---|---|
| **Esri World Imagery** `/World_Imagery/MapServer/export` | Yes: arbitrary bbox → 1024×1024 JPEG, no key. Test pull resolved individual cars, rooftop HVAC, backyard pools | **Primary** |
| **Esri Wayback** | Yes, **196 dated releases, 2014-06 → 2026-08**, keyless | **Vintage + temporal bonus** |
| **NAIP** | Yes: keyless 0.6 m; catalog exposes exact `acquisition_date` (pulled e.g. `2023-05-27`). Caveat: `exportImage` 502'd once | 3rd view; date truth; **NIR → canopy occlusion** |

---

## 3. SLURM architecture

**Stage separation is mandatory.** CARC compute nodes have restricted egress; HF downloads inside a GPU job are the #1 way this dies at 2 a.m.

```
 Stage A  ── CPU node, internet, ~30 min, NO GPU ────────────
   fetch imagery (Esri/Wayback/NAIP) ─┐
   fetch OSM via Overpass             ├─▶ $SCRATCH/cache/
   PRE-DOWNLOAD ALL HF WEIGHTS ───────┘   $SCRATCH/hf/
   run L2 free-API judgment pass ─────────▶ preds/api_*.jsonl
   (APIs need internet — they belong HERE, not in the GPU job)

 Stage B  ── GPU node, HF_HUB_OFFLINE=1 ────────────────────
   array, one model per task:
     [0] Qwen3-VL-32B (a100:1) [1] InternVL3-38B (a100:1)
     [2] Molmo-7B-D   (a40:1)  [3] OWLv2+GDINO+SAM2+BDAPPV (a40:1)
   → preds/{model}.jsonl  (+ logprobs)
   Array isolates failures: one model OOMs, the rest still land.

 Stage C  ── CPU node, seconds ─────────────────────────────
   vector/CV ▸ fuse ▸ calibrate (LOSO CV) ▸ CSV + figures
```

```bash
#SBATCH --partition=gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=8 --mem=64G --time=04:00:00 --array=0-3
export HF_HOME=/scratch1/$USER/hf HF_HUB_OFFLINE=1
```

**Putting the API pass in Stage A is deliberate** — GPU nodes may not reach the internet at all, and you should never hold an A100 while waiting on a rate-limited HTTP call.

**CARC specifics that bite:** `$HOME` quota is small — weights and cache go in `/scratch1/$USER` (mind the ~30-day purge). Load `conda`/`cuda` modules explicitly; pin the torch/CUDA pair to the module tree.

> **Note: The grader has neither an A100 nor necessarily your API keys.** This is a live risk to criterion #1 ("we will check out your code and execute it"). **Mitigation: commit `preds/*.jsonl`.** Then `python -m src.run --offline` recomputes fusion → calibration → `measurements.csv` on any laptop in seconds, no GPU, no keys, no network. The GPU stage reproduces on CARC; the *result* reproduces anywhere. **Lead the README with this.**

---

## 4. Steps, with alternatives at each

### Step 1 — Campus resolution
The task warns CCD coordinates may be the mailing address or district office. This is upstream of everything: **if you don't know where the campus is, no field can be confident.**

- **Primary (L0):** Overpass `amenity=school` within ~800 m → fuzzy name match (`rapidfuzz`; normalize ELEM/Elementary, HS/High). *Verified: picks "Beverly Hills High School" over the adjacent "Good Shepherd Catholic School".*
- **Alt 1 (L0):** Overture via DuckDB spatial. **Alt 2 (L0):** MS footprint clustering. **Alt 3:** NCES EDGE geocodes — independent federal geocoding of the same school.
- **Alt 4 (L1):** RemoteCLIP / Qwen3-VL over a 3×3 tile grid → "is this a school campus?" → argmax tile.
- **Alt 5 (L2):** Gemini on the overview tile for the genuinely ambiguous ones.
- **Fallback ladder:** OSM polygon → building-cluster hull → **enrollment-scaled radius** (199-student middle school vs. 2,338-student HS need very different boxes) → CCD point + 300 m.
- **Emit `campus_confidence`; multiply every downstream field confidence by it.**
- **Detection:** area-per-student sanity (flag <5 or >300 m²/student); centroid↔CCD distance; multiple candidates in radius.

### Step 2 — Imagery
Per school: campus overview (polygon + 15% margin), 2× building crops at high zoom (solar), 4× perimeter crops (fencing), + oldest/newest Wayback.
- **Alternative:** stitch XYZ tiles at fixed z=19/20 → exact, auditable m/px. **Fallback:** Wayback → NAIP.
- **Foolproofing:** cache key `sha1(bbox|zoom|source|date)`; commit the cache.

> **Note: The silent killer: metres-per-pixel.** `m/px = 156543.03392·cos(lat)/2^zoom`, halved again for retina tiles. Wrong → every solar area off by 2× or 4× **with no error raised**. **Mitigation: known-size objects as an in-image scale bar** — track 400 m, tennis court 23.77×10.97 m, basketball 28×15 m. Assert within 10%, fail loudly. Also assert bbox aspect ratio matches pixel aspect (catches EPSG:3857↔4326 mix-ups).

### Step 3A — Rooftop solar (present/absent + area m²)
**Never ask a model for square metres.** Get a *mask*; convert with your own verified geometry.
- **Primary (L1):** BDAPPV DeepLab → per-pixel probability → threshold → area = pixels × m²/px. Mask probability doubles as confidence.
- **Alt 1 (L1):** OWLv2 "solar panel" → SAM2 refine → mask area. **Alt 2 (L1):** RemoteCLIP presence-only. **Alt 3 (L1):** classical OpenCV (low-saturation blue-grey + rectilinearity) — weak but independent.
- **Alt 4 (L2):** Gemini returns normalized polygons → you compute area.
- **Alt 5 (100-hr):** fine-tune SegFormer-B2 on **Duke/Bradbury** (19,433 polygons at 0.3 m, Fresno — **US-matched**, unlike BDAPPV's France/Belgium). ~1–3 h on one A100. This is the US domain-shift fix.
- **Failures:** dark/wet roofs, skylights, HVAC arrays, **solar carports over parking** (codebook decision), ground-mount arrays, panels installed after capture.
- **Detection:** flag panel area >60% of roof area; flag mask-vs-VLM extent disagreement.

### Step 3B — Portable / modular classrooms (count)
- **Primary (L0):** vector geometry on campus-clipped footprints — area 40–250 m², rectangularity >0.9, aspect 1.3–4.0, detached, row/grid. *Verified: 6 candidates at 128–139 m² on the Hollister campus.*
- **Alt 1 (L1):** **Molmo pointing count** — point, then count points.
- **Alt 2 (L1):** OWLv2 / G-DINO above a score threshold. **Alt 3 (L1):** SAM2 "segment everything" → geometric filter (works where no footprint data exists).
- **Alt 4 (L2):** Gemini adjudicates portable-vs-shed on disagreement — **this is exactly the judgment call the frontier model is worth spending a free call on.**
- **Failures:** **definitional ambiguity dominates** — portable vs. permanent modular wing vs. storage shed vs. dumpster enclosure vs. covered walkway; footprint staleness.
- **Mitigation:** codebook applied identically in labeling and prompting. Report ±1 as the honest resolution.

### Step 3C — Perimeter fencing (full/partial/none + type)
**Candour: the weakest field, for a physical reason.** Chain-link is ~2 mm wire at 0.3–0.6 m/px — **not resolvable from nadir at any model quality.** Visible instead: shadow lines, vegetation breaks, gates, walls.
- **Primary (L0):** OSM `barrier=fence|wall` ∩ campus perimeter → **fraction of perimeter covered** → full (>0.8) / partial (0.2–0.8) / none. `fence_type` gives material free. *Verified present on the test campus.*
- **Alt 1 (L1/L2):** VLM vote over 4 max-zoom perimeter crops. **Alt 2 (L1):** shadow-line detection — Hough lines parallel to boundary at consistent offset. **Alt 3:** Mapillary (free key, spotty). **Alt 4:** Street View — paid key; **skip and say so.**
- **The key asymmetry:** OSM **presence = high-precision positive evidence; OSM absence ≈ uninformative**, because fence-mapping recall is low and varies by urbanicity. Let calibration *learn* that recall rather than assuming it.
- **Fencing confidence should be systematically lowest of the four. Claiming 0.9 here is miscalibrated — saying so scores better than guessing well.**

### Step 3D — Athletic facilities (track y/n, fields, courts, pool y/n)
Most tractable — use it to show the calibration machinery working.
- **Primary (L0):** OSM `leisure=track|pitch|swimming_pool|sports_centre` + `sport=`, clipped. *Verified: 2 pitches with areas.*
- **Alt 1 (L1):** OWLv2/G-DINO open-vocab: "running track", "swimming pool", "tennis court", "baseball diamond".
- **Alt 2 (L1):** classical CV — Hough ellipse (track), HSV blue blob 100–1250 m² (pool), rectangle+surface colour (courts). Pools are the most detectable object in aerial imagery.
- **Failures:** practice vs. "full-size" (**define ≥90×45 m**); overlapping multi-use polygons double-counted; indoor/covered pools invisible; canopy-occluded courts; a `pitch` polygon that is really the whole complex (the 18,787 m² one needs an area cap).

---

## 5. Confidence & calibration — the centrepiece

**Features per (school, field):**

| # | Feature | Layer |
|---|---|---|
| 1 | VLM **token logprob** — `P("yes")` over the yes/no token set | L1 (+L2 where `logprobs` is exposed) |
| 2 | **Detector max score** (OWLv2 / G-DINO logits) | L1 |
| 3 | **Segmentation mean probability** over mask (BDAPPV) | L1 |
| 4 | **Dihedral TTA variance** (8 rot/flip views) | L1 |
| 5 | **Ensemble disagreement** across families incl. the free-API models | L1+L2 |
| 6 | **Cross-modality agreement** — vector vs. CV vs. neural | L0+L1 |
| 7 | `campus_confidence` | L0 |
| 8 | **Canopy fraction** from NAIP NIR (NDVI) — occlusion prior | L0 |
| 9 | **`n_sources_available`** | all |
| 10 | Field-type indicator (fencing gets its own intercept) | — |

**Map features → probability:** logistic (Platt) or isotonic regression on blind hand labels; temperature scaling for raw logit heads.

> **Why feature #9 matters for foolproofing:** free tiers can throttle or vanish mid-run. Because `n_sources_available` is a *feature*, losing a source makes confidence drop **honestly and automatically** rather than silently. The architecture degrades correctly by construction.

> **Note: Leakage.** Fitting calibration on the same 25 schools you report accuracy on inflates every number. **Fit and evaluate under leave-one-school-out CV, and say so.** This single point will separate the submission from the field.

**Report:** reliability diagram (5 bins), **ECE**, **Brier**, and a **coverage–accuracy curve** — *"at 100% coverage accuracy is X; flagging the lowest-confidence 30% raises accuracy on the rest to Y."* That answers the prompt's own framing in its own words. Emit `flag_for_review`. **Abstention is a feature.**

---

## 6. Ground truth (order matters)

1. Cache imagery for all 25.
2. **Write the codebook first** — portable, "full-size", court, carport.
3. **Hand-label all 25 BEFORE running any model** — small local HTML page (image + form → JSON). Blind labeling prevents anchoring, and you can say so.
4. **Re-label 5 schools hours later** → intra-rater agreement.

Step 4 is the highest-value 20 minutes in the project. It licenses the memo line:

> *"My own labels are only ~92% self-consistent on portables, so a measured accuracy above that is inside my own measurement error."*

Classical attenuation/measurement-error framing, in an economist's native vocabulary — it converts a limitation into evidence of rigour.

- **Alternative:** second labeler on 10 schools → inter-rater κ.
- **Do NOT** use OSM as pseudo-truth for athletics — it is an *input*, so that is circular.
- Report per-field accuracy with **Wilson intervals**; n=25 gives wide bands, and saying so is candour.

---

## 7. Temporal bonus (nearly free)

Wayback's 196 dated releases mean two-date measurement costs one extra fetch per school and covers **all 25**, not "a handful."
- **Primary:** BDAPPV DeepLab on oldest + newest usable release → solar change with explicit dates. Solar is the right attribute: it genuinely changes on a 5-year horizon and it is what the eventual project cares about.
- **Alternative:** NAIP multi-year via Google Earth Engine (free for research; exact acquisition dates).
- **Foolproofing:** a Wayback release date is *publication*, not capture. Say so; cross-check against NAIP `acquisition_date` where they overlap.

---

## 8. Cost at scale (130,000 schools)

Instrument the run — log GPU-seconds, API calls, and wall-clock per school; extrapolate rather than assert.

- **L0 vector:** free, milliseconds. **L1 GPU:** ~200–600 A100-hours for 130k → **$0 on CARC allocation** (~$200–1,200 cloud-equivalent).
- **L2 API:** at a ~40% escalation rate, ~104k calls — **this is the layer that does *not* scale on free tiers.** At 250–1,500 RPD you would need years. At national scale L2 must either become paid (a few hundred to ~$2k depending on model) or be replaced by L1, which is exactly why the heavy lifting lives in L1. **Say this explicitly — it is the honest scaling answer and it shows you understand where your own design breaks.**

**The judgment that matters — inference is not the binding constraint.** Two things dominate:
1. **Imagery rights and throughput.** Keyless Esri endpoints are fine for 25 schools and inappropriate for 130,000. At national scale: **NAIP bulk (public domain, on AWS)** or a commercial contract.
2. **Human review of the flagged tail.** At a 30% flag rate: 39,000 × ~2 min ≈ **1,300 hours ≈ $26k** — an order of magnitude above compute.

**End the memo on this: cutting the flag rate is worth far more than cutting compute cost by 10×.** That is what makes calibration an economic argument, not merely a statistical one.

---

## 9. Failure register

**Compute / HPC**
| # | Risk | Mitigation |
|---|---|---|
| 1 | **HF download inside a GPU job** (restricted egress) | Stage A pre-stages weights; Stage B `HF_HUB_OFFLINE=1` |
| 2 | **API call inside a GPU job** | L2 runs in Stage A — never hold an A100 on an HTTP wait |
| 3 | `$HOME` quota blowout | `HF_HOME=/scratch1/$USER/hf`; mind the ~30-day purge |
| 4 | OOM on 32B/38B | bf16 + `--max-model-len` cap; fall back to Qwen3-VL-8B / 30B-A3B MoE |
| 5 | CUDA/torch mismatch vs. module tree | Pin the pair; build the env once in Stage A |
| 6 | Original G-DINO CUDA extension fails | **Use the HF `transformers` port** |
| 7 | GPU queue exceeds the time box | Array by model; task 3 (small specialists) alone yields a complete pipeline |
| 8 | Walltime hit | Checkpoint per school; resume skips completed ids |
| 9 | **Grader has no GPU and no keys** | **Commit `preds/*.jsonl`; `--offline` reproduces the CSV** |
| 10 | **Free tier throttles / quota exhausted mid-run** | Provider fallback chain (Gemini→NIM→Groq→OpenRouter); `n_sources_available` makes confidence degrade honestly |
| 11 | **API model silently updated → non-reproducible** | Pin model strings, cache every response by content hash, log the served model id |
| 12 | `logprobs` unsupported by a provider | **Probe at runtime, record support, fall back to agreement-only features** |
| 13 | Overpass 429 (2 slots/IP) | Cache, backoff, one batched query per school, mirror fallback |
| 14 | Esri throttling / NAIP 502 (**observed**) | Cache + commit; NAIP never load-bearing |
| 15 | Nondeterminism | `temperature=0`, fixed seeds, `VLLM_SEED`; cache by content hash |

**Method / data**
| # | Risk | Detection |
|---|---|---|
| 16 | **m/px scale error (silent, 2–4×)** | Known-object scale-bar assertion |
| 17 | EPSG:3857 ↔ 4326 confusion | Aspect-ratio assertion |
| 18 | Wrong campus → neighbour's pool counted | Area-per-student sanity; name match score |
| 19 | Shared campuses (`HILLDALE LOWER ES`) | Multiple-candidate flag |
| 20 | Urban rooftop schools (`PS 78`, Bronx) | Explicit **NA ≠ 0** semantics |
| 21 | Tree canopy occlusion | NAIP NIR → NDVI → confidence penalty |
| 22 | **BDAPPV domain shift** (France/Belgium → US) | Validate on US hand labels; optionally fine-tune on Duke/Fresno |
| 23 | Imagery date ≠ 2024 enrolment | Report `imagery_date` per row |
| 24 | Portable definitional ambiguity | Codebook written before labeling |
| 25 | OSM staleness / uneven recall | Presence strong, absence weak; learn recall by urbanicity |
| 26 | Calibration leakage | Leave-one-school-out CV |
| 27 | Labeling anchoring bias | Label before any model output is visible |

---

## 10. Deliverables & verification

- `measurements.csv` — one row per `school_id`; every field + `*_confidence`; `imagery_date`, `imagery_source`, `campus_method`, `campus_confidence`, `flag_for_review`, `solar_*_t0/_t1`.
- `README.md` — **lead with the zero-GPU, zero-key offline path**; then free-API path; then full SLURM path.
- `memo.pdf` — one page: Validation / Failures / Cost at scale / Next / Tooling. **Name failed schools by name.** Tooling section must state which layer used what.

**Verify before submitting:**
1. Fresh clone, **no GPU, no keys, no network**: `python -m src.run --offline` reproduces `measurements.csv` byte-identically. *Criterion #1 — actually test it.*
2. Scale-bar assertion passes on every school with a detected track.
3. `metrics.json` has ECE, Brier, per-field accuracy + Wilson intervals, coverage–accuracy curve — all under LOSO CV.
4. All 25 rows present, with explicit `NA` + reason where measurement failed. **No silent blanks.**
5. `sbatch` runs end-to-end on CARC from a clean `$SCRATCH`.

---

## 11. Open decisions

1. **Ensemble size.** Qwen3-VL-32B + InternVL3-38B + Molmo-7B-D + Gemini is the diversity sweet spot. If the queue is tight: Qwen3-VL-8B + Molmo-7B-D + small specialists on one A40, plus the free APIs, still gives genuine cross-family disagreement.
2. **Fine-tune on Duke/Fresno?** BDAPPV is France/Belgium. A 1–3 h A100 fine-tune fixes the US domain shift — worth it only if GPU time is free and the 5-hour *human* box is respected.
3. **Solar carports over parking** — count as rooftop solar or not? Affects several CA schools. Codebook line needed either way.
4. **Second labeler** for 10 schools → upgrades intra-rater consistency to inter-rater κ.

> **On the time box:** the task says it "will not reward work that clearly went well beyond" five hours. GPU hours and API calls are not human hours — a SLURM job running overnight costs nothing in the budget being judged. Keep *your* hands-on time near the box, let the cluster and the free tiers do the rest, and **report both numbers honestly in the memo.**

---

## 12. Implementation findings (measured during the build, not predicted)

These changed the design. Each was verified by running the thing, not by reasoning about it.

| # | Finding | Consequence |
|---|---|---|
| 1 | **Overpass returns HTTP 406 for the default `python-requests` User-Agent.** With mirror fallback in place this presents as a network hang, not an error. | Explicit `User-Agent`; mirrors tried in order with fast failure. |
| 2 | **A regional Overpass mirror answers a US query with HTTP 200 and an empty result.** `overpass.osm.ch` returned 272 bytes where the main instance returned 484 KB. Accepting it would have silently meant "no buildings, no fences". | `_looks_like_full_planet()` validates the response *shape*, not just the status code. |
| 3 | **Esri World Imagery enforces a per-location resolution ceiling** and returns a deterministic HTTP 500 below it (4/4 retries). Measured: a 200 m extent fails at 1024 px (0.195 m/px) in rural Alabama but succeeds at 768 px (0.26 m/px); the same request succeeds at 1024 px in Los Angeles. | A resolution ladder that steps the *pixel count* down while holding the ground extent fixed, so the campus stays framed. `m_per_px` became a calibration feature. |
| 4 | **A non-square campus rendered into a square image gives different m/px per axis.** A 317 × 277 m campus tripped the aspect assertion. | `square_bbox()` at fetch time; the assertion stays as a guard. |
| 5 | **Colour alone produces confident false positives.** A dark navy play structure scored as a pool; baseball-infield dirt scored as a running track at 0.99. | Pools need a high value floor (water is *bright* cyan); tracks need an **annulus** test — a hole plus a band ratio of 0.10–0.60 — not a coloured blob. Both false positives eliminated. |
| 6 | **`building=house` inside a generous campus polygon has exactly the footprint of a portable.** | Residential tags are decisive negatives in the portable filter. |
| 7 | **OSM recall is the real limit, not precision.** At Cerra Vista OSM reports 0 portables where the imagery plainly shows a row; at Wells Middle a `leisure=swimming_pool` tag fires where no pool is visible. | Confirms the presence-strong / absence-weak asymmetry, and that L1 vision is needed to fill gaps rather than merely adjudicate. |
| 8 | **The CPU colour heuristic badly under-measures solar.** Wells Middle has extensive carport arrays; the heuristic scored 132 m². | Justifies BDAPPV on GPU for area. The CV detector stays as one weak, independent vote. |
| 9 | **10 of 23 two-date measurements are physically implausible** (large *decreases* in installed PV). | Added a direction-aware plausibility flag: installed PV is essentially never removed, so a >50% loss is detector noise, not decommissioning. Reporting it unflagged would present an artefact as a finding. |
| 10 | **BDAPPV cross-provider IoU is 0.561 vs 0.884 same-provider**, and it trains at 400 px on ~0.1 m/px. Our imagery is Esri — a *third* provider — often 2–3× coarser. | Solar crops fix the ground extent (120 m) and take the best resolution available, rather than demanding an unattainable m/px. Its confidence is re-calibrated locally, never inherited from the model card. |

### Verified working, end to end

- **Offline reproduction is byte-identical in 0.94 s** with no network, no keys and no GPU — evaluation criterion #1.
- 25/25 campuses resolved: 18 by OSM name match, 4 by nearest-school, 3 by building-cluster fallback.
- 257 images cached (55 MB), every row carrying a real imagery date.
- 23/25 schools have two dated observations (2014-02-20 → 2026-08-05).
- 18/18 invariant tests pass, including the scale-bar check that catches a doubled m/px.
- Calibration verified end to end on labelled input: LOSO CV → ECE, Brier, reliability diagram, per-field Wilson intervals, coverage–accuracy curve.

### Still required (needs the user, or a GPU)

1. **Hand-label all 25 schools** via `python -m src.run label` — blind, before reading any output. Nothing in the validation section can be filled in until this exists.
2. **Run Stage B on CARC** (`sbatch slurm/stage_b.sh`) to add BDAPPV, OWLv2 and Qwen3-VL. Currently only 2 of 6 sources are active, so agreement features are thin and `fencing_type` is unmeasured for all 25.
3. **Fill `memo_template.md`** from `out/metrics.json`.
