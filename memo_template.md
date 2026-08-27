# Measuring School Facilities from Imagery — Method Memo

*Soorya Vijayakumar · USC Dornsife School Facilities CV take-home*

**Fill the bracketed numbers from `out/metrics.json` after labelling. Everything
unbracketed is already established.**

## Validation

Ground truth is my own hand labelling of all 25 schools, done **before any model
output existed**, using the codebook in `src/prompts.py` — the same text injected
into every model prompt, so measured error is disagreement about imagery rather
than about definitions. I then re-labelled [N] schools [X] days later: my own
labels agree with themselves [XX]% overall and [XX]% on portable counts. That is
the ceiling on any accuracy claim here, and pipeline accuracy above it would not
be meaningful.

Confidence is not a model's self-report. It is a logistic map from 14 agreement
features — cross-source agreement, OWLv2 detector logits, BDAPPV segmentation
probabilities, 8-view test-time-augmentation variance, yes/no token logprob
margin, campus-resolution confidence, and achieved imagery resolution — fitted on
those labels. **Every reported number is out-of-fold under leave-one-school-out
cross-validation, grouped by school**, because fields within a school share a
campus polygon and an image and are not independent.

Overall accuracy [XX]% (n=[N] scoreable field-observations across 25 schools),
ECE [X.XX], Brier [X.XX]. Coverage–accuracy: flagging the least-confident 30%
raises accuracy on the remainder from [XX]% to [XX]%. Per-field accuracy with
Wilson intervals is in `metrics.json`; with n=25 those intervals are wide, which
is itself worth stating.

## Failures

- **Perimeter fencing is the weakest field, and for a physical reason rather
  than a modelling one.** Chain-link is ~2 mm wire at 0.3–0.6 m/px — below the
  resolution limit of nadir imagery at any model quality. OSM is the only source
  that observes fence *material*, and its fence-mapping recall is low and
  uneven. Fencing confidence is systematically the lowest of the four fields.
- **Campus extent, not measurement, is the dominant upstream failure.** The CCD
  coordinate is sometimes hundreds of metres off. [N] of 25 schools had no OSM
  school polygon and fell back to a building-cluster hull or a radius disc; those
  carry `campus_confidence` 0.30–0.45, which multiplies into every field. Named
  failures: [Wedowee Middle (308 m² hull), Spring Lake Heights (804 m²/student),
  Adena Elementary (777 m²/student), Coleman HS (871 m²/student)].
- **Imagery resolution varies by location and is capped.** Esri enforces a
  per-location resolution ceiling and returns HTTP 500 below it — measured
  0.26 m/px in rural Alabama versus 0.156 m/px in Los Angeles. The pipeline steps
  its requested pixel count down until the request succeeds and records what it
  got; `m_per_px` is a calibration feature because coarse pixels genuinely
  predict error.
- **The solar segmenter is out of distribution.** BDAPPV reports IoU 0.884
  within Google imagery but 0.561 across providers; ours is Esri — a third
  provider — often 2–3× coarser than its 0.1 m/px training data. Its confidence
  is re-calibrated against my labels rather than inherited from the model card.
- **OSM absence is not absence.** A missing tag usually means nobody mapped it.
  Presence is treated as strong evidence, absence as weak, and the calibrator
  learns how weak. [At Cerra Vista, OSM reports 0 portables where the imagery
  plainly shows a row of them.]

## Cost at scale

Measured: [XX] s wall-clock per school in Stage A, almost entirely network wait,
and [XX] GPU-seconds per school in Stage B. Extrapolating to ~130,000 schools:
[XXX]–[XXX] A100-hours, which is **$0 marginal on a CARC allocation** and roughly
$200–1,200 at cloud rates. Vector extraction is free and effectively instant.

Two honest caveats about my own design. First, **the free-tier API layer does not
scale** — at ~40% escalation that is ~104,000 calls against quotas of a few
hundred per day. At national scale it must become paid (a few hundred to ~$2k) or
be replaced by the GPU layer, which is why the heavy lifting already lives there.
Second, **keyless Esri endpoints are appropriate for 25 schools and not for
130,000**; that job needs NAIP bulk from AWS (public domain) or a commercial
imagery contract.

Neither is the binding constraint. **Human review of the flagged tail is.** At a
30% flag rate that is 39,000 schools × ~2 min ≈ 1,300 hours ≈ $26k — an order of
magnitude above compute. So reducing the flag rate is worth far more than
reducing compute cost tenfold, which is what makes calibration an economic
argument rather than only a statistical one.

## Next (100 hours instead of 5)

1. **Fine-tune the solar segmenter on US imagery.** Duke/Bradbury has 19,433
   annotated PV polygons at 0.3 m/px over Fresno — resolution- and
   country-matched, unlike BDAPPV. A 1–3 h A100 fine-tune should close most of
   the 0.884→0.561 cross-provider gap.
2. **Fix campus extent properly**, since it gates everything else: NCES EDGE
   geocodes plus Microsoft Building Footprints plus parcel data, with a learned
   campus-boundary model rather than a hull.
3. **Get fencing from the ground.** Mapillary street-level imagery where it
   exists is the only way to see chain-link; nadir imagery cannot.
4. **Expand ground truth and make it inter-rater.** n=25 gives Wilson intervals
   ±0.15; a second labeller on a few hundred schools would give both tighter
   bounds and a κ rather than a self-agreement figure.
5. **Active learning on the flagged tail** — label exactly the cases the
   calibrator is least sure about, which is the cheapest route to lowering the
   flag rate and therefore the dominant cost at scale.

## Tooling

- **Imagery** (all keyless, free): Esri World Imagery; Esri Wayback (196 dated
  releases 2014–2026) for vintage and the two-date measurement; USGS NAIP for a
  third view and exact acquisition dates.
- **Vector**: OpenStreetMap via Overpass — campus polygons, building footprints,
  `leisure=*`, `barrier=*`.
- **Open weights**: `gabrielkasmi/bdappv-models` DeepLabV3-ResNet101 for PV
  segmentation; OWLv2-large for open-vocabulary detection; SAM 2.1 for mask
  refinement; Qwen3-VL-32B served by vLLM (chosen specifically because it
  returns token logprobs). Run on USC CARC A100/A40 via SLURM.
- **Free-tier APIs**: Gemini 2.5 Flash for adjudication; NVIDIA NIM
  (`llama-3.2-90b-vision`), Groq, OpenRouter as fallbacks.
- **Assistant**: Claude Code (Opus 5) for the implementation throughout —
  pipeline structure, the geometry and calibration modules, and debugging. Every
  external claim in this memo was verified by running the endpoint or reading the
  model card, not taken from the assistant's recall.
- **No paid API was used. Total spend: $0.**
