"""Prompts and the measurement codebook.

The codebook is the single most important text in the project: the same
definitions must be applied by the human labeller and by every model, or the
"error rate" measures disagreement about definitions rather than about imagery.
It is therefore written once, here, and injected into every prompt.
"""
from __future__ import annotations

CODEBOOK = """DEFINITIONS (apply exactly; these override your own intuitions):

ROOFTOP SOLAR
- Counts: photovoltaic panel arrays on building roofs, AND solar carports/
  canopies over parking lots or walkways on the campus. Both are "rooftop solar"
  for this task because both are campus-sited distributed PV.
- Does not count: solar water-heating panels, skylights, dark roof sections,
  HVAC units, ground-mounted utility arrays outside the campus boundary.
- Panels look like flat, uniform, dark blue-grey rectangles in regular rows,
  usually with visible seams between modules.

PORTABLE / MODULAR CLASSROOMS
- Counts: detached, single-storey, rectangular buildings of roughly 40-250 m2
  used for instruction, typically arranged in rows or clusters, often with
  external stairs/ramps and a uniform light-coloured roof.
- Does not count: permanent wings attached to the main building, storage sheds
  smaller than ~40 m2, dumpster enclosures, covered walkways, garages,
  greenhouses, restroom blocks.
- Count distinct BUILDINGS, not classroom units: a double-wide portable is one.

PERIMETER FENCING
- "full"    = essentially the whole campus boundary is fenced or walled (>80%).
- "partial" = some boundary segments fenced, others open (20-80%).
- "none"    = little or no boundary fencing (<20%).
- Judge the CAMPUS boundary, not fences around individual courts or pools.
- Type: chain-link, wrought iron, wall, other. Chain-link is often only visible
  as a thin line or its shadow. If you cannot tell the material, say null.

ATHLETIC FACILITIES
- running track: an oval running track (typically ~400 m, ~176 x 92 m).
  A straight sprint strip alone does not count.
- full-size sports fields: marked or clearly maintained fields of at least
  90 x 45 m (soccer/football/baseball outfield). Count distinct fields.
  Do not count a practice strip or an unmarked lawn.
- hard courts: paved courts marked for basketball, tennis, volleyball etc.
  Count distinct COURTS (a double basketball slab with two hoops = 2).
- pool: an outdoor swimming pool. Indoor pools are not visible; answer no
  unless you can see open water.

GENERAL
- Judge only the campus shown. Ignore adjacent private property.
- If the imagery is too obscured (cloud, tree canopy, low resolution) to judge a
  field, return null for that field rather than guessing."""

SCHEMA_HINT = """Return ONLY a JSON object with exactly these keys:
{
  "solar_present":  true | false | null,
  "solar_area_m2":  number | null,      // your best estimate of TOTAL panel area
  "portable_count": integer | null,
  "fencing_extent": "full" | "partial" | "none" | null,
  "fencing_type":   "chain-link" | "wrought iron" | "wall" | "other" | null,
  "track_present":  true | false | null,
  "field_count":    integer | null,
  "court_count":    integer | null,
  "pool_present":   true | false | null,
  "notes":          "one short sentence on anything ambiguous"
}
No markdown, no code fences, no commentary outside the JSON."""


def survey_prompt(school_name: str, level: str, enrollment: int) -> str:
    return f"""You are looking at overhead satellite imagery of a US public school campus.

School: {school_name}
Level: {level}
Enrollment: {enrollment}

The campus of interest is at the CENTRE of the image. Surrounding properties are
not part of it.

{CODEBOOK}

{SCHEMA_HINT}"""


def adjudicate_prompt(school_name: str, field: str, disagreement: str) -> str:
    """Used only where independent sources disagree -- the informative cases."""
    return f"""Overhead satellite imagery of {school_name}, a US public school campus.
The campus of interest is at the CENTRE of the image.

Independent automated methods disagree about ONE attribute: {field}.
{disagreement}

{CODEBOOK}

Judge {field} carefully and return ONLY a JSON object:
{{"{field}": <value>, "confidence": <0..1>, "reason": "<one short sentence>"}}"""


# Binary questions asked one at a time so a yes/no token logprob is meaningful.
# A logprob over "yes"/"no" is a real probability; a verbalised "0.9" is not.
BINARY_QUESTIONS = {
    "solar_present": "Are there photovoltaic solar panels on this campus (on roofs or as parking carports)?",
    "track_present": "Is there an oval running track on this campus?",
    "pool_present": "Is there an outdoor swimming pool on this campus?",
}


def binary_prompt(school_name: str, question: str) -> str:
    return f"""Overhead satellite imagery of {school_name}, a US public school campus,
centred on the campus.

{question}

Answer with exactly one word: yes or no."""
