"""Blind hand-labelling: ground truth, and a small tool to produce it.

Protocol, and the order is the point:

  1. cache imagery for all 25 schools
  2. write the codebook (src/prompts.py) BEFORE looking at anything
  3. label all 25 by hand with NO model output visible -- blind labelling, so
     the labels cannot be anchored to what the pipeline predicted
  4. re-label a handful of schools later to measure intra-rater agreement

Step 4 is what licenses an honest statement about the ceiling: if my own labels
only agree with themselves ~92% of the time on portables, then a measured
pipeline accuracy above that is inside my own measurement error and should not
be reported as if it were real.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

TRUTH_CSV = config.LABELS / "truth.csv"
RELABEL_CSV = config.LABELS / "truth_relabel.csv"

LABEL_FIELDS = [
    "solar_present", "solar_area_m2", "portable_count",
    "fencing_extent", "fencing_type", "track_present",
    "field_count", "court_count", "pool_present", "label_notes",
]

_BOOL = {"true": True, "yes": True, "1": True, "y": True,
         "false": False, "no": False, "0": False, "n": False}

_MISSING = ("na", "n/a", "null", "unknown")
# "none" is a real answer for fencing extent -- the codebook's three categories
# are full / partial / none -- so it is a missing-value sentinel everywhere
# except here. Treating it as missing silently discards every unfenced campus,
# which is most of them, and shrinks the field's n to the handful that are
# fenced.
_NONE_IS_A_VALUE = {"fencing_extent"}


def coerce_field(field: str, raw: str) -> Any:
    """Parse one CSV cell into the type the pipeline compares on.

    Shared with `compare`, which reads predictions back out of measurements.csv:
    a binary field round-trips as the string "False", and every non-empty string
    is truthy, so scoring without this returns nonsense.
    """
    raw = (raw or "").strip()
    low = raw.lower()
    if raw == "" or low in _MISSING or (low == "none" and field not in _NONE_IS_A_VALUE):
        return None
    if field in config.BINARY_FIELDS:
        return _BOOL.get(raw.lower())
    if field in config.COUNT_FIELDS:
        try:
            return int(float(raw))
        except ValueError:
            return None
    if field == "solar_area_m2":
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


def load_truth(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = Path(path) if path else TRUTH_CSV
    if not p.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("school_id") or "").strip()
            if not sid:
                continue
            out[sid] = {f: coerce_field(f, row.get(f, "")) for f in LABEL_FIELDS
                        if f != "label_notes"}
            out[sid]["label_notes"] = row.get("label_notes", "")
    return out


def intra_rater_agreement() -> Dict[str, Any]:
    """Compare the first labelling pass against the later re-label.

    This measures the ceiling on any accuracy claim: the pipeline cannot be
    shown to be more accurate than the ground truth is self-consistent.
    """
    a, b = load_truth(TRUTH_CSV), load_truth(RELABEL_CSV)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_schools": 0,
                "note": "no re-labelled schools yet; run `label --relabel`"}
    per_field: Dict[str, Dict[str, int]] = {}
    for sid in shared:
        for field in LABEL_FIELDS:
            if field == "label_notes":
                continue
            va, vb = a[sid].get(field), b[sid].get(field)
            if va is None and vb is None:
                continue
            d = per_field.setdefault(field, {"n": 0, "agree": 0})
            d["n"] += 1
            if field in config.COUNT_FIELDS:
                try:
                    same = abs(int(va) - int(vb)) <= config.COUNT_TOLERANCE.get(field, 1)
                except (TypeError, ValueError):
                    same = va == vb
            else:
                same = va == vb
            d["agree"] += int(bool(same))
    for field, d in per_field.items():
        d["agreement"] = round(d["agree"] / d["n"], 3) if d["n"] else None
    overall_n = sum(d["n"] for d in per_field.values())
    overall_k = sum(d["agree"] for d in per_field.values())
    return {"n_schools": len(shared), "per_field": per_field,
            "overall_agreement": round(overall_k / overall_n, 3) if overall_n else None}


# --------------------------------------------------------------------------- #
# The labelling tool
# --------------------------------------------------------------------------- #
_HTML_HEAD = """<!doctype html><meta charset="utf-8">
<title>School facilities - blind labelling</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1b;padding:10px 16px;border-bottom:1px solid #333;z-index:9}
 .wrap{display:flex;gap:16px;padding:16px;align-items:flex-start}
 .imgs{flex:1;min-width:0}
 .imgs img{max-width:100%;border:1px solid #333;margin-bottom:8px;background:#000}
 form{width:330px;position:sticky;top:60px}
 label{display:block;margin:9px 0 2px;color:#9cf;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
 input,select,textarea{width:100%;padding:6px;background:#222;color:#eee;border:1px solid #444;border-radius:4px}
 button{margin-top:12px;padding:8px 14px;background:#2a6;color:#fff;border:0;border-radius:4px;cursor:pointer;font-size:14px}
 button.sec{background:#444}
 .muted{color:#888;font-size:12px}
 .codebook{padding:12px 16px;background:#181818;border-top:1px solid #333;white-space:pre-wrap;
           font:12px/1.45 ui-monospace,monospace;color:#bbb;max-height:230px;overflow:auto}
</style>
<header>
  <b>Blind labelling</b> &mdash; <span id="pos"></span> &nbsp;
  <button class="sec" onclick="prev()">&larr; prev</button>
  <button class="sec" onclick="next()">next &rarr;</button>
  <button onclick="download()">download truth.csv</button>
  <span class="muted" id="saved"></span>
</header>
<div class="wrap">
  <div class="imgs" id="imgs"></div>
  <form id="form"></form>
</div>
<div class="codebook" id="codebook"></div>
<script>
"""

_HTML_TAIL = """
const FIELDS=[
 ["solar_present","select",["","yes","no"]],
 ["solar_area_m2","number",null],
 ["portable_count","number",null],
 ["fencing_extent","select",["","full","partial","none"]],
 ["fencing_type","select",["","chain-link","wrought iron","wall","other"]],
 ["track_present","select",["","yes","no"]],
 ["field_count","number",null],
 ["court_count","number",null],
 ["pool_present","select",["","yes","no"]],
 ["label_notes","text",null]];
let i=0;
const store=JSON.parse(localStorage.getItem("schoolcv_labels")||"{}");
function cur(){return SCHOOLS[i];}
function render(){
 const s=cur();
 document.getElementById("pos").textContent=`${i+1}/${SCHOOLS.length}  ${s.name} (${s.level}, ${s.enrollment} students)`;
 document.getElementById("imgs").innerHTML=s.images.map(p=>`<img loading="lazy" src="${p}">`).join("");
 const v=store[s.id]||{};
 document.getElementById("form").innerHTML=FIELDS.map(([f,t,opts])=>{
  const val=v[f]!==undefined?v[f]:"";
  if(t==="select")return `<label>${f}</label><select name="${f}">`+
    opts.map(o=>`<option ${o===val?"selected":""}>${o}</option>`).join("")+`</select>`;
  return `<label>${f}</label><input name="${f}" type="${t==='number'?'number':'text'}" value="${val}">`;
 }).join("");
 document.getElementById("form").oninput=save;
 document.getElementById("codebook").textContent=CODEBOOK;
}
function save(){
 const s=cur(),fd=new FormData(document.getElementById("form")),o={};
 for(const [k,val] of fd.entries())o[k]=val;
 store[s.id]=o;localStorage.setItem("schoolcv_labels",JSON.stringify(store));
 const n=Object.keys(store).length;
 document.getElementById("saved").textContent=`saved (${n}/${SCHOOLS.length})`;
}
function next(){if(i<SCHOOLS.length-1){i++;render();}}
function prev(){if(i>0){i--;render();}}
function download(){
 const cols=["school_id","school_name"].concat(FIELDS.map(f=>f[0]));
 const rows=[cols.join(",")];
 for(const s of SCHOOLS){
  const v=store[s.id]||{};
  rows.push(cols.map(c=>{
   let x = c==="school_id"?s.id : c==="school_name"?s.name : (v[c]!==undefined?v[c]:"");
   x=String(x).replace(/"/g,'""');
   return /[",]/.test(x)?`"${x}"`:x;
  }).join(","));
 }
 const b=new Blob([rows.join("\\n")],{type:"text/csv"});
 const a=document.createElement("a");
 a.href=URL.createObjectURL(b);a.download="truth.csv";a.click();
}
document.addEventListener("keydown",e=>{
 if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT"||e.target.tagName==="TEXTAREA")return;
 if(e.key==="ArrowRight")next(); if(e.key==="ArrowLeft")prev();
});
render();
</script>
"""


def build_label_tool(out_path: Optional[Path] = None) -> Path:
    """Emit a self-contained HTML page for labelling from the cached imagery.

    Deliberately shows imagery only -- no model predictions anywhere on the
    page -- so the labels stay blind.
    """
    from .pipeline import _jsonl_read
    from .resolve import load_schools
    from . import prompts

    out_path = Path(out_path) if out_path else config.LABELS / "label_tool.html"
    manifest = {r["school_id"]: r for r in
                _jsonl_read(config.PREDS / "imagery_manifest.jsonl")}
    schools = []
    for row in load_schools():
        sid = row["school_id"]
        m = manifest.get(sid, {})
        imgs = []
        for meta in [m.get("overview")] + (m.get("solar_crops") or []) + \
                    (m.get("perimeter_crops") or []):
            if meta and meta.get("cache_key"):
                rel = Path("..") / "cache" / "imagery" / f"{meta['cache_key']}.jpg"
                imgs.append(str(rel))
        schools.append({"id": sid, "name": row["school_name"],
                        "level": row.get("level", ""),
                        "enrollment": row.get("enrollment_2024", ""),
                        "images": imgs})

    html = (_HTML_HEAD
            + "const SCHOOLS=" + json.dumps(schools) + ";\n"
            + "const CODEBOOK=" + json.dumps(prompts.CODEBOOK) + ";\n"
            + _HTML_TAIL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
