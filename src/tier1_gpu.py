"""Layer 1: open-weight models on a GPU (USC CARC).

This layer exists to produce signals a hosted API physically cannot return:
segmentation masks, detector logits, and enough throughput for test-time
augmentation. Those are the inputs the calibration step actually needs.

Runs on CARC as Stage B with `HF_HUB_OFFLINE=1`; weights are pre-staged by
Stage A. Outputs go to `preds/{model}.jsonl`, which are committed so the
pipeline reproduces without a GPU.

Two domain-shift facts drive the solar code, both from the BDAPPV model card:

  * Segmentation IoU falls from 0.884 (Google->Google) to 0.561 when the
    imagery provider changes. Our imagery is Esri World Imagery -- a *third*
    provider. So the honest prior is the cross-provider number or worse, and
    the model's own confidence must be re-calibrated against our hand labels
    rather than inherited from the card.
  * The model was trained at 400x400 px on ~0.1 m/px imagery. A campus overview
    at 0.31 m/px is three times coarser than training. So we run it on tight
    building-centred crops sized to land near 0.1 m/px, not on the overview.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, prompts

# Target ground resolution for the solar segmenter, chosen to match its training
# distribution rather than whatever the campus overview happens to be.
SOLAR_TARGET_MPP = 0.12
SOLAR_CROP_PX = 400

OWLV2_QUERIES = [
    "a solar panel array", "a swimming pool", "a running track",
    "a tennis court", "a basketball court", "a baseball diamond",
    "a portable classroom building",
]
OWLV2_FIELD_MAP = {
    "a solar panel array": "solar_present",
    "a swimming pool": "pool_present",
    "a running track": "track_present",
    "a tennis court": "court_count",
    "a basketball court": "court_count",
    "a baseball diamond": "field_count",
    "a portable classroom building": "portable_count",
}


# --------------------------------------------------------------------------- #
# Test-time augmentation
# --------------------------------------------------------------------------- #
def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 symmetries of a square: 4 rotations x optional flip.

    Overhead imagery has no canonical orientation, so all 8 are in-distribution.
    This is the cheapest genuine source of epistemic uncertainty available, and
    it is only affordable because this layer has no per-request quota.
    """
    out = np.rot90(arr, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def undihedral_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """Invert `dihedral` so masks from augmented views can be averaged."""
    out = np.fliplr(mask) if k >= 4 else mask
    return np.ascontiguousarray(np.rot90(out, -(k % 4)))


# --------------------------------------------------------------------------- #
# Solar segmentation (BDAPPV DeepLabV3-ResNet101)
# --------------------------------------------------------------------------- #
class SolarSegmenter:
    def __init__(self, provider: str = "google", device: str = "cuda"):
        import importlib.util
        import torch
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("gabrielkasmi/bdappv-models", "model.py")
        spec = importlib.util.spec_from_file_location("bdappv_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.torch = torch
        self.device = device
        self.model = mod.load_segmentation_model(provider, device=device)
        self.provider = provider

    def _prep(self, rgb: np.ndarray):
        import torch
        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def probability_map(self, rgb: np.ndarray, tta: int = 8) -> Tuple[np.ndarray, float]:
        """Mean and dispersion of the panel probability over TTA views.

        Returns (mean probability map, mean per-pixel std across views). The std
        is the epistemic-uncertainty feature; it is high exactly where the model
        is guessing.
        """
        torch = self.torch
        maps = []
        with torch.no_grad():
            for k in range(max(1, tta)):
                view = dihedral(rgb, k)
                logits = self.model(self._prep(view))["out"]
                prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
                maps.append(undihedral_mask(prob, k))
        stack = np.stack(maps, axis=0)
        return stack.mean(axis=0), float(stack.std(axis=0).mean())


def solar_from_crops(segmenter: "SolarSegmenter", crops: List[Dict[str, Any]],
                     threshold: float = config.SOLAR_MASK_THRESHOLD,
                     tta: int = 8) -> Dict[str, Any]:
    """Aggregate panel area over building-centred crops.

    Area comes from counting mask pixels and multiplying by the crop's own
    metres-per-pixel. No model is ever asked for a number in square metres.
    """
    total_m2 = 0.0
    per_crop, probs, stds = [], [], []
    for crop in crops:
        rgb, mpp = crop["rgb"], crop["m_per_px"]
        prob, std = segmenter.probability_map(rgb, tta=tta)
        mask = prob >= threshold
        area = float(mask.sum()) * mpp * mpp
        total_m2 += area
        peak = float(prob.max())
        mean_in_mask = float(prob[mask].mean()) if mask.any() else 0.0
        per_crop.append({
            "bbox": crop.get("bbox"), "m_per_px": round(mpp, 4),
            "area_m2": round(area, 1), "peak_prob": round(peak, 4),
            "mean_prob_in_mask": round(mean_in_mask, 4),
            "tta_std": round(std, 4),
        })
        probs.append(peak)
        stds.append(std)
    peak_overall = max(probs) if probs else 0.0
    return {
        "solar_present": {
            "value": bool(total_m2 > 8.0 and peak_overall >= 0.6),
            "source": f"bdappv_deeplab_{segmenter.provider}",
            "score": peak_overall,
            "tta_std": float(np.mean(stds)) if stds else None,
            "evidence": {"crops": per_crop},
        },
        "solar_area_m2": {
            "value": round(total_m2, 1),
            "source": f"bdappv_deeplab_{segmenter.provider}",
            "score": peak_overall,
            "tta_std": float(np.mean(stds)) if stds else None,
            "evidence": {"crops": per_crop},
        },
    }


# --------------------------------------------------------------------------- #
# Open-vocabulary detection (OWLv2)
# --------------------------------------------------------------------------- #
class OwlDetector:
    """OWLv2-large. Pure `transformers`, so no CUDA-extension build to fail."""

    def __init__(self, device: str = "cuda",
                 model_id: str = "google/owlv2-large-patch14-ensemble"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, rgb: np.ndarray, queries: Sequence[str],
               threshold: float = 0.12) -> List[Dict[str, Any]]:
        import torch
        from PIL import Image
        img = Image.fromarray(rgb)
        inputs = self.processor(text=[list(queries)], images=img,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target = torch.tensor([[img.height, img.width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target, threshold=threshold)[0]
        dets = []
        for score, label, box in zip(results["scores"], results["labels"],
                                     results["boxes"]):
            dets.append({
                "query": queries[int(label)],
                "score": float(score),           # a real logit-derived score
                "box": [round(float(v), 1) for v in box.tolist()],
            })
        dets.sort(key=lambda d: -d["score"])
        return dets


def owl_to_fields(dets: List[Dict[str, Any]], mpp: float) -> Dict[str, Any]:
    """Turn detections into per-field values plus their max detector score."""
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        field = OWLV2_FIELD_MAP.get(d["query"])
        if field:
            by_field.setdefault(field, []).append(d)
    out: Dict[str, Any] = {}
    for field, ds in by_field.items():
        top = max(d["score"] for d in ds)
        value = (len(ds) if field in config.COUNT_FIELDS else True)
        out[field] = {"value": value, "source": "owlv2", "score": top,
                      "evidence": {"detections": ds[:8]}}
    for field in ("solar_present", "pool_present", "track_present"):
        out.setdefault(field, {"value": False, "source": "owlv2", "score": 0.0,
                               "evidence": {"detections": []}})
    return out


# --------------------------------------------------------------------------- #
# Large VLM via vLLM (token logprobs are the point)
# --------------------------------------------------------------------------- #
class VllmVlm:
    """Wraps a vLLM OpenAI-compatible server.

    We use vLLM rather than raw `transformers` specifically because it returns
    `logprobs`, which is what makes P(yes) a real probability instead of a
    number the model wrote down.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen3-VL-32B-Instruct"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        import requests
        r = requests.post(f"{self.base_url}/chat/completions", json=body,
                          timeout=(10, 300))
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _content(prompt: str, jpeg_b64: str) -> List[Dict[str, Any]]:
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
        ]

    def binary(self, jpeg_b64: str, school: str, field: str) -> Dict[str, Any]:
        from .tier2_api import _extract_yes_logprob
        prompt = prompts.binary_prompt(school, prompts.BINARY_QUESTIONS[field])
        payload = self._post({
            "model": self.model,
            "messages": [{"role": "user",
                          "content": self._content(prompt, jpeg_b64)}],
            "temperature": 0.0, "max_tokens": 4,
            "logprobs": True, "top_logprobs": 10,
        })
        text = payload["choices"][0]["message"]["content"].strip().lower()
        p_yes = _extract_yes_logprob(payload)
        return {"value": text.startswith("y") if text else None,
                "source": f"vlm:{self.model}", "p_yes": p_yes,
                "score": p_yes if p_yes is not None else None,
                "evidence": {"text": text}}

    def survey(self, jpeg_b64: str, school: str, level: str,
               enrollment: int) -> Dict[str, Any]:
        from .tier2_api import parse_json_response
        payload = self._post({
            "model": self.model,
            "messages": [{"role": "user", "content": self._content(
                prompts.survey_prompt(school, level, enrollment), jpeg_b64)}],
            "temperature": 0.0, "max_tokens": 700,
        })
        text = payload["choices"][0]["message"]["content"]
        return {"parsed": parse_json_response(text), "text": text,
                "source": f"vlm:{self.model}"}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    """Stage B entry point. One model family per SLURM array task."""
    ap = argparse.ArgumentParser(description="GPU inference stage")
    ap.add_argument("--task", required=True,
                    choices=["solar", "owl", "vlm"],
                    help="which model family this array task runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--vlm-url", default="http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from .pipeline import gpu_stage   # imported late so CPU-only runs need no torch
    gpu_stage(args)


if __name__ == "__main__":
    main()
