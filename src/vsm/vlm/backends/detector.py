"""Grounding-DINO-tiny. Per-phrase 'class .', remove_combined=True, no repetition trick."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from vsm.config import REPO_ROOT, cfg_dict
from vsm.constants import COLS, DETECTOR_MODEL_ID, DETECTOR_PHRASES, QWEN_BBOX_MAX

_DET = cfg_dict("model")["backends"]["detector"]
_THRESHOLDS = [float(x) for x in _DET["box_threshold_sweep"]]
_DEFAULT_THR = float(_DET["box_threshold_default"])
_NMS_IOU = float(_DET["nms_iou"])
_CACHE = REPO_ROOT / str(_DET["cache_dir"])


class DetectorUnavailable(RuntimeError):
    """Torch/transformers extra is not installed. CI must not import this path."""


def _nms_class_agnostic(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Return keep indices. Class-agnostic, IoU from configs/model.yaml."""

    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thr]
    return keep


def _to_qwen(box: list[float], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        int(np.clip(x1 / w * QWEN_BBOX_MAX, 0, QWEN_BBOX_MAX - 1)),
        int(np.clip(y1 / h * QWEN_BBOX_MAX, 0, QWEN_BBOX_MAX - 1)),
        int(np.clip(x2 / w * QWEN_BBOX_MAX, 0, QWEN_BBOX_MAX - 1)),
        int(np.clip(y2 / h * QWEN_BBOX_MAX, 0, QWEN_BBOX_MAX - 1)),
    ]


class GroundingDinoTiny:
    """Open-vocab detector. Phrases are lowercase '{cls} .' with remove_combined=True."""

    name = "detector"

    def __init__(self, box_threshold: float | None = None) -> None:
        if box_threshold is not None and box_threshold not in _THRESHOLDS and box_threshold != _DEFAULT_THR:
            raise ValueError(f"box_threshold {box_threshold} not in frozen sweep {_THRESHOLDS}")
        self.box_threshold = float(box_threshold or _DEFAULT_THR)
        self.model_id = DETECTOR_MODEL_ID
        self._proc = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise DetectorUnavailable("install extra [detect] (timebox 30 min; not in CI)") from exc
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.eval()

    def detect_objects(self, frames: Any) -> pl.DataFrame:
        """Return one row per box. Cache parquet under data/cache/detector/."""

        self._load()
        import torch
        from PIL import Image

        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        phrases = list(DETECTOR_PHRASES)
        text = " ".join(phrases)  # processor still gets per-phrase queries below
        del text
        out_rows: list[dict] = []
        for row in rows:
            path = Path(str(row.get(COLS.CLIP_REF.value) or ""))
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            w, h = image.size
            for phrase in phrases:
                # per-phrase "class ." lowercase; no repetition trick
                inputs = self._proc(images=image, text=phrase.lower(), return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model(**inputs)
                results = self._proc.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs["input_ids"],
                    threshold=self.box_threshold,
                    text_threshold=float(_DET["text_threshold"]),
                    target_sizes=[(h, w)],
                )[0]
                boxes = (
                    results["boxes"].cpu().numpy() if hasattr(results["boxes"], "cpu") else np.asarray(results["boxes"])
                )
                scores = (
                    results["scores"].cpu().numpy()
                    if hasattr(results["scores"], "cpu")
                    else np.asarray(results["scores"])
                )
                labels = results.get("labels", ["object"] * len(boxes))
                if len(boxes) == 0:
                    continue
                keep = _nms_class_agnostic(np.asarray(boxes), np.asarray(scores), _NMS_IOU)
                for idx in keep:
                    box = [float(v) for v in boxes[idx]]
                    out_rows.append(
                        {
                            "frame_id": row.get(COLS.FRAME_ID.value),
                            "class": str(labels[idx] if idx < len(labels) else phrase).strip(),
                            "bbox_2d": _to_qwen(box, w, h),
                            "logit": float(scores[idx]),
                            "source": "grounding_dino_tiny",
                            "box_threshold": self.box_threshold,
                            "phrase": phrase,
                        }
                    )
        df = (
            pl.DataFrame(out_rows)
            if out_rows
            else pl.DataFrame(
                schema={
                    "frame_id": pl.Int64,
                    "class": pl.Utf8,
                    "bbox_2d": pl.List(pl.Int64),
                    "logit": pl.Float64,
                    "source": pl.Utf8,
                    "box_threshold": pl.Float64,
                    "phrase": pl.Utf8,
                }
            )
        )
        _CACHE.mkdir(parents=True, exist_ok=True)
        df.write_parquet(_CACHE / "boxes_logits.parquet")
        return df

    def mine_window(self, frames: Any, fps: float, prompts: dict[str, str]) -> Any:
        """Detector is T1, not a VLM; this adapter fills WindowOutput for smoke tests."""

        del fps, prompts
        from vsm.constants import GEOMETRY_B, PROVENANCE_GRID
        from vsm.vlm.schema import ObjectClaim, VLMFinding, WindowOutput, class_family

        df = self.detect_objects(frames)
        findings = []
        rows = frames.to_dicts() if isinstance(frames, pl.DataFrame) else list(frames)
        window_id = "detector"
        if rows:
            window_id = (
                f"{rows[0].get(COLS.SCENE_NAME.value)}_{rows[0].get(COLS.CAMERA.value)}_{rows[0].get(COLS.TS_US.value)}"
            )
        for rec in df.to_dicts() if df.height else []:
            bbox = tuple(int(v) for v in rec["bbox_2d"][:4])
            raw = str(rec["class"]).lower()
            mapped = "lane_split_moto" if "motor" in raw else ("pedestrian_cross" if "ped" in raw else None)
            if mapped is None:
                continue
            findings.append(
                VLMFinding(
                    frame_idx=int(rec["frame_id"] or 0),
                    objects=[ObjectClaim(cls=mapped, bbox_2d=bbox, lane_position="unknown")],  # type: ignore[arg-type]
                    claim=mapped,
                    supporting_frames=[int(rec["frame_id"] or 0)],
                    confidence=float(rec["logit"]),
                    family=class_family(mapped),
                    signal_state="unclear",
                    attributes={"source": "detector", "raw": raw},
                )
            )
        return WindowOutput(
            window_id=window_id,
            findings=findings,
            none_=len(findings) == 0,
            uncertain=False,
            parse_ok=True,
            backend=self.name,
            geometry_mode=GEOMETRY_B,
            provenance=PROVENANCE_GRID,
            model_id=self.model_id,
            detector_supported=True,
        )


def detect_objects(frames: Any) -> pl.DataFrame:
    """Module-level wrapper matching the protocol name."""

    return GroundingDinoTiny().detect_objects(frames)
