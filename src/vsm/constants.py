"""Column names, class ids, and invariant identifiers. Numbers live in configs/."""

from __future__ import annotations

from enum import Enum

from vsm.config import cfg_dict

_INGEST = cfg_dict("ingest")
_TAX = cfg_dict("taxonomy")
_MODEL = cfg_dict("model")
_EVAL = cfg_dict("eval")

CAMERAS: tuple[str, ...] = tuple(_INGEST["cameras"])
WINDOW_W: int = int(_INGEST["window"]["w"])
WINDOW_FPS: float = float(_INGEST["window"]["fps"])
WINDOW_STRIDE: int = int(_INGEST["window"]["stride"])
IMAGE_W: int = int(_INGEST["image"]["width"])
IMAGE_H: int = int(_INGEST["image"]["height"])
QWEN_BBOX_MAX: int = int(_INGEST["image"]["qwen_bbox_max"])
QUAT_NORM_EPS: float = float(_INGEST["integrity"]["quat_norm_eps"])
TS_GAP_S: float = float(_INGEST["integrity"]["ts_gap_s"])
TS_GAP_EPS_S: float = float(_INGEST["integrity"]["ts_gap_eps_s"])
N_GT: int = int(_EVAL["n_gt"])
TIOU: tuple[float, ...] = tuple(float(x) for x in _EVAL["tIoU"])
GEOMETRY_A: str = str(_INGEST["geometry_mode"]["mapful"])
GEOMETRY_B: str = str(_INGEST["geometry_mode"]["mapless"])
PROVENANCE_GRID: str = str(_INGEST["provenance"]["grid"])
PROVENANCE_ANCHOR: str = str(_INGEST["provenance"]["anchor"])

CLASS_IDS: tuple[str, ...] = tuple(_TAX["classes"].keys())
FAMILIES: dict[str, list[str]] = dict(_TAX["families"])
DETECTOR_PHRASES: tuple[str, ...] = tuple(_TAX["detector_phrases"])
DETECTOR_MODEL_ID: str = str(_MODEL["backends"]["detector"]["model_id"])
API_MODEL_ID: str = str(_MODEL["backends"]["api_vlm"]["model_id"])
FALLBACK_NAME: str = str(_MODEL["backends"]["fallback"]["name"])


class COLS(str, Enum):
    """Column names used past the ingestion gate (frames.parquet)."""

    FRAME_ID = "frame_id"
    SAMPLE_TOKEN = "sample_token"
    SCENE_TOKEN = "scene_token"
    SCENE_NAME = "scene_name"
    CAMERA = "camera"
    TS_US = "ts_us"
    IS_KEYFRAME = "is_keyframe"
    OBJECTS_JSON = "objects_json"
    EGO_POSE_JSON = "ego_pose_json"
    CLIP_REF = "clip_ref"
    PROVENANCE = "provenance"
    CONDITION_LIGHT = "condition_light"
    BLUR_QUARTILE = "blur_quartile"
    DENSITY_BIN = "density_bin"
    VISIBILITY_BIN = "visibility_bin"
    GT_DROP_RATE = "gt_drop_rate"
    MAP_NAME = "map_name"
    QUARANTINE = "quarantine"


class WIN_COLS(str, Enum):
    """windows.parquet columns."""

    WINDOW_ID = "window_id"
    SCENE_TOKEN = "scene_token"
    CAMERA = "camera"
    TS_START_US = "ts_start_us"
    TS_END_US = "ts_end_us"
    FRAME_IDS = "frame_ids"
    PROVENANCE = "provenance"
    GEOMETRY_MODE = "geometry_mode"


class Invariant(str, Enum):
    """Ingestion invariants. Fail → QUARANTINE the scene, do not auto-repair."""

    FINITE_TRANSLATION = "I_finite_translation"
    FINITE_ROTATION = "I_finite_rotation"
    FINITE_CALIB = "I_finite_calib"
    QUAT_UNIT = "I_quat_unit"
    REF_INTEGRITY = "I_referential"
    TS_STRICT_INC = "I_ts_strictly_increasing"
    TS_NO_DUP = "I_ts_dup_zero"
    TS_GAP = "I_ts_gap"
    KEYFRAME_FLAG = "I_keyframe_flag"
    NAN_POISON = "I_nan_poison"
    MAP_BOUNDS = "I_map_bounds"
    MAP_VERSION = "I_map_version"
