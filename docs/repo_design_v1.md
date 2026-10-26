# REPO_DESIGN v1 — Cấu trúc repo, quy trình, tên file/hàm (trước khi code)

> Phạm vi: **P0 của v5 MVP** (GT n=60, 2-3 backends, keyword index, CLI). P1/P2 chỉ ghi chỗ trống, không thiết kế sâu.
> Nguồn rationale: `architecture_final_v5_mvp.md` + `architecture_final_v4.md` (DESIGN_LOG). Mọi quyết định non-obvious kèm trích mục DESIGN_LOG để đỡ phải trả lời lại.

## 0. Nguyên tắc đặt tên (freeze trước commit đầu)

- Python: snake_case; class PascalCase; hàm public có docstring 1 dòng nói OUTPUT TRẢ GÌ.
- Module = tầng kiến trúc; không dùng tên mơ hồ (`utils.py`, `helpers.py`, `common.py` CẤM).
- Scope commit: `feat(GEOM):`, `fix(INGEST):`, `eval(METRICS):`, `docs(README):` — scope = thư mục.
- Mọi số/config không hard-code: vào `configs/*.yaml` (zero magic strings — DESIGN_LOG #18).
- Mọi record machine-generated mang: `prompt_version_hash`, `backend`, `model_id`, `decoding_params_hash`, `geometry_mode`, `provenance`.

## 1. Cây thư mục

```
vlm-scenario-mining/
├── README.md                     # 3 tầng (T1 TL;DR ≤150 từ + GIF / T2 1-page / T3 → docs/)
├── LICENSE, CHANGELOG.md, DECISIONS.md, PORTFOLIO_TEMPLATE.md(ref repo#1)
├── .gitignore                    # data/*, *.parquet, *.tgz, .cache — ngày 1
├── .github/workflows/ci.yml      # ≤5-15 phút, SYNTHETIC-only
├── pyproject.toml, uv.lock, Makefile
├── configs/
│   ├── taxonomy.yaml             # 9+1 class: TRIGGER/EXCLUDE/precedence/family/PEGASUS values/VN synonyms
│   ├── ingest.yaml               # cams(front-triad), w=6, fps=2, stride=3, mini URL, map pin v1.3
│   ├── model.yaml                # backends, dated model_id, temp=0/top_p=1, max_tokens, retry_repair=1
│   └── eval.yaml                 # n_gt=60, tIoU{0.3,0.5,0.7}, seeds, allocation rule, kill-switch checklist
├── src/vsm/
│   ├── config.py, constants.py
│   ├── ingest/      (download, convert, windows, conditions, integrity)
│   ├── geometry/    (projection, map_layers, fixture, synth_fixtures)
│   ├── vlm/         (schema, prompt, parse, backends/{base,fallback,detector,api_vlm,local_vlm})
│   ├── mining/      (rules, closure, cascade, cache, ledger)
│   ├── consolidate/ (stitch, identity)
│   ├── eval/        (gt, match, metrics, claim_audit)
│   ├── index/       (store, text, search, serve_cli)
│   ├── labeling/    (tool.py — 6-frame viewer + form + timer)
│   └── viz/         (plots.py, gif.py)
├── scripts/         (download_data, build_gt, run_mining, run_eval, run_index, make_assets, killswitch_check)
├── tests/           (mirror từng module — tên bên dưới)
├── data/            # GITIGNORE — raw/ frames/ cache/ coda/
└── results/         # bảng csv + plots commit được (nhỏ), GIF ≤10MB
```

## 2. Schema dữ liệu (Parquet + DuckDB) — freeze cùng pre-reg

```python
# frames.parquet (base table — TRUNG TÂM; JD2 "indexing raw sensor data")
frames: frame_id(int), sample_token(str), scene_token(str), scene_name(str),
        camera(str), ts_us(int64 ABSOLUTE), is_keyframe(bool),
        objects_json(str),          # detector/GT boxes: [{class, bbox_2d[0,1000), conf, source}]
        ego_pose_json(str),         # [x,y,heading] per ts
        clip_ref(str),              # path frame files
        provenance(str: grid|anchor), condition_light(str: day|dusk|night),
        blur_quartile(int), density_bin(str), visibility_bin(int)

# windows.parquet
windows: window_id(str: f"{scene}_{cam}_{ts0}"), scene_token, camera,
         ts_start_us, ts_end_us, frame_ids(list), provenance, geometry_mode(str: A_mapful|B_mapless)

# gt.parquet (labeler output — JSONL→Parquet)
gt: window_id, event_id(str — HUMAN-assigned cross-window, stitcher KHÔNG đụng GT),
    class_set(list[str]),          # multi-label: 1 maneuver + 0-1 spatial + ped_cross
    ts_start_us, ts_end_us,        # annotate TRÊN TIMELINE, không per-window
    signal_state(on|off|unclear), attributes_json, labeler, latency_s(float),
    definition_doubt(bool), note(str)

# findings (derived view)
findings: finding_id, window_id, frame_idx, class, family,
          objects_json(list bbox), claim(str), supporting_frames(list),
          confidence(float), confidence_source(verbalized|agreement),
          none(bool), uncertain(bool), parse_ok(bool),
          parse_subfail(refused|truncated|loop_detected|valid_but_empty|schema_violation|None),
          finish_reason, completion_tokens, truncated(bool),
          backend, model_id, decoding_hash, prompt_version_hash,
          geometry_mode, provenance, detector_supported(bool)

# ledger (cost/cache)
ledger: window_id, backend, model_id, cache_hit(bool), tokens_in, tokens_out,
        t_start, t_end, attempts, error_kind(None|rate|quota|network|schema|oom), precision_tag(fp16|4bit)

# pseudo_gt: window_id, class, ts_start_us, ts_end_us, join_rule_version, se_rule_checked(bool)
```

## 3. Module — file & hàm chính (signature thật)

### 3.1 `ingest/`
```python
# download.py
def download_minifest(url, dest) -> Path                      # curl -C - resume; KHÔNG tự SHA
def integrity_gate(tgz: Path) -> None                         # gzip -t + tables parse + scene==10 + count band  [CT-D1/D2]
# convert.py
def build_frames_table(mini_root, map_root, cfg) -> pl.DataFrame   # 1,200 rows (10 scenes × 40 kf × 3 cams)
def extract_gt_boxes(nusc, cfg) -> pl.DataFrame                    # GT 3D → projected 2D + drop_rate column
def pre_download_coda(dest) -> None                                # ngày 1, +SHA
# windows.py
def build_grid_windows(frames: pl.DataFrame, cfg) -> pl.DataFrame  # N=⌊(K−w)/s⌋+1 per (scene,cam) → 360
# conditions.py
def add_condition_columns(frames) -> pl.DataFrame                  # light từ description+ts; blur=Laplacian var→quartile; density; visibility_bin
# integrity.py
def run_ingestion_gate(frames) -> GateReport                       # isfinite + quat-norm ±ε + referential + 4 ts-invariants → QUARANTINE report
def nan_poison_test() -> None                                      # inject NaN → phải raise/flag (fixture test)
```

### 3.2 `geometry/`
```python
# projection.py — devkit-native path (DESIGN_LOG #10)
def project_box_global_to_pixel(box, nusc, sd_token) -> np.ndarray  # ego TRƯỚC cs; view_points axis-2
def filter_boxes_in_image(boxes, W, H, near=0.1) -> (kept, drop_rate)  # box_in_image + drop-rate
def rescale_qwen_bbox(bbox_1000, W, H) -> tuple                     # W/H RIÊNG từng trục
# map_layers.py
def load_map_layers(map_name) -> dict[str, Polygon]                 # driveable_area, sidewalk, ped_crossing; bounds assert
def bev_lane_position(x, y, lane_name) -> str                       # project_pose_to_lane → lateral offset → {in_lane|adjacent|off_lane}  [DESIGN_LOG #12]
# fixture.py — chạy trong CI trên SYNTHETIC
def run_geometry_fixture() -> FixtureReport                         # 6 asserts + MUTATION test (inject từng bug → phải fail) [R4 #2]
def lidar_points_in_box_check(sample) -> float                      # % lidar pts trong GT box ≥ ngưỡng — independent path
```

### 3.3 `vlm/`
```python
# schema.py
class ObjectClaim(BaseModel):  cls: str; bbox_2d: tuple[int,int,int,int]; lane_position: str
class VLMFinding(BaseModel):   frame_idx, objects[list[ObjectClaim]], claim, supporting_frames,
                               confidence, family, signal_state: Literal[on,off,unclear], attributes: dict
class WindowOutput(BaseModel): window_id, findings, none_: bool, uncertain: bool, parse_ok,
                               parse_subfail: Subfail|None, finish_reason, completion_tokens, truncated,
                               prompt_version_hash, backend, geometry_mode, provenance
# prompt.py
def build_system_prompt(taxonomy) -> str                # "no clear scenario → none; not sure → uncertain"
def build_few_shot(taxonomy, corpus, gt_scenes) -> list # 2/class — SCENE-DISJOINT với GT (assert)
def prompt_version_hash(...) -> str
# parse.py
def parse_output(raw, finish_reason, max_tokens) -> WindowOutput    # strip fence 0-call; truncated flag ≥0.95×
def classify_subfail(raw) -> Subfail                                # loop = n-gram repetition threshold
def repair_once(output, error_kind) -> WindowOutput                 # fence→strip(0 call); truncation→max_tokens×1.5 1 lần
# backends/base.py
class VLMBackend(Protocol):    def mine_window(self, frames, fps, prompts) -> WindowOutput
class DetectorBackend(Protocol): def detect_objects(self, frames) -> pl.DataFrame  # boxes+logits → cache Parquet
# backends/detector.py — GD-Tiny: per-phrase "class .", remove_combined=True, sweep {0.25,0.30,0.35,0.40} frozen, class-agnostic NMS (supervision, IoU 0.5)
# backends/api_vlm.py — qwen3-vl-plus: video-file + fps param; log usage.prompt_tokens THẬT; Free Quota Only mode; 429=retryable backoff full-jitter, 403/quota=TERMINAL
# backends/local_vlm.py — Qwen3-VL-2B (branch local-only); precision_tag fp16|4bit
# backends/fallback.py — keyword CI-only, comment "NOT A BASELINE"
```

### 3.4 `mining/`
```python
# rules.py — T0, required_capabilities per rule; Mode A (BEV map) / Mode B (bbox-only, flag geometry_mode=degraded)
RULES: dict[str, Rule]   # Rule(capabilities, fn, family, expected_fp_mode)
def rule_lane_split_moto(boxes, mode) -> list[Flag]   # straddle ±0.75m ranh giới (Mode A) / band trung tâm pixel (Mode B), ≥3 frames, moving
def rule_lane_intrusion(...)                          # bbox∩ego-lane ≥25% ≥2 frames — predicate DƯƠNG
def rule_sidewalk_mount(...)                          # ≥50% bbox trên walkway (A) / road-vs-nonroad mask (B)
def rule_cut_in(...); rule_lead_braking(...)          # closure + velocity
# closure.py — deterministic, $0 (BUILD — DESIGN_LOG #26)
def scale_slope_flag(boxes_seq) -> ClosureFlag        # slope-fit 1/h qua window (KHÔNG finite-diff)
def relative_closure(boxes_seq, ego_traj) -> float    # tách ego-odometry khỏi lead-brake
def jitter_sensitivity(frames, eps_px) -> FlipRate    # ±2/4px injection → flip rate
# cascade.py
def run_cascade(windows, backends, cfg) -> pl.DataFrame   # T0 auto-accept/flag → T1 screen → T2 verify
def assert_t0_no_gt_access(manifest) -> None              # SANITIZED-MANIFEST: xóa sample_annotation → output identical [R4 #3 — assert nguy nhất]
# cache.py
def cache_key(window, prompt_hash, model_id, decoding_hash) -> str   # SHA
def load_or_run(...) -> (WindowOutput, ledger_row)                   # atomic write tmp+rename; ledger row pending TRƯỚC call
# ledger.py
def append_ledger(row) -> None                       # error_kind {rate|quota|network|schema|oom} — OOM-tag riêng, checkpoint N=50, heartbeat file
```

### 3.5 `consolidate/`
```python
# stitch.py
def stitch_events(findings, cfg) -> pl.DataFrame     # union-merge per class, gap≤1 window, over-merge guard; pre/post-stitch COUNT DELTA ghi riêng (pred-only)
# identity.py
def associate_windows(events, tau=0.3, size_scaled=True) -> list[Event]  # Hungarian-on-IoU + center-distance fallback (CenterPoint-style)
def event_identity_key(event) -> str                 # class-SET + track — multi-label 2 trục
```

### 3.6 `eval/`
```python
# gt.py
def prevalence_audit(windows, pseudo, maps) -> PrevalenceTable   # stratified THEO MAP ≥5/map; <3 pos → out-of-headline (NGÀY 2)
def build_gt_list(frames, cfg) -> list[window_id]                # stratified metadata + SHA + allocation rule
def labeling_session(...)                                        # gọi labeling/tool.py; interleave scene; cap 75'
# match.py
def temporal_iou(pred, gt) -> float
def greedy_score_match(preds, gts, tIoU) -> MatchReport          # sort by score; P/R@τ; AP = secondary
def frame_iou_decompose(...) -> Recal3                           # localized | poorly | missed
# metrics.py
def neff_wilson(k, n, icc, m̄) -> CI                              # ICC từ data (ANOVA ~10 dòng); deff∈[1,2.5]
def per_scene_range(recall_by_scene) -> Report                   # 10 số min/med/max
def condition_breakdown(df, by="light") -> Table                 # <5 pos → "insufficient support"
def interpretation_gate(primary_claims) -> None                  # assert README số đầu = primary table
# claim_audit.py
def gt_bounded_hallucination_rate(findings, gt) -> float         # CHỈ trên GT windows — tên này mới được gọi "hallucination"
def unsupported_claim_rate(findings, detector_boxes) -> float    # corpus-wide — "includes detector misses"
def visibility_stratified_audit(...) -> Table                    # ACC@IoU{0.3,0.5,0.7} + Center-match per visibility token
```

### 3.7 `index/`
```python
# text.py
def objects_to_text(objects_json) -> str        # template DUY NHẤT cho dense (P2) + keyword; caption VLM → keyword field
def vn_synonym_map() -> dict[str, list[str]]    # 10-40 entry, artifact review được
def normalize_vn_text(s) -> str                 # accent/đ offline đồng bộ doc+query — test "đ" ngay
# store.py
def create_duckdb(frames, windows, findings, gt) -> duckdb.DuckDBPyConnection   # 1 DB duy nhất; view findings = derived
def create_fts_index(conn) -> None              # stemmer='none'; FTS trên EN docs
# search.py
def search_scenarios(conn, query_vn, top_k) -> list[WindowHit]      # synonyms → EN → BM25; rank unit = WINDOW
def search_frames(conn, query, top_k) -> list                       # diagnostic only
def negative_space(conn, cls=None) -> CoverageReport                # count / T0-evidence / parse-fail TÁCH RIÊNG — KHÔNG gọi "recall"
# serve_cli.py — CLI primary: `vsm search "..."` | `vsm stats` | `vsm coverage` | `vsm fetch <window_id>`
```

### 3.8 `labeling/tool.py` (0.5 ngày, smoke test 5 windows trước phiên đầu)
```
6-frame grid + form pydantic (1 maneuver + 0-1 spatial + none/uncertain/doubt-field)
auto-timer (open→submit → latency_s) · blinded-relabel mode (anchor) · JSONL out theo gt schema
CẤM gold-plate: không video scrubber, không đẹp — pandas phải đọc được output nếu tool chết
```

### 3.9 `viz/`
```python
def frontier_plot(ledger, eval) -> results/frontier.png     # trục $/1k vs recall@0.5, annotate τ points
def quadrant_heatmap(t0_flags, vlm_flags) -> results/quadrant.png
def demo_gif() -> results/demo.gif                          # vsm search → findings → frame highlight; ≤10MB
```

## 4. Tests — tên file 1-1 với module

```
tests/test_integrity_gate.py      # CT-D1..D7: corrupt tgz, NaN-poison, ts-invariants, map-version, bounds
tests/test_geometry_fixture.py    # 6 asserts + mutation + lidar-check
tests/test_schema.py              # VLMFinding validation, truncated flag, subfail classification
tests/test_parse_repair.py        # fence/truncation/loop → repair đúng loại
tests/test_rules_modeAB.py        # Mode A vs Mode B cùng input synthetic → cùng class-set (nếu có thể)
tests/test_closure_flags.py       # slope-fit + jitter-injection flip rate
tests/test_stitch_identity.py     # over-merge guard, gap-tolerance, pre/post delta, class-set identity
tests/test_match_metrics.py       # greedy vs Hungarian khác gì, tIoU sweep, neff-wilson vs CP naive
tests/test_claim_audit.py         # GT-bounded vs unsupported — denominator asserts
tests/test_search_negative.py     # synonym VN→EN, negative-space, coverage 3 cột
tests/test_cascade_synthetic.py   # E2E mini-pipeline trên 3 windows SYNTHETIC (CI job chính)
tests/test_no_gt_leak.py          # sanitized-manifest + few-shot scene-disjoint + percentile nguồn τ
```

## 5. CI workflow (`.github/workflows/ci.yml`) — ≤5-15 phút

```yaml
jobs:
  lint: ruff check + ruff format --check
  test: pytest -m "not slow"          # toàn bộ trên synthetic fixtures — KHÔNG tải nuScenes
  mini-pipeline: python -m vsm.selftest cascade_synthetic   # 3 windows synthetic, fallback backend
# KHÔNG matrix, KHÔNG data download — đỏ luôn là đỏ-thật
```

## 6. Makefile + CLI

```makefile
install, data (download+gate+convert), windows, mine (cascade + cache), index, eval,
report (tables + plots), label (launch tool), assets (GIF/plots), test, lint, tags
```

## 7. Thứ tự build (dependency graph — map vào timeline v5)

```
Ngày 1: config+constants → ingest/(download,integrity,convert,windows,conditions)
        → geometry/(projection,map_layers,fixture)  [CHẶN: không convert trước khi fixture xanh]
        → vlm/schema.py + labeling/tool.py + smoke test
Ngày 2: eval/gt.py prevalence_audit → pre-reg 8 mục freeze → GT list + SHA → label 10
Ngày 3: vlm/prompt+parse+backends(detector,api) → mining/(rules,closure,cascade) → smoke E2E 5 windows BẮT BUỘC
Ngày 4-5: mining/(cache,ledger) + overnight detector cache → mining 60 GT → error taxonomy
Ngày 6-7: index/(store,text,search,serve_cli) → consolidate/ → KILL-SWITCH 18:00 ngày 7
Ngày 9-10: eval/(match,metrics,claim_audit) run → condition breakdown
Ngày 12: viz/ + README 3 tầng + GIF
```

Rule ràng buộc: **không module nào import module ở tầng cao hơn** (ingest không biết về mining; geometry không biết về vlm). Mỗi module xong = test xanh + 1 commit `feat(SCOPE):` — decision lớn (VD chọn Mode B threshold) = decision-commit riêng chạm configs/ + DECISIONS.md.

## 8. Artifacts — tên output chuẩn

```
results/gt_prevalence.csv · results/headline_metrics.csv · results/claim_audit.csv
results/condition_breakdown.csv · results/cost_frontier.csv · results/ledger_final.csv
results/{frontier,quadrant}.png · results/demo.gif
docs/EVAL_PRE_REGISTRATION.md · docs/DATA_SOURCE.md · docs/CAPABILITY_MANIFEST.md
DECISIONS.md  # 30+ findings R1-R4: ADOPTED (mỗi dòng 1 fix) / REJECTED (mỗi dòng 1 lý do)
```

## 9. Điểm mở cho P1/P2 (chỉ tên slot, không thiết kế)
- `index/embed.py` (bge-m3 + RRF) · `serve/mcp_server.py` (stdio, 5 tools) · `eval/routing.py` (τ per family + frontier) · `eval/calibration.py` (Platt) · `mining/crop_verify.py` (pass-2) — chỉ tạo khi kill-switch cho phép v1.0-track.
