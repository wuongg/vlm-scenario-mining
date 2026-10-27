# Outreach materials (draft — user review Days 12–14)

Numbers are the frozen table. Do not invent a fourth frontier point.

## CV line — short (LinkedIn, ≤20 words)

3-stage VLM data engine mines ADAS edge cases from 360 driving windows — detector-audited, pre-registered eval (repo).

Word count: 18.

## CV line — long (repo / technical)

Built a three-stage labeling cascade (rules → open-vocab detector → VLM) over a frame-level index of nuScenes mini (360 windows, GT n=60). Pre-registered eval: headline `{pedestrian_cross}` event recall@tIoU=0.5 = 1.0 at P=0.196; the story is the join (box-level VRU P=0.440 / R=0.212). $0 cost frontier (detector / API-GT60 / union). Claim-audit on 4 API claims (2 GT-bounded hallucinations; qualitative only). CODA-LM Test 500 as a cross-locale object-level sanity check (R=0.345; barrier/cone structural zeros). Kill-switch 3/3. Routing is a family-level skeleton — confidence source unavailable (logprob stub).

No inter-rater claim. CODA is its own bullet, not mixed into the mini headline.

## 90-second script

See [`docs/SCRIPT_90S.md`](SCRIPT_90S.md). Two acts: repo#1 (EV scheduler / VOI) → this repo (frozen cell + join + limitations).

## Definition cards

Class cards + eval vocabulary (policy / hallucination / frontier / pre-registered / negative-space): [`docs/DEFINITION_CARDS.md`](DEFINITION_CARDS.md).

## LinkedIn / VN-DS post (draft — do not publish until user review)

Detector event recall is 1.0. Box recall is 0.212. The join rule is the story.

I built a three-stage labeling cascade (rules → open-vocab detector → VLM) over a frame-level index of nuScenes mini.

- What: mines ADAS events from 360 driving windows; keyword search returns window_id + findings + coverage.
- Eval: pre-registered headline `{pedestrian_cross}` @ tIoU=0.5. Kill-switch 3/3. Claim-audit: 2/4 GT-bounded hallucinations (qualitative, n_pos<5). CODA-LM Test 500 as a cross-locale object check (R=0.345).
- Frontier: three $0 points. Routing stays a skeleton — the reseller's logprobs flag is a stub.

Repo: https://github.com/wuongg/vlm-scenario-mining

#computervision #autonomousdriving #dataengineering

(~145 words. No “VN data”. No agent count.)
