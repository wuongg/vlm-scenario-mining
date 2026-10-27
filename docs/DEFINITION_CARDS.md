# 90-second definition cards (1 / class)

Read TRIGGER then EXCLUDE. Gold for the Day-2+ anchor-alarm. Do not edit after the first real labeling session without a decision-commit.

`uncertain` is a schema slot, not a class — do not write it into `class_set`. `signal_state` is an attribute (on|off|unclear) and does not participate in matching.

| class | family | TRIGGER | EXCLUDE |
|---|---|---|---|
| cut_in | safety_maneuver | Another vehicle enters the ego lane from an adjacent lane and becomes the lead, closing laterally then longitudinally. | Ego lane-change; already-ahead vehicle; parked vehicle that never occupies the ego path. |
| lead_braking | safety_maneuver | Lead vehicle decelerates sharply in-lane (closure slope on 1/h, not a single-frame jump). | Ego braking with a stable lead; stopped traffic already at rest for the whole window. |
| oncoming | safety_maneuver | WRONG-WAY-ONLY: actor occupies a lane whose legal direction is opposite the actor heading. | Traffic in a correctly oriented opposite lane. Ignore this EXCLUDE and prevalence ≈ 100%. |
| turn_lane_drift | safety_maneuver | Vehicle in a dedicated turn-lane goes straight instead of turning (one frozen turn direction). | Vehicle that turns as marked; through-lane traffic that never entered a turn pocket. |
| lane_split_moto | safety_spatial | Motorcycle straddles the lane boundary (≥3 frames, moving) — VN "xe máy cắt làn". | Motorcycle fully inside one lane; bicycle; parked two-wheeler on the shoulder. |
| sidewalk_mount | safety_spatial | ≥50% of the object bbox sits on walkway (Mode A) or non-road mask (Mode B). | Pedestrian on sidewalk; vehicle legally in a driveway cut. |
| lane_intrusion | safety_spatial | Object bbox ∩ ego-lane polygon ≥25% for ≥2 frames (positive predicate, not a residual). | Legal occupancy of the ego lane (lead/behind); map-mode degraded with no pixel-band fallback. |
| pedestrian_cross | safety_spatial | Pedestrian or cyclist crosses the ego path or a marked/unmarked crossing in-window. | Pedestrian waiting on the curb with no trajectory into the roadway. |
| red_light_violation | compliance | EVENT: a relevant actor is in the intersection against a red facing signal. | Lamp-only (no actor); `signal_state` is an attribute, not the class. No signal GT on mini. |
| none | none | No listed scenario is clearly present (empty-per-taxonomy). | Do not combine with any positive class. Use uncertain (schema slot) when the definition is doubtful. |

## Eval vocabulary (Day 9 — not class cards)

Five terms from the claims-audit. Not labeling TRIGGER/EXCLUDE; do not use these as `class_set` values.

| term | TRIGGER (what it is) | EXCLUDE (what it is not) |
|---|---|---|
| policy | A dated, frozen rule in `configs/*.yaml` + pre-reg that binds a later table. Changing it is a version bump. | A vibe, a post-hoc threshold pick, or a README number that disagrees with `headline_metrics.csv`. |
| hallucination | **GT-bounded:** a class claimed on a GT window whose label is not that class (claim-audit raw counts). | A schema-forced none (`parse_ok=false`); a detector miss; an unsupported corpus claim with no GT. |
| frontier | The $0 Pareto of (recall, precision, cost) for the backends that actually ran. 3 points: detector / API-GT60 / union. | A fourth box-level VRU point; a corpus-API point that was never mined (D20). |
| pre-registered | The 8 load-bearing items frozen before the official table. Day-9 executed that table. | A fishing license on the tIoU sweep; redrawing GT because scene-0061 is 15/60. |
| negative-space | Coverage of windows the index *cannot* retrieve for a query (true-none + out-of-headline). | A precision number; a claim that "none" on GT-60 is model accuracy. |
