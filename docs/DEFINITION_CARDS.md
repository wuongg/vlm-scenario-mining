# 90-second definition cards (1 / class)

Read TRIGGER then EXCLUDE. These cards are the gold for the Day-2+ anchor-alarm. Do not edit after the first real labeling session without a decision-commit.

| class | family | TRIGGER | EXCLUDE |
|---|---|---|---|
| cut_in | maneuver | Another vehicle enters the ego lane from an adjacent lane and becomes the lead, closing laterally then longitudinally. | Ego lane-change; already-ahead vehicle; parked vehicle that never occupies the ego path. |
| cut_out | maneuver | Lead vehicle leaves the ego lane, exposing a new lead or empty path. | Lead turning at an intersection while remaining in a dedicated turn lane that ego is not in. |
| lead_braking | maneuver | Lead vehicle decelerates sharply in-lane (closure slope on 1/h, not a single-frame jump). | Ego braking with a stable lead; stopped traffic already at rest for the whole window. |
| overtaking | maneuver | A vehicle passes ego or ego is passed, with a completed along-track order swap. | Parallel travel with no order swap; oncoming traffic. |
| lane_split_moto | spatial | Motorcycle straddles the lane boundary (≥3 frames, moving) — VN "xe máy cắt làn". | Motorcycle fully inside one lane; bicycle; parked two-wheeler on the shoulder. |
| lane_intrusion | spatial | Object bbox intersects the ego-lane polygon ≥25% for ≥2 frames (positive predicate). | Legal occupancy of the ego lane (lead/behind); map-mode degraded without a lane polygon AND no pixel-band fallback fired. |
| sidewalk_mount | spatial | ≥50% of the object bbox sits on walkway (Mode A) or non-road mask (Mode B). | Pedestrian on sidewalk; vehicle legally in a driveway cut. |
| ped_cross | vru | Pedestrian or cyclist crosses the ego path or a marked/unmarked crossing in-window. | Pedestrian waiting on the curb with no trajectory into the roadway. |
| red_light | signal | A relevant actor (ego or other) is in the intersection while the facing signal is red. | No GT for signal state on nuScenes mini — do not score as a VLM miss; `signal_state=unclear` when the lamp is unreadable. |
| none | none | No listed scenario is clearly present in the window. | Do not combine with any positive class. Use uncertain instead of none when the definition is doubtful. |
