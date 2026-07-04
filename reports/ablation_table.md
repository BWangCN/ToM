| arm | pathway | chunk L2 vs GT | rollout deceived | rollout goal | perturb ratio | open-loop ms/step | rollout ms/step | notes |
|---|---|---|---|---|---|---|---|---|
| (a) | discrete argmax | 1.9727 | 0.45 | 0.0 | shift=2.671 (det.) | 173.7 | 188.7 | — |
| (b) | flow, cond=think+act | 2.0234 | 0.4 | 0.0 | 10.044 | 197.9 | 206.0 | — |
| (c) | flow, cond=think | 2.3029 | 0.65 | 0.0 | 2.045 | 216.2 | 210.5 | emit_discrete=false: 157.2 ms/step |
