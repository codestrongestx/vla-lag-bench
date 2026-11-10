# Task 9 moving cabinet, bowl centered

Rerun of the LIBERO-Spatial Task 9 moving-cabinet sweeps for pi0.5 and GR00T
after commit `a21005b4`, which pins the bowl at the center of the cabinet top
in all 50 states of `cabinet_ellipse_paired_200.yaml`. Before that commit the
bowl kept its official xy, 0.2–5.3 cm from the top center, so the carried
bowl path varied between episodes.

Nine speeds (0.001, 0.05–0.4 by 0.05 m/s); synchronous at 0/100/200 ms, lazy
and continuous at 100/200 ms: 63 points × 200 episodes per model, 25,200
episodes in total.

The specs copy the H100 cabinet specs unchanged except for the study names:
`libero_spatial_task9_20261001/pi05_b32/cabinet_*.yaml` (pi0.5, batched up to
32, max-autotune) and `libero_spatial_task9_20261001/gr00t/cabinet_*.yaml`
(GR00T N1.7 `libero_spatial`, batched up to 32, eager). Seeds 7/7/7.
