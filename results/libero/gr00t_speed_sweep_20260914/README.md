# GR00T N1.7 moving-target speed sweep — RTX 3090

The paper uses the basket half of this sweep; its drawer half used the original
cabinet placement and was superseded by `../20260926_moving-drawer-fixed/gr00t/`.

The figures and CSVs in this directory now use **only the complete RTX 3090 server rerun**, executed September 15–17, 2026. They replace an earlier mixed-GPU execution; no results from it are pooled into these plots.

- Moving basket: [figure](../20260908_moving-basket/gr00t/success_rates.png) · [data](../20260908_moving-basket/gr00t/plot_data.csv) · [provenance](../20260908_moving-basket/gr00t/provenance.yaml)
- [Results and verification summary](results.yaml)

All 144 points (72 per task) completed 200 episodes each: **28,800 episodes**. Every run records NVIDIA GeForce RTX 3090 hardware. Each episode uses a 600-active-step deadline at 20 Hz, execution horizon 5, and predicted action chunks of 16. Speeds span 0.001–0.2 m/s; 0.001 m/s is a slow-motion anchor, not a static baseline. Synchronous and lazy inference use 0, 100, and 200 ms simulated latency; continuous inference uses 100 and 200 ms. Error bars are pointwise 95% Wilson intervals. The synchronous 100 and 200 ms points are not published: they were rerun with the arm held still during waits (`../20260908_moving-basket/gr00t-sync-hold-pose/`).

The four frozen [study specifications](../../../configs/studies/gr00t_speed_sweep_20260914/) retain the original settings and seeds. The full audit verified all 28,800 NPZ checksums, episode IDs 0–199 for every point, recorded settings, success totals, and the active-step limit. No episode errors or dirty source records were found. Run IDs, source revisions, and execution times accompany each task's CSV in its provenance YAML.
