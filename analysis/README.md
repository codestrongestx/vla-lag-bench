# Analysis

Reproduces the paper's 24 figures and its analysis-derived numbers from the
published success tables in `results/` and the derived tables shipped in
`analysis/data/`. No simulator or policy runs.

```sh
python -m analysis.reproduce                      # all figures, then all numbers (about 20 minutes on a laptop CPU)
python -m analysis.reproduce pace tolerance-zone  # selected figures
python -m analysis.reproduce --compare PAPER/generated/board-figures   # pixel check against the paper's PNGs
python -m analysis.numbers model deployment       # selected number sections only
python -m unittest discover -s analysis/tests -t .
```

Figures go to `analysis/build/figures/<name>.png` (`--out DIR` or `ANALYSIS_OUTPUT`).
Needs numpy, scipy, matplotlib, pillow, and pyyaml; the three scene figures and
the measurements also need mujoco and opencv, as in the benchmark's environment.

## Figures

| Figure | Renderer (`analysis/figures/`) | Inputs |
| --- | --- | --- |
| `overview` | `overview.py`, `overview_figure/` | recorded frames (see below); needs tectonic, pdftoppm, and Helvetica Neue |
| `vla-architecture`, `action-chunking` | `action_chunking.py` | `configs/experiments/libero/pi05_libero_goal_task0_2x_lazy_100ms.yaml`, `src/vlalagbench/timing.py` |
| `task-overview` | `task_overview.py` | recorded frames, traces, run configurations |
| `dynamicvla-task-overview` | `task_overview.py --dynamicvla` | recorded frames, run configurations |
| `schedule-quantities` | `execution_modes.py` | `configs/experiments/libero/pi05_libero_goal_task0_2x_lazy_100ms.yaml`, `src/vlalagbench/timing.py` |
| `task-success`, `success-grid`, `gr00t-success-grid`, `dynamicvla-success-grid`, `prediction-summary`, `prediction-examples`, `molmoact2-prediction-examples` | `success_grid.py` | episode tables; model fits (`latency_model/fits.py`) |
| `mode-summary`, `mode-comparison`, `molmoact2-mode-comparison`, `dynamicvla-mode-comparison` | `mode_comparison.py` | episode tables |
| `policy-summary` | `policy_summary.py` | episode tables (pi0.5, GR00T N1.7, MolmoAct2) |
| `policy` | `policy_comparison.py` | episode tables (pi0.5, GR00T N1.7, MolmoAct2) |
| `tolerance-zone`, `dynamicvla-tolerance-zone` | `tolerance_zone.py` (side views: `stage_views.py`) | `basket_releases/attempts`, `plate_cabinet_replay/attempts`, `dynamicvla_replay/attempts`; the side views also need traces and the full `basket-alignment` and `plate-cabinet-episodes` outputs |
| `pace` | `pace.py` | `basket_alignment_speed`, `drawer_alignment_speed`, `plate_cabinet_replay/alignment` |
| `deployment-modes`, `deployment-rate` | `deployment.py` | set-down and grasping episode tables, the pi0.5 inference-stack tables, and the `pi05-horizon-validation` tables in `results/` |

Synchronous settings at a nonzero latency on the four LIBERO tasks and both
DynamicVLA tasks come from the hold_pose reruns (`results/*/*-sync-hold-pose/`),
in which the arm holds its pose while it waits (`analysis/sync_hold.py`). The
episode tables read those points from the reruns' own episode records where the
tables are built from published results (`libero_episodes`, `plate_cabinet_episodes`,
`dynamicvla_episodes`); `latency_grid_episodes` and `molmoact2_episodes` keep each
study's original records, and their synchronous successes are replaced by the
reruns' totals when loaded, as in the paper.

Episode tables: `libero_episodes/{pi05,gr00t}` (release, articulated),
`plate_cabinet_episodes` (set-down, grasping), `latency_grid_episodes` (pi0.5
at 50/150/250 ms), `molmoact2_episodes`, `dynamicvla_episodes`; replay tables:
`dynamicvla_replay/attempts`.

## Numbers

`python -m analysis.numbers` prints, by section:

- `crossings`: Finding 1 and Appendix B: pi0.5's zero-latency 50% speeds
  (0.04 to 0.19 m/s) and success at 0.2 m/s (5.0 to 46.0%, against 93.0 to 98.5%
  near static); set-down at 0.1 m/s and 100 ms (71.5 to 40.5%) and grasping
  (86.5 to 74.0%); DynamicVLA's 50% speeds at 240 ms (release 0.21 to 0.13,
  grasping 0.080 to 0.047 m/s).
- `tolerance`: tolerance zones (set-down 0 to 3.0 cm; release 0 to 7.0 cm,
  against the 12.2 cm containment region).
- `hand-speed`: median alignment hand speed per setting; release slows 34% at
  synchronous 200 ms.
- `model`: Table `tab:sensitivities` (pi0.5, MolmoAct2, GR00T N1.7) and Table
  `tab:blind` (5.7 points for pi0.5, 7.8 for MolmoAct2); the held-out error by
  latency, the noise floor (4.0 and 3.9 points), and where speed scaling
  departs; the 50 ms horizons H = 2 to 4 (3.3 points on set-down, 2.8 on
  release, from the `pi05-horizon-validation` tables); DynamicVLA's fits
  (grasping a 0.16, b 1.37; release a 0.38, b 0.38).
- `calibration`: Table `tab:two-point` (pi0.5 median 7.7 points, 80% within
  4.5 to 14.4, the unseen zero-latency curve 6.8; MolmoAct2 11.1; GR00T N1.7
  11.6; pi0.5 with both points at any speed pair 9.4).
- `deployment`: Table `tab:latency`, Table `tab:deployment-error` (133 points,
  4.2 RMS), Table `tab:deployment-systems` (Systems A to D, predicted and
  measured), and the fact check (36 of 45 cells counting ties).

The model and calibration sections take about 5 to 10 minutes on a laptop CPU.

`python -m analysis.latency_model` is the model's own command line (`fit`,
`predict`, `probe`, `transfer`, `next`, `validate [--policy MolmoAct2]`); see
`latency_model/speed_latency_model.py`.

## Derived tables and raw inputs

`analysis/data/` holds the derived tables as gzip CSV (9.8 MB) with
`MANIFEST.json` (rows, SHA-256); `reproduce` checks them first. Each comes from
a measurement over benchmark run records and MuJoCo traces:

```sh
export VLALAGBENCH_ARTIFACTS=/path/to/artifacts     # holds studies/<study>/points/...
export VLALAGBENCH_CHECKOUT=/path/to/benchmark      # third_party/ with the LIBERO assets the traces reference
python -m analysis.measure --list
python -m analysis.measure plate-cabinet-episodes plate-cabinet-replay --workers 8
```

A measurement writes its full outputs and provenance to `analysis/build/derived/`
and exports the figure columns to `analysis/data/`. `plate-cabinet-replay` reads
the full `plate-cabinet-episodes` output and `basket-releases` the full
`basket-alignment` output, so run those first. Rebuilt from the raw files, every
shipped table reproduces byte for byte, and each equals the table the paper's
figures were rendered from.

The three scene figures read, under `VLALAGBENCH_ARTIFACTS/studies/`, the files
listed with their SHA-256 in `SOURCES` and `DYNAMICVLA_SOURCES`
(`figures/task_overview.py`): four LIBERO episodes (agentview video, trace
`.npz`, `experiment.yaml`) and two DynamicVLA episodes (video,
`experiment.yaml`). Without them `reproduce` skips these figures.
