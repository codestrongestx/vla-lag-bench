# LIBERO-10 and LIBERO-90, official initial states

Static LIBERO-10 (10 tasks) and LIBERO-90 (90 tasks), synchronous inference
at 0 ms latency, 50 episodes per task (one per official initial state), seeds
7/7/7, step limits 520 (LIBERO-10) and 400 (LIBERO-90) after 10 warmup steps. Specs: `configs/studies/libero_10_90_official_20261013`. Run on H100 GPUs
6–7 with batched serving, 01:55–04:55 UTC on October 13, 2026; every study
passed `study verify` with no episode errors.

Neither model was trained on LIBERO-90: `pi05_libero` was trained on the
LIBERO-10, Goal, Object and Spatial demos, and GR00T-N1.7-LIBERO has one
checkpoint per suite and none for LIBERO-90. GR00T's LIBERO-90 runs use the
`libero_10` checkpoint, which a 450-episode pilot picked over the other three
(16.0% vs 1.1–1.3%; `gr00t-libero90-checkpoint-pilot/pilot_summary.yaml`).

| Suite | pi0.5 | GR00T N1.7 |
| --- | --- | --- |
| LIBERO-10 | 464/500 (92.8%) | 446/500 (89.2%) |
| LIBERO-90 | 1,275/4,500 (28.3%) | 754/4,500 (16.8%) |

On LIBERO-90, pi0.5 never succeeds on 43 tasks and GR00T on 61; tasks close to
the trained suites (basket and tray pick-and-place, plate, stove, microwave)
carry almost all successes. The hardest LIBERO-10 task for both is task 8,
"put both moka pots on the stove" (62% and 60%).

- `pi05/`, `gr00t/`: `per_task.csv` (successes and 95% Wilson interval per
  task) and `per_task_summary.yaml` (run IDs, study hashes, episode digests).
- `pi05-vs-gr00t/`: one per-task figure per suite.
- `gr00t-libero90-checkpoint-pilot/`: pilot totals per GR00T checkpoint.

## Per-task success

Successes out of 50 per task, ranked by the mean of the two models (high to low); Δ is pi0.5 minus GR00T in points.

### LIBERO-10

| Rank | Task | Instruction | pi0.5 | GR00T | Mean | Δ |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | put both the cream cheese box and the butter in the basket | 50 (100%) | 49 (98%) | 99% | +2 |
| 2 | 7 | put both the alphabet soup and the cream cheese box in the basket | 50 (100%) | 47 (94%) | 97% | +6 |
| 3 | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 48 (96%) | 49 (98%) | 97% | -2 |
| 4 | 2 | turn on the stove and put the moka pot on it | 48 (96%) | 48 (96%) | 96% | +0 |
| 5 | 5 | pick up the book and place it in the back compartment of the caddy | 48 (96%) | 48 (96%) | 96% | +0 |
| 6 | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 49 (98%) | 45 (90%) | 94% | +8 |
| 7 | 9 | put the yellow and white mug in the microwave and close it | 49 (98%) | 42 (84%) | 91% | +14 |
| 8 | 0 | put both the alphabet soup and the tomato sauce in the basket | 46 (92%) | 45 (90%) | 91% | +2 |
| 9 | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 45 (90%) | 43 (86%) | 88% | +4 |
| 10 | 8 | put both moka pots on the stove | 31 (62%) | 30 (60%) | 61% | +2 |

### LIBERO-90 (GR00T `libero_10` checkpoint)

| Rank | Task | Instruction | pi0.5 | GR00T | Mean | Δ |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 24 | put the black bowl in the bottom drawer of the cabinet | 50 (100%) | 49 (98%) | 99% | +2 |
| 2 | 22 | close the bottom drawer of the cabinet | 50 (100%) | 47 (94%) | 97% | +6 |
| 3 | 46 | pick up the alphabet soup and put it in the basket | 49 (98%) | 46 (92%) | 95% | +6 |
| 4 | 20 | turn on the stove | 45 (90%) | 49 (98%) | 94% | -8 |
| 5 | 67 | put the white mug on the left plate | 49 (98%) | 43 (86%) | 92% | +12 |
| 6 | 47 | pick up the cream cheese box and put it in the basket | 50 (100%) | 40 (80%) | 90% | +20 |
| 7 | 68 | put the yellow and white mug on the right plate | 50 (100%) | 39 (78%) | 89% | +22 |
| 8 | 51 | pick up the butter and put it in the basket | 50 (100%) | 37 (74%) | 87% | +26 |
| 9 | 33 | close the microwave | 47 (94%) | 40 (80%) | 87% | +14 |
| 10 | 57 | pick up the cream cheese and put it in the tray | 49 (98%) | 35 (70%) | 84% | +28 |
| 11 | 19 | put the moka pot on the stove | 46 (92%) | 34 (68%) | 80% | +24 |
| 12 | 56 | pick up the butter and put it in the tray | 41 (82%) | 37 (74%) | 78% | +8 |
| 13 | 72 | put the white mug on the plate | 48 (96%) | 29 (58%) | 77% | +38 |
| 14 | 60 | pick up the black bowl on the left and put it in the tray | 27 (54%) | 49 (98%) | 76% | -44 |
| 15 | 70 | put the chocolate pudding to the right of the plate | 36 (72%) | 38 (76%) | 74% | -4 |
| 16 | 55 | pick up the alphabet soup and put it in the tray | 50 (100%) | 23 (46%) | 73% | +54 |
| 17 | 54 | pick up the tomato sauce and put it in the basket | 42 (84%) | 19 (38%) | 61% | +46 |
| 18 | 50 | pick up the alphabet soup and put it in the basket | 46 (92%) | 14 (28%) | 60% | +64 |
| 19 | 9 | put the black bowl on the plate | 48 (96%) | 8 (16%) | 56% | +80 |
| 20 | 28 | close the top drawer of the cabinet | 19 (38%) | 32 (64%) | 51% | -26 |
| 21 | 61 | pick up the chocolate pudding and put it in the tray | 50 (100%) | 0 (0%) | 50% | +100 |
| 22 | 10 | put the black bowl on top of the cabinet | 47 (94%) | 0 (0%) | 47% | +94 |
| 23 | 2 | put the black bowl in the top drawer of the cabinet | 40 (80%) | 0 (0%) | 40% | +80 |
| 24 | 29 | put the black bowl in the top drawer of the cabinet | 31 (62%) | 7 (14%) | 38% | +48 |
| 25 | 30 | put the black bowl on the plate | 35 (70%) | 0 (0%) | 35% | +70 |
| 26 | 64 | stack the right bowl on the left bowl and place them in the tray | 21 (42%) | 9 (18%) | 30% | +24 |
| 27 | 38 | put the right moka pot on the stove | 19 (38%) | 8 (16%) | 27% | +22 |
| 28 | 82 | pick up the book and place it in the left compartment of the caddy | 26 (52%) | 0 (0%) | 26% | +52 |
| 29 | 59 | pick up the tomato sauce and put it in the tray | 13 (26%) | 10 (20%) | 23% | +6 |
| 30 | 12 | put the black bowl at the back on the plate | 22 (44%) | 0 (0%) | 22% | +44 |
| 31 | 79 | pick up the book and place it in the left compartment of the caddy | 18 (36%) | 4 (8%) | 22% | +28 |
| 32 | 14 | put the middle black bowl on the plate | 18 (36%) | 0 (0%) | 18% | +36 |
| 33 | 77 | pick up the book and place it in the back compartment of the caddy | 10 (20%) | 0 (0%) | 10% | +20 |
| 34 | 31 | put the black bowl on top of the cabinet | 7 (14%) | 2 (4%) | 9% | +10 |
| 35 | 0 | close the top drawer of the cabinet | 4 (8%) | 4 (8%) | 8% | +0 |
| 36 | 43 | put the white bowl on top of the cabinet | 5 (10%) | 0 (0%) | 5% | +10 |
| 37 | 53 | pick up the orange juice and put it in the basket | 3 (6%) | 0 (0%) | 3% | +6 |
| 38 | 44 | turn on the stove | 2 (4%) | 1 (2%) | 3% | +2 |
| 39 | 26 | put the wine bottle in the bottom drawer of the cabinet | 2 (4%) | 0 (0%) | 2% | +4 |
| 40 | 73 | pick up the book and place it in the front compartment of the caddy | 2 (4%) | 0 (0%) | 2% | +4 |
| 41 | 74 | pick up the book and place it in the left compartment of the caddy | 2 (4%) | 0 (0%) | 2% | +4 |
| 42 | 7 | open the top drawer of the cabinet | 1 (2%) | 0 (0%) | 1% | +2 |
| 43 | 11 | open the top drawer of the cabinet | 1 (2%) | 0 (0%) | 1% | +2 |
| 44 | 13 | put the black bowl at the front on the plate | 1 (2%) | 0 (0%) | 1% | +2 |
| 45 | 15 | put the middle black bowl on top of the cabinet | 1 (2%) | 0 (0%) | 1% | +2 |
| 46 | 41 | put the frying pan on top of the cabinet | 1 (2%) | 0 (0%) | 1% | +2 |
| 47 | 63 | stack the left bowl on the right bowl and place them in the tray | 1 (2%) | 0 (0%) | 1% | +2 |
| 48 | 52 | pick up the milk and put it in the basket | 0 (0%) | 1 (2%) | 1% | -2 |
| 49 | 1 | close the top drawer of the cabinet and put the black bowl on top of it | 0 (0%) | 0 (0%) | 0% | +0 |
| 50 | 3 | put the butter at the back in the top drawer of the cabinet and close it | 0 (0%) | 0 (0%) | 0% | +0 |
| 51 | 4 | put the butter at the front in the top drawer of the cabinet and close it | 0 (0%) | 0 (0%) | 0% | +0 |
| 52 | 5 | put the chocolate pudding in the top drawer of the cabinet and close it | 0 (0%) | 0 (0%) | 0% | +0 |
| 53 | 6 | open the bottom drawer of the cabinet | 0 (0%) | 0 (0%) | 0% | +0 |
| 54 | 8 | open the top drawer of the cabinet and put the bowl in it | 0 (0%) | 0 (0%) | 0% | +0 |
| 55 | 16 | stack the black bowl at the front on the black bowl in the middle | 0 (0%) | 0 (0%) | 0% | +0 |
| 56 | 17 | stack the middle black bowl on the back black bowl | 0 (0%) | 0 (0%) | 0% | +0 |
| 57 | 18 | put the frying pan on the stove | 0 (0%) | 0 (0%) | 0% | +0 |
| 58 | 21 | turn on the stove and put the frying pan on it | 0 (0%) | 0 (0%) | 0% | +0 |
| 59 | 23 | close the bottom drawer of the cabinet and open the top drawer | 0 (0%) | 0 (0%) | 0% | +0 |
| 60 | 25 | put the black bowl on top of the cabinet | 0 (0%) | 0 (0%) | 0% | +0 |
| 61 | 27 | put the wine bottle on the wine rack | 0 (0%) | 0 (0%) | 0% | +0 |
| 62 | 32 | put the ketchup in the top drawer of the cabinet | 0 (0%) | 0 (0%) | 0% | +0 |
| 63 | 34 | put the yellow and white mug to the front of the white mug | 0 (0%) | 0 (0%) | 0% | +0 |
| 64 | 35 | open the microwave | 0 (0%) | 0 (0%) | 0% | +0 |
| 65 | 36 | put the white bowl on the plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 66 | 37 | put the white bowl to the right of the plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 67 | 39 | turn off the stove | 0 (0%) | 0 (0%) | 0% | +0 |
| 68 | 40 | put the frying pan on the cabinet shelf | 0 (0%) | 0 (0%) | 0% | +0 |
| 69 | 42 | put the frying pan under the cabinet shelf | 0 (0%) | 0 (0%) | 0% | +0 |
| 70 | 45 | turn on the stove and put the frying pan on it | 0 (0%) | 0 (0%) | 0% | +0 |
| 71 | 48 | pick up the ketchup and put it in the basket | 0 (0%) | 0 (0%) | 0% | +0 |
| 72 | 49 | pick up the tomato sauce and put it in the basket | 0 (0%) | 0 (0%) | 0% | +0 |
| 73 | 58 | pick up the ketchup and put it in the tray | 0 (0%) | 0 (0%) | 0% | +0 |
| 74 | 62 | pick up the salad dressing and put it in the tray | 0 (0%) | 0 (0%) | 0% | +0 |
| 75 | 65 | put the red mug on the left plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 76 | 66 | put the red mug on the right plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 77 | 69 | put the chocolate pudding to the left of the plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 78 | 71 | put the red mug on the plate | 0 (0%) | 0 (0%) | 0% | +0 |
| 79 | 75 | pick up the book and place it in the right compartment of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 80 | 76 | pick up the yellow and white mug and place it to the right of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 81 | 78 | pick up the book and place it in the front compartment of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 82 | 80 | pick up the book and place it in the right compartment of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 83 | 81 | pick up the book and place it in the front compartment of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 84 | 83 | pick up the book and place it in the right compartment of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 85 | 84 | pick up the red mug and place it to the right of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 86 | 85 | pick up the white mug and place it to the right of the caddy | 0 (0%) | 0 (0%) | 0% | +0 |
| 87 | 86 | pick up the book in the middle and place it on the cabinet shelf | 0 (0%) | 0 (0%) | 0% | +0 |
| 88 | 87 | pick up the book on the left and place it on top of the shelf | 0 (0%) | 0 (0%) | 0% | +0 |
| 89 | 88 | pick up the book on the right and place it on the cabinet shelf | 0 (0%) | 0 (0%) | 0% | +0 |
| 90 | 89 | pick up the book on the right and place it under the cabinet shelf | 0 (0%) | 0 (0%) | 0% | +0 |

## Archive

Clips are three successes and three failures per
task from agentview and robot0_eye_in_hand at 256 px; a task with fewer than
three of an outcome has all of them.
