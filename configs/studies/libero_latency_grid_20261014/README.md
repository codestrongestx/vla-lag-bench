# LIBERO latency grid: 50, 150, 250 ms

pi0.5 on the four LIBERO tasks of the report at the delays between the
existing 0/100/200 ms points: synchronous, lazy and continuous at 50, 150 and
250 ms. Synchronous and lazy run first; continuous follows on the same GPUs.

| Task | Base experiment | Speeds (m/s) |
| --- | --- | --- |
| Moving basket | `libero_object_task1_native_basket_ellipse/basket_ellipse_paired_200.yaml` | 0.001, 0.025–0.2 by 0.025 |
| Moving drawer | `libero_goal_task0_moving_drawer/drawer_ellipse_paired_200.yaml` | 0.001, 0.025–0.2 by 0.025 |
| Moving plate | `libero_spatial_task9_bowl_to_plate/plate_ellipse_paired_200.yaml` | 0.001, 0.025–0.2 by 0.025 |
| Moving bowl | `libero_spatial_task9_bowl_to_plate/cabinet_ellipse_paired_200.yaml` (bowl centered) | 0.001, 0.05–0.4 by 0.05 |
