# FlashRT bf16 on set-down: every speed and delay

pi0.5 served by FlashRT bf16 instead of openpi's JAX model on set-down
(`libero_spatial_task9_bowl_to_plate/plate_ellipse_paired_200.yaml`), on the
same grid as the openpi pi0.5 runs: speeds 0.001 and 0.025–0.2 m/s by 0.025;
synchronous at 0, 50, 100, 150, 200 and 250 ms; lazy and continuous at 50–250
ms. 15 studies, 144 points, 28,800 episodes; 200 episodes per point over the 50
official states with the four phase/direction conditions; 20 Hz, 10 warm-up
steps, chunks of 10, execution horizon 5, 600 acting steps, seeds 7/7/7.

Serving: one batched FlashRT server per GPU, every pass padded to 16 rows
(`batch_padding: fixed`), so a request's actions do not depend on its
batch-mates; checkpoint `$VLALAGBENCH_CHECKPOINTS/pi05_libero_pytorch`;
`FVK_RTX_FA2=0`, since FlashRT builds no FA2 module for the H100 and
attention runs through pip flash_attn. 48 workers per GPU, render gate and
dynamic episode queue.
