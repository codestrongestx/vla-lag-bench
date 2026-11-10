# FlashRT set-down: lazy at 50 ms with execution horizon 4

One point added to `flashrt_setdown_grid_20261018`: pi0.5 served by FlashRT
bf16 on set-down at 0.075 m/s, lazy at 50 ms, execution horizon 4. Lazy with
horizon 4 requests every 4 steps (200 ms) and executes actions 1-4 with no
idle step, which is continuous execution capped at 5 Hz. It pairs with the
grid's synchronous (77.5%) and lazy horizon-5 (77.0%) points at the same
speed and delay. Every other setting copies `lazy50_slow.yaml`.
