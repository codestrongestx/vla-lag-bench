"""FlashRT bf16 serving pi0.5 on set-down: every speed and delay of the openpi grid.

Fifteen studies `flashrt_setdown_grid_20261018`: nine speeds; synchronous at 0-250 ms and lazy and continuous at
50-250 ms; 200 episodes per setting; FlashRT 0.2.0 bf16 on H100s, every batched pass padded to 16 rows; the pi0.5
LIBERO checkpoint converted to PyTorch. The published table results/libero/20261001_task9-moving-plate/
pi05-flashrt-bf16/plot_data.csv holds the verified success counts of all 144 settings.
"""
import csv

from analysis.paths import RESULTS

TABLE = RESULTS / 'libero/20261001_task9-moving-plate/pi05-flashrt-bf16/plot_data.csv'


def counts():
    """(speed, mode, latency) -> [successes, episodes] for every setting of the set-down grid (K = 10, H = 5)."""
    out = {}
    with TABLE.open() as stream:
        for row in csv.DictReader(stream):
            if (row['action_chunk_steps'], row['execution_horizon_steps']) != ('10', '5'):
                continue
            key = (float(row['trajectory.speed_mps']), row['inference.mode'], int(row['inference.latency_ms']))
            if key in out:
                raise ValueError(f'{TABLE}: duplicate setting {key}')
            out[key] = [int(row['successes']), int(row['episodes'])]
    if len(out) != 144 or any(n != 200 for _, n in out.values()):
        raise ValueError(f'FlashRT set-down grid incomplete in {TABLE}: {len(out)} settings')
    return out
