"""Rebuild the exact pi0.5 policy requests of recorded LIBERO episodes.

For each selected request of each trace, load the recorded MuJoCo state the
request was sent from, regenerate the camera observation, and apply the same
openpi LIBERO conventions the evaluator used. The output holds the request
elements (224 px images, state, prompt), their request identities (which fix
the diffusion noise), and the chunks the recording policy returned, so any
inference stack can be timed on identical inputs offline.

Run with the benchmark environment (it needs LIBERO and robosuite), from the
repository root:

    .venv/bin/python scripts/latency/policy_frames.py OUT.npz TRACE.npz [...] --per-episode 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlalagbench.benchmarks.libero.envs import make_env_from_trace
from vlalagbench.benchmarks.libero.evaluator import _policy_element
from vlalagbench.simulators.mujoco.tracing import EpisodeTrace, get_mj_sim, get_sim_state


def trace_requests(path: Path, per_episode: int, resize_size: int, policy_seed: int | None) -> list[dict]:
    trace = EpisodeTrace.load(path)
    meta = trace.meta
    extra = meta.model_extra or {}
    warmup = int(extra["num_steps_wait"])
    if "policy_seed" in extra:
        policy_seed = int(extra["policy_seed"])
    elif policy_seed is None:
        raise ValueError(f"{path} records no policy seed; pass --policy-seed")
    count = len(trace.request_steps)
    picks = np.unique(np.linspace(0, count - 1, min(per_episode, count)).round().astype(int))
    env = make_env_from_trace(meta)
    rows = []
    try:
        env.reset()
        restore_origins = getattr(env, "restore_actor_origins", None)
        if restore_origins is not None:
            restore_origins(trace.model_xml)
        env.set_init_state(trace.states[0])
        sim = get_mj_sim(env)
        apply_dynamic_time = getattr(env, "apply_dynamic_time", None)
        for request_id in picks:
            step = int(trace.request_steps[request_id])
            index = warmup + step
            sim.set_state_from_flattened(np.asarray(trace.states[index], dtype=np.float64))
            sim.forward()
            if apply_dynamic_time is not None:
                apply_dynamic_time(step)
            obs = env.regenerate_obs_from_state(get_sim_state(env))
            element = _policy_element(obs, meta.task_description, resize_size)
            rows.append({
                **element,
                "identity": {"policy_seed": policy_seed, "task_id": int(extra["task_id"]),
                             "episode_id": int(extra["episode_id"]), "request_id": int(request_id)},
                "recorded_chunk": np.asarray(trace.chunks[request_id]),
                "source": f"{path}#{request_id}",
            })
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--per-episode", type=int, default=8, help="requests per trace, evenly spaced")
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--policy-seed", type=int, help="for traces recorded before the policy seed was stored")
    args = parser.parse_args()
    rows = [row for path in args.traces
            for row in trace_requests(path, args.per_episode, args.resize_size, args.policy_seed)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        image=np.stack([row["observation/image"] for row in rows]),
        wrist_image=np.stack([row["observation/wrist_image"] for row in rows]),
        state=np.stack([row["observation/state"] for row in rows]),
        prompt=np.array([row["prompt"] for row in rows]),
        identity=np.array([json.dumps(row["identity"], sort_keys=True) for row in rows]),
        recorded_chunk=np.stack([row["recorded_chunk"] for row in rows]),
        source=np.array([row["source"] for row in rows]),
    )
    print(f"wrote {len(rows)} requests from {len(args.traces)} traces to {args.out}")


if __name__ == "__main__":
    main()
