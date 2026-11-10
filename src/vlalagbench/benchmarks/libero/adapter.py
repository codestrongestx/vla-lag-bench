"""LIBERO implementation of the benchmark adapter contract."""

from __future__ import annotations

from pathlib import Path

from vlalagbench.benchmarks.base import EpisodeOutput
from vlalagbench.benchmarks.libero.envs import require_supported_control_hz
from vlalagbench.benchmarks.libero.evaluator import EvalConfig, run_eval
from vlalagbench.benchmarks.libero.policy_server import batching_identity, validate_runtime
from vlalagbench.config import Experiment, Runtime


class LiberoAdapter:
    def validate(self, experiment: Experiment, runtime: Runtime) -> None:
        if not isinstance(experiment, Experiment) or not isinstance(runtime, Runtime):
            raise TypeError("libero experiments require a LIBERO runtime")
        validate_runtime(runtime)
        require_supported_control_hz(experiment.benchmark.control_hz)

    def evaluate(
        self,
        experiment: Experiment,
        runtime: Runtime,
        artifact_dir: Path,
    ) -> list[EpisodeOutput]:
        raw = run_eval(_eval_config(experiment, runtime, artifact_dir))
        return [
            {
                **episode,
                "seed": episode.get("environment_seed", experiment.seeds.environment),
            }
            for episode in raw["episodes"]
        ]


def _eval_config(experiment: Experiment, runtime: Runtime, artifact_dir: Path) -> EvalConfig:
    validate_runtime(runtime)
    task_ids = None if experiment.benchmark.task_ids == "all" else experiment.benchmark.task_ids
    return EvalConfig(
        task_suite=experiment.benchmark.suite,
        task_ids=task_ids,
        trials_per_task=experiment.benchmark.episodes_per_task,
        episode_ids=experiment.benchmark.episode_ids,
        model=experiment.model,
        openpi_config=getattr(experiment.model, "openpi_config", ""),
        checkpoint=experiment.model.checkpoint,
        mode=experiment.inference.mode,
        host=runtime.policy_server.host,
        port=runtime.policy_server.port,
        action_chunk_steps=experiment.inference.action_chunk_steps,
        execution_horizon_steps=experiment.inference.execution_horizon_steps,
        latency_steps=experiment.effective_latency_steps,
        unavailable_action=experiment.inference.unavailable_action,
        dynamics=experiment.dynamics,
        dynamics_by_task=experiment.dynamics_by_task,
        trajectory=experiment.trajectory,
        dynamics_seed=experiment.seeds.dynamics,
        num_steps_wait=experiment.benchmark.warmup_steps,
        max_steps=experiment.benchmark.max_episode_steps,
        initial_state_indices=experiment.benchmark.initial_state_indices,
        initial_state_overrides=experiment.initial_state_overrides,
        seed=experiment.seeds.environment,
        policy_seed=experiment.seeds.policy,
        pytorch_compile_mode=runtime.policy_server.pytorch_compile_mode,
        deterministic_ops=runtime.policy_server.deterministic_ops,
        batching=batching_identity(runtime.policy_server),
        workers=runtime.environment.workers,
        process_start_method=runtime.environment.process_start_method,
        render_policy_observations_only=runtime.environment.render_policy_observations_only,
        dynamic_episode_queue=runtime.environment.dynamic_episode_queue,
        out_dir=artifact_dir,
    )


ADAPTER = LiberoAdapter()
