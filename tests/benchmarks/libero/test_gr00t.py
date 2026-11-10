import numpy as np
import pytest

from vlalagbench.config import Gr00tModelSpec, ModelSpec
from vlalagbench.benchmarks.libero.policy import validate_actions
from vlalagbench.benchmarks.libero.policy_server import METADATA_KEY, validate_metadata


def model():
    return Gr00tModelSpec(backend="gr00t", checkpoint="nvidia/GR00T-N1.7-LIBERO",
                          revision="2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21",
                          checkpoint_subdir="libero_spatial")


def test_legacy_model_serialization():
    value = {"openpi_config": "pi05_libero", "checkpoint": "checkpoint"}
    assert ModelSpec.model_validate(value).model_dump() == value
    assert Gr00tModelSpec.model_validate(model().model_dump()) == model()


def test_gr00t_handshake_rejects_wrong_checkpoint():
    from vlalagbench.benchmarks.libero.gr00t_adapter import UPSTREAM_REVISION
    from vlalagbench.benchmarks.libero.policy import RNG_SCHEME
    identity = {"upstream_revision": UPSTREAM_REVISION, "rng_scheme": RNG_SCHEME, "files_sha256": {"processor_config.json": "a" * 64},
                "backbone_processors_sha256": {f"{name}/tokenizer.json": "a" * 64 for name in ("observation", "collator")},
                **model().model_dump(), "action_chunk_steps": 16, "implementation": "single",
                "precision": "bfloat16", "inference": "eager", "max_batch_size": 1, "deterministic_ops": False}
    kwargs = dict(model=model(), openpi_config="", checkpoint=model().checkpoint, action_chunk_steps=16)
    validate_metadata({METADATA_KEY: identity}, **kwargs)
    for key, value in [("revision", "a" * 40), ("action_chunk_steps", 10), ("precision", "float32"), ("backbone_processors_sha256", {})]:
        with pytest.raises(RuntimeError, match="handshake"):
            validate_metadata({METADATA_KEY: {**identity, key: value}}, **kwargs)


def test_backbone_preflight_follows_local_link_without_hub_access(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from vlalagbench.benchmarks.libero.gr00t_adapter import check_backbone_access

    monkeypatch.chdir(tmp_path)
    hub_calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda *args: hub_calls.append(args)))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_name": "local-backbone"}))
    link = tmp_path / "local-backbone"
    identities = []
    for revision in ("a" * 40, "b" * 40):
        snapshot = tmp_path / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps({"revision": revision}))
        link.unlink(missing_ok=True)
        link.symlink_to(snapshot, target_is_directory=True)
        identities.append(check_backbone_access(checkpoint))
    assert not hub_calls
    assert identities[0] != identities[1]
    assert identities[1]["local-backbone"]["revision"] == "b" * 40


def test_loaded_processor_identity_covers_tokenizer_images_and_templates():
    from types import SimpleNamespace
    from vlalagbench.benchmarks.libero.gr00t_adapter import backbone_processor_identity

    class Processor:
        def __init__(self):
            self.files = {name: b"original" for name in ("tokenizer.json", "merges.txt", "preprocessor_config.json", "chat_template.jinja")}

        def save_pretrained(self, directory):
            directory.mkdir()
            for name, content in self.files.items():
                (directory / name).write_bytes(content)

    observation, collator = Processor(), Processor()
    policy = SimpleNamespace(processor=SimpleNamespace(processor=observation), collate_fn=SimpleNamespace(processor=collator))
    original = backbone_processor_identity(policy)
    assert original == backbone_processor_identity(policy)
    for processor in (observation, collator):
        for name in processor.files:
            processor.files[name] = b"changed"
            assert original != backbone_processor_identity(policy)
            processor.files[name] = b"original"


@pytest.mark.parametrize("actions", [np.zeros((10, 7)), np.zeros((16, 8)), np.full((16, 7), np.nan), np.ones((16, 7), dtype=complex)])
def test_reject_malformed_chunk(actions):
    with pytest.raises(ValueError, match="finite"):
        validate_actions(actions, 16)


def observation():
    return {"agentview_image": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
            "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([1., 2., 3.]),
            "robot0_eef_quat": np.array([0., 0., np.sin(.25), np.cos(.25)]),
            "robot0_gripper_qpos": np.array([.01, -.01])}


def test_native_observation_and_gripper_conventions():
    from vlalagbench.benchmarks.libero.gr00t_adapter import AXES, decode_actions, format_observation
    obs = observation()
    formatted = format_observation(obs, "pick up the bowl")
    image = formatted["video"]["image"]
    assert image.shape == (1, 1, 256, 256, 3)
    assert image.flags.c_contiguous
    np.testing.assert_array_equal(image[0, 0], obs["agentview_image"][::-1, ::-1])
    np.testing.assert_allclose(formatted["state"]["yaw"], [[[.5]]])
    np.testing.assert_allclose(formatted["state"]["gripper"], [[[.01, -.01]]])
    assert formatted["language"]["annotation.human.action.task_description"] == [["pick up the bowl"]]
    actions = {key: np.full((1, 16, 1), i, dtype=np.float32) for i, key in enumerate(AXES)}
    actions["gripper"] = np.resize([0., .5, 1.], (1, 16, 1))
    chunk = decode_actions(actions)
    np.testing.assert_array_equal(chunk[:, :6], np.tile(np.arange(6), (16, 1)))
    np.testing.assert_array_equal(chunk[:3, -1], [1, 0, -1])
    assert actions["gripper"][0, 0, 0] == 0


def test_rng_isolation_repeated_interleaved_requests_and_failure():
    import random
    import torch
    from vlalagbench.benchmarks.libero.gr00t_adapter import AXES, Gr00tRequestPolicy, isolated_rng
    from vlalagbench.benchmarks.libero.policy import REQUEST_KEY, validate_response_identity

    class PublicPolicy:
        def get_action(self, obs):
            value = random.random() + np.random.rand() + torch.rand(1).item()
            return {key: np.full((1, 16, 1), value) for key in (*AXES, "gripper")}, {}

    policy = Gr00tRequestPolicy(PublicPolicy())
    identity = {"policy_seed": 7, "task_id": 2, "episode_id": 4, "request_id": 0}
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    a = policy.infer({REQUEST_KEY: identity})
    b = policy.infer({REQUEST_KEY: {**identity, "episode_id": 5}})
    again = policy.infer({REQUEST_KEY: identity})
    np.testing.assert_array_equal(a["actions"], again["actions"])
    assert not np.array_equal(a["actions"], b["actions"])
    validate_response_identity(a, identity, backend="gr00t")
    with pytest.raises(RuntimeError):
        validate_response_identity(b, identity, backend="gr00t")
    with pytest.raises(ValueError):
        with isolated_rng(5):
            random.random(); np.random.rand(); torch.rand(1)
            raise ValueError("inference failed")
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(1), expected[2])


def test_checkpoint_content_and_upstream_identity_are_immutable(tmp_path, monkeypatch):
    from vlalagbench.benchmarks.libero import gr00t_adapter as gr00t
    monkeypatch.setattr(gr00t.subprocess, "check_output", lambda args, **kw: gr00t.UPSTREAM_REVISION + "\n" if "rev-parse" in args else "")
    for file in ("config.json", "processor_config.json", "statistics.json"):
        (tmp_path / file).write_text('{}')
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    original = gr00t.checkpoint_identity(model(), tmp_path, tmp_path)
    for file in ("processor_config.json", "statistics.json", "model.safetensors"):
        before = (tmp_path / file).read_bytes()
        (tmp_path / file).write_bytes(before + b" ")
        assert original != gr00t.checkpoint_identity(model(), tmp_path, tmp_path)
        (tmp_path / file).write_bytes(before)
    monkeypatch.setattr(gr00t.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    with pytest.raises(ValueError, match="upstream revision"):
        gr00t.checkpoint_identity(model(), tmp_path, tmp_path)


def test_inputs_and_gripper_match_pinned_upstream():
    import ast
    import math
    from pathlib import Path
    from types import SimpleNamespace
    from vlalagbench.benchmarks.libero.gr00t_adapter import AXES, decode_actions, format_observation
    root = Path(__file__).resolve().parents[3]
    path = root / "third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py"
    if not path.exists():
        pytest.skip("initialize the pinned Isaac-GR00T submodule")
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                 {"quat2axisangle", "normalize_gripper_action", "invert_gripper_action"}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv")
    functions.append(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_process_observation"))
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    raw = observation()
    reference = namespace["_process_observation"](SimpleNamespace(_task_description="move"), raw)
    actual = format_observation(raw, "move")
    for kind in ("video", "state"):
        for key, value in actual[kind].items():
            np.testing.assert_allclose(value[0, 0], reference[f"{kind}.{key}"], rtol=1e-6, atol=1e-7)
    assert actual["language"]["annotation.human.action.task_description"][0][0] == reference["annotation.human.action.task_description"]
    decoded = {key: np.random.default_rng(i).uniform(0, 1, (1, 16, 1)) for i, key in enumerate((*AXES, "gripper"))}
    reference_actions = np.concatenate(list(decoded.values()), axis=-1)[0]
    reference_actions = namespace["invert_gripper_action"](namespace["normalize_gripper_action"](reference_actions))
    np.testing.assert_array_equal(decode_actions(decoded), reference_actions)
