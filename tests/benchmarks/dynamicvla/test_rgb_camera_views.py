"""RGB-only camera reads return upstream's retained views without segmentation work."""
import types

import numpy as np
import pytest
import torch

from vlalagbench.benchmarks.dynamicvla import runtime


class Camera:
    def __init__(self, output):
        self.data = types.SimpleNamespace(output=output, info=[{}])


class ContactSensor:
    pass


def convert(data, backend):
    assert backend == "numpy"
    return {key: value.detach().cpu().numpy() for key, value in data.items()}


def upstream_views(sensors, views):
    converted = {}
    for name, sensor in sensors.items():
        if type(sensor).__name__ == "Camera":
            outputs = convert(dict(sensor.data.output), "numpy")
            if "semantic_segmentation" in outputs:
                outputs["seg"] = np.zeros_like(outputs.pop("semantic_segmentation")[..., :1])
            converted[name] = {k: v for k, v in outputs.items() if k in views}
    return converted


def sensors():
    rgb = torch.arange(2 * 3 * 4 * 3, dtype=torch.uint8).reshape(2, 3, 4, 3)
    return {
        "wrist_cam": Camera({"rgb": rgb, "semantic_segmentation": torch.ones(2, 3, 4, 4)}),
        "contact": ContactSensor(),
        "side_cam": Camera({"semantic_segmentation": torch.ones(1, 1, 1, 4), "rgb": rgb[:1] + 1}),
        "empty_cam": Camera({"semantic_segmentation": torch.ones(1, 1, 1, 4)}),
    }


def test_rgb_only_views_match_upstream_order_keys_dtype_and_bytes():
    source = sensors()
    expected = upstream_views(source, ["rgb"])
    actual = runtime.rgb_camera_views(source, convert)
    runtime._assert_same_camera_views(expected, actual)
    assert list(actual) == ["wrist_cam", "side_cam", "empty_cam"]
    assert actual["empty_cam"] == {}


def test_verify_rejects_any_pixel_difference():
    source = sensors()
    expected = upstream_views(source, ["rgb"])
    actual = runtime.rgb_camera_views(source, convert)
    actual["side_cam"]["rgb"] = actual["side_cam"]["rgb"].copy()
    actual["side_cam"]["rgb"][0, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="side_cam/rgb"):
        runtime._assert_same_camera_views(expected, actual)


def test_unknown_mode_is_rejected(monkeypatch):
    monkeypatch.setenv(runtime.ENV_RGB_ONLY_CAMERAS, "yes")
    with pytest.raises(ValueError, match="1, 0, or verify"):
        runtime.read_policy_observation(None, types.SimpleNamespace(unwrapped=None))
