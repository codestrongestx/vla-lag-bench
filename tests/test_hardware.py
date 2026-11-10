import types
from vlalagbench import hardware


def test_hardware_is_discovered_without_provider_metadata(monkeypatch):
    monkeypatch.setattr(hardware.socket, "gethostname", lambda: "gpu-node-1")
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout=(
                "NVIDIA GeForce RTX 4090\n"
                "NVIDIA GeForce RTX 4090\n"
                "NVIDIA A100 80GB PCIe\n"
            )
        ),
    )

    audit = hardware.collect_hardware()

    assert audit.hostname == "gpu-node-1"
    assert audit.cpu_count > 0
    assert {gpu.type: gpu.count for gpu in audit.gpus} == {
        "NVIDIA A100 80GB PCIe": 1,
        "NVIDIA GeForce RTX 4090": 2,
    }


def test_slow_nvidia_smi_is_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) < 3:
            raise hardware.subprocess.TimeoutExpired("nvidia-smi", kwargs["timeout"])
        return types.SimpleNamespace(stdout="NVIDIA H100 80GB HBM3\n")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    monkeypatch.setattr(hardware.time, "sleep", lambda seconds: None)

    assert {gpu.type: gpu.count for gpu in hardware.collect_hardware().gpus} == {"NVIDIA H100 80GB HBM3": 1}
    assert len(calls) == 3


def test_missing_nvidia_smi_is_not_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    monkeypatch.setattr(hardware.time, "sleep", lambda seconds: calls.append("sleep"))

    assert hardware._nvidia_gpu_types() == []
    assert len(calls) == 1
