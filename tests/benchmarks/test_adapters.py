import subprocess
import sys

import pytest

from vlalagbench.benchmarks import get_adapter
from vlalagbench.benchmarks.dynamicvla.adapter import ADAPTER as DYNAMICVLA_ADAPTER
from vlalagbench.benchmarks.libero.adapter import ADAPTER


def test_get_adapter_selects_libero_and_rejects_unknown_kinds():
    assert get_adapter("libero") is ADAPTER
    assert get_adapter("dynamicvla") is DYNAMICVLA_ADAPTER
    with pytest.raises(ValueError, match="unknown benchmark adapter"):
        get_adapter("unknown")


def test_adapter_contract_import_does_not_load_config_stack():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vlalagbench.benchmarks; "
            "assert 'pydantic' not in sys.modules; assert 'yaml' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
