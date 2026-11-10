from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import pytest

from vlalagbench.plotting import (
    CONFIDENCE_INTERVAL_METADATA_KEY,
    CONFIDENCE_INTERVAL_METHOD,
    ERRORBAR_SERIES_METADATA_KEY,
    save_confidence_interval_figure,
)


def test_confidence_interval_figure_requires_errorbar_artists(tmp_path: Path):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [50, 60])

    with pytest.raises(RuntimeError, match="expected 1 error-bar series, found 0"):
        save_confidence_interval_figure(
            figure,
            tmp_path / "invalid.png",
            expected_errorbar_series=1,
        )
    plt.close(figure)


def test_confidence_interval_figure_records_validated_metadata(tmp_path: Path):
    figure, axis = plt.subplots()
    axis.errorbar([0, 1], [50, 60], yerr=[2, 3])
    output = tmp_path / "valid.png"

    save_confidence_interval_figure(
        figure,
        output,
        expected_errorbar_series=1,
    )
    plt.close(figure)

    with Image.open(output) as image:
        assert image.info[CONFIDENCE_INTERVAL_METADATA_KEY] == CONFIDENCE_INTERVAL_METHOD
        assert image.info[ERRORBAR_SERIES_METADATA_KEY] == "1"
