"""Plot-output validation shared by result-rendering scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure


CONFIDENCE_INTERVAL_METHOD = "pointwise 95% Wilson"
CONFIDENCE_INTERVAL_METADATA_KEY = "ConfidenceInterval"
ERRORBAR_SERIES_METADATA_KEY = "ErrorbarSeries"


def save_confidence_interval_figure(
    figure: Figure,
    output: Path,
    *,
    expected_errorbar_series: int,
    **savefig_options: Any,
) -> None:
    """Validate error-bar artists and mark the saved PNG as a CI artifact."""
    if expected_errorbar_series <= 0:
        raise ValueError("expected_errorbar_series must be positive")
    observed = sum(
        isinstance(container, ErrorbarContainer)
        for axis in figure.axes
        for container in axis.containers
    )
    if observed != expected_errorbar_series:
        raise RuntimeError(
            "refusing to save confidence-interval figure: "
            f"expected {expected_errorbar_series} error-bar series, found {observed}"
        )
    metadata = dict(savefig_options.pop("metadata", {}))
    metadata.update(
        {
            CONFIDENCE_INTERVAL_METADATA_KEY: CONFIDENCE_INTERVAL_METHOD,
            ERRORBAR_SERIES_METADATA_KEY: str(observed),
        }
    )
    figure.savefig(output, metadata=metadata, **savefig_options)
