"""The introduction's overview (Figure 1): the benchmark's five numbered factors (task type, target speed, VLA
policy, execution mode, inference latency), the research questions with their sections, and the success model as one
black box: calibration data in, success rates out; below it, the best execution mode for each system.

The figure is drawn in TikZ (analysis/figures/overview_figure/) in the visual style of VLA-Perf's Figure 1: Helvetica
Neue, blue section labels, pastel knobs joined by crosses, cream questions with checks, and a lavender success model.
This script exports the task photos, one recorded interaction frame per task (VLALAGBENCH_ARTIFACTS) cropped around its
boxed moving object, compiles the TikZ with Tectonic, and writes the vector PDF the paper prints and a PNG beside it
(pdftoppm). Needs tectonic, pdftoppm, and the Helvetica Neue font."""
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from analysis.paths import ARTIFACTS, figure_path, use_benchmark_src

use_benchmark_src()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.figures.style import TARGET_BOX
from analysis.figures.task_overview import SOURCES, draw_box, frame, target_boxes
from analysis.figures.typography import DPI

SOURCE = Path(__file__).with_name("overview_figure")
# Each photo shows this fraction of the frame, a square centered on the boxed target and kept inside the frame, so
# the robot and the moving object fill the thumbnail.
CROP = .75


def photo(key, spec, path):
    point = ARTIFACTS / spec["point"]
    video = point / spec["video"]
    if hashlib.sha256(video.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Recorded video changed: {video}")
    index = spec["frames"][1]
    pixels = frame(video, index)
    (target,) = target_boxes({**spec, "frames": (index,)}, point, pixels.shape[:2])
    height, width = pixels.shape[:2]
    side = int(CROP * min(height, width))
    cx, cy = (target[0] + target[1]) / 2, (target[2] + target[3]) / 2
    x0 = int(min(max(cx - side / 2, 0), width - side))
    y0 = int(min(max(cy - side / 2, 0), height - side))
    fig = plt.figure(figsize=(side / 100, side / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pixels, interpolation="none")
    ax.set_xlim(x0 - .5, x0 + side - .5)
    ax.set_ylim(y0 + side - .5, y0 - .5)
    ax.axis("off")
    draw_box(ax, target, TARGET_BOX, pixels.shape[:2])
    # The crop is ~1.9 in wide here but prints at 0.7 in, so the box is thickened to stay visible.
    for patch in ax.patches:
        patch.set_linewidth(3)
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    destination = figure_path("overview").with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        for name in ("overview.tex", "style.tex"):
            shutil.copy(SOURCE / name, work / name)
        for key, spec in SOURCES.items():
            photo(key, spec, work / f"task-{key}-crop.png")
        # A fixed timestamp keeps the PDF byte-identical across runs.
        subprocess.run(["tectonic", "-X", "compile", "overview.tex"], cwd=work, check=True,
                       env={**os.environ, "SOURCE_DATE_EPOCH": "0"}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        shutil.copy(work / "overview.pdf", destination)
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(destination), str(destination.with_suffix(""))],
                   check=True)


if __name__ == "__main__":
    main()
