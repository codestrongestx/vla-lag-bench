"""Verify extraction preserves recorded frames without starting a simulator."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from vlalagbench.study.videos import Selection, _render_one


def test_recorded_camera_crop_preserves_frame_count_and_view(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is required for recorded-video extraction')
    cv2 = pytest.importorskip('cv2')
    frames = np.zeros((4, 32, 96, 3), dtype=np.uint8)
    frames[:, :, :32, 0] = 255
    frames[:, :, 32:64, 1] = 255
    frames[:, :, 64:, 2] = 255
    source = tmp_path / 'live.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '96x32', '-r', '25', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(source)], input=frames.tobytes(), check=True)
    selected = Selection('success', 0, 0, Path('unused.npz'), kind='dynamicvla',
                         recorded_video=source, recorded_cameras=('left', 'middle', 'right'))
    output = tmp_path / 'middle.mp4'
    assert _render_one((selected, 'middle', output, 256)) == output
    cap = cv2.VideoCapture(str(output))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (32, 32, 3)
        assert frame[:, :, 1].mean() > 240
        assert frame[:, :, (0, 2)].mean() < 10
        count += 1
    cap.release()
    assert count == 4
    assert not list(tmp_path.glob('*.partial.mp4'))
