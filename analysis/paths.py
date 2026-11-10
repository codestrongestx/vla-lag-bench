"""Where the analysis reads and writes, relative to the repository root.

Inputs:
  results/          the benchmark's published success tables (results/libero/..., results/dynamicvla/...)
  analysis/data/    derived tables shipped with the analysis (gzip CSV), so every figure rebuilds without raw traces
  configs/          benchmark configurations (the execution-mode timeline reads one)
Optional raw inputs, needed only to re-measure a derived table or to draw the recorded scene frames:
  VLALAGBENCH_ARTIFACTS   benchmark artifact storage holding studies/<study>/... run records, traces, and videos
                        (default: artifacts/)
  VLALAGBENCH_CHECKOUT    a benchmark checkout whose third_party/ holds the LIBERO assets the recorded MuJoCo models
                        reference (default: this repository)
Outputs, under ANALYSIS_BUILD (default: analysis/build):
  ANALYSIS_OUTPUT       rendered figures (default: <build>/figures)
  ANALYSIS_DERIVED      full measurement outputs before export to analysis/data (default: <build>/derived)
  <build>/latency_model the latency model's fit and sweep tables
"""
import csv
import functools
import glob
import gzip
import hashlib
import io
import os
import re
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
CONFIGS = ROOT / 'configs'
DATA = Path(os.environ.get('ANALYSIS_DATA', ROOT / 'analysis/data')).expanduser().resolve()
ARTIFACTS = Path(os.environ.get('VLALAGBENCH_ARTIFACTS', ROOT / 'artifacts')).expanduser().resolve()
STUDIES = ARTIFACTS / 'studies'
CHECKOUT = Path(os.environ.get('VLALAGBENCH_CHECKOUT', ROOT)).expanduser().resolve()
BUILD = Path(os.environ.get('ANALYSIS_BUILD', ROOT / 'analysis/build')).expanduser().resolve()
OUTPUT = Path(os.environ.get('ANALYSIS_OUTPUT', BUILD / 'figures')).expanduser().resolve()
DERIVED = Path(os.environ.get('ANALYSIS_DERIVED', BUILD / 'derived')).expanduser().resolve()
# pi0.5 inference-stack latency and success tables.
PI05_STACKS = RESULTS / 'libero/20261014_pi05-inference-stacks'


def use_benchmark_src():
    """Import vlalagbench from this repository's src/ (the simulator and scheduler code the analysis replays)."""
    src = str(ROOT / 'src')
    if sys.path[:1] != [src]:
        sys.path.insert(0, src)
    return src


def output(name):
    path = BUILD / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def roots():
    return {'artifacts': ARTIFACTS, 'results': RESULTS, 'derived': DERIVED, 'data': DATA, 'checkout': CHECKOUT,
            'repo': ROOT}


def portable(path):
    """A path named by its root, so provenance records do not depend on where the inputs live."""
    path = Path(path).resolve()
    for prefix, root in sorted(roots().items(), key=lambda pair: len(pair[1].parts), reverse=True):
        if path.is_relative_to(root):
            return prefix + '/' + str(path.relative_to(root))
    raise ValueError(f'Path outside the analysis roots: {path}')


def resolve(name):
    prefix, relative = name.split('/', 1)
    if '..' in Path(relative).parts or Path(relative).is_absolute():
        raise ValueError(f'Unsafe path: {name}')
    return roots()[prefix] / relative


def trace_path(name):
    """A recorded file named relative to the benchmark root (artifacts/studies/...)."""
    return ARTIFACTS / Path(name).relative_to('artifacts')


def read_rows(path):
    """Rows of a CSV table, gzip-compressed or not."""
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', newline='') as stream:
            return list(csv.DictReader(stream))
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def data_rows(name):
    """Rows of a shipped derived table, analysis/data/<name>.csv.gz."""
    return read_rows(DATA / f'{name}.csv.gz')


def write_gzip_csv(path, rows, fields=None, drop=()):
    """Write rows as a gzip CSV with a fixed timestamp, so the same rows give the same bytes."""
    fields = [f for f in (fields or list(rows[0])) if f not in drop]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fields, extrasaction='ignore', lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows)
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=9, mtime=0, filename='') as stream:
        stream.write(text.getvalue().encode())
    return path


def localize_xml(xml):
    """Point the asset paths a recorded MuJoCo model names on its evaluation host to this machine."""
    local = str(CHECKOUT)
    packages = sysconfig.get_paths()['purelib']
    xml = re.sub(r"/(?:root|home)/[^\"']*?/dynamic-robot-benchmark", local, xml)
    xml = xml.replace('/opt/gr00t_runtime/envs/sim/lib/python3.11/site-packages', packages)
    xml = xml.replace('/opt/gr00t_runtime/repo', local)
    xml = re.sub(r'"/[^"]*?/third_party/', '"' + local + '/third_party/', xml)
    xml = re.sub(r'"/[^"]*?/site-packages/', '"' + packages + '/', xml)
    return xml


@functools.lru_cache(maxsize=None)
def _source_root(source):
    """The study directory that holds <source>/points, one or two levels below studies/."""
    for pattern in (STUDIES / source, STUDIES / '*' / source):
        for folder in sorted(glob.glob(str(pattern))):
            if os.path.isdir(os.path.join(folder, 'points')):
                return folder
    # Older layouts dropped the mode suffix from the directory name.
    stem = source.rsplit('_', 1)[0]
    for pattern in (STUDIES / stem, STUDIES / '*' / stem):
        for folder in sorted(glob.glob(str(pattern))):
            if os.path.isdir(os.path.join(folder, 'points')):
                return folder
    raise FileNotFoundError(source)


def point_dir(source, speed, mode, latency):
    folder = os.path.join(_source_root(source), 'points', f'{float(speed):g}_{mode}_{latency}')
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    return folder


def figure_path(name):
    """A rendered paper figure, named as in the paper's generated/board-figures/<name>.png."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT / f'{name}.png'
