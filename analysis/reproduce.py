"""Rebuild the paper's 24 figures and print its analysis-derived numbers.

    python -m analysis.reproduce [--out DIR] [--compare DIR] [--no-numbers] [FIGURE ...]

Figures are named as in the paper (generated/board-figures/<name>.png). Each renderer runs in its own process and
writes into --out (default analysis/build/figures). overview, task-overview, dynamicvla-task-overview, and
tolerance-zone draw recorded simulator frames and need raw files from benchmark artifact storage (VLALAGBENCH_ARTIFACTS)
and the LIBERO assets (VLALAGBENCH_CHECKOUT), and tolerance-zone also the full basket-alignment and
plate-cabinet-episodes outputs (python -m analysis.measure); without them they are skipped with a note. overview also
needs tectonic, pdftoppm, and the Helvetica Neue font. --compare checks every rendered figure pixel for
pixel against a folder of reference PNGs. The shipped derived tables are checked against analysis/data/MANIFEST.json
first.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from analysis.paths import ARTIFACTS, ROOT

# Renderer module (and arguments) -> the figures it writes.
RENDERERS = {
    ('analysis.figures.overview',): ('overview',),
    ('analysis.figures.action_chunking',): ('vla-architecture', 'action-chunking'),
    ('analysis.figures.task_overview',): ('task-overview',),
    ('analysis.figures.task_overview', '--dynamicvla'): ('dynamicvla-task-overview',),
    ('analysis.figures.execution_modes',): ('schedule-quantities',),
    ('analysis.figures.success_grid',): ('task-success', 'prediction-summary', 'success-grid', 'gr00t-success-grid',
                                         'prediction-examples', 'molmoact2-prediction-examples',
                                         'dynamicvla-success-grid'),
    ('analysis.figures.mode_comparison',): ('mode-summary', 'mode-comparison', 'molmoact2-mode-comparison',
                                            'dynamicvla-mode-comparison'),
    ('analysis.figures.tolerance_zone',): ('tolerance-zone', 'dynamicvla-tolerance-zone'),
    ('analysis.figures.pace',): ('pace',),
    ('analysis.figures.policy_summary',): ('policy-summary',),
    ('analysis.figures.policy_comparison',): ('policy',),
    ('analysis.figures.deployment',): ('deployment-modes', 'deployment-rate'),
}
RAW = {'analysis.figures.overview', 'analysis.figures.task_overview', 'analysis.figures.tolerance_zone'}
FIGURES = [name for names in RENDERERS.values() for name in names]


def compare(path, reference):
    """'identical', or the size or pixel difference against the reference PNG."""
    import numpy as np
    from PIL import Image
    if not reference.exists():
        return 'no reference'
    a, b = (np.asarray(Image.open(p).convert('RGBA'), dtype=int) for p in (path, reference))
    if a.shape != b.shape:
        return f'size {a.shape[1]}x{a.shape[0]} against {b.shape[1]}x{b.shape[0]}'
    changed = np.abs(a - b).max(-1)
    return 'identical' if not changed.any() else f'{np.count_nonzero(changed)} pixels differ (largest {changed.max()})'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figures', nargs='*', metavar='FIGURE', help=', '.join(FIGURES))
    parser.add_argument('--out', type=Path, help='folder for the rendered figures')
    parser.add_argument('--compare', type=Path, help='folder of reference PNGs to compare against')
    parser.add_argument('--no-numbers', action='store_true', help='skip the cited model numbers')
    args = parser.parse_args(argv)
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        parser.error(f'unknown figures {sorted(unknown)}; choose from {", ".join(FIGURES)}')
    from analysis.measure import verify
    problems = verify()
    if problems:
        sys.exit('Shipped tables differ from analysis/data/MANIFEST.json:\n  ' + '\n  '.join(problems))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if args.out:
        env['ANALYSIS_OUTPUT'] = str(args.out.resolve())
    out = Path(env.get('ANALYSIS_OUTPUT', ROOT / 'analysis/build/figures'))
    env.setdefault('MPLCONFIGDIR', str(out.parent / 'matplotlib'))
    selected = set(args.figures or FIGURES)
    results, failed = {}, False
    for command, names in RENDERERS.items():
        if not selected & set(names):
            continue
        if command[0] in RAW and not (ARTIFACTS / 'studies').is_dir():
            for name in names:
                results[name] = f'skipped: needs recorded frames under {ARTIFACTS / "studies"} (VLALAGBENCH_ARTIFACTS)'
            continue
        print(f'$ python -m {" ".join(command)}', flush=True)
        run = subprocess.run([sys.executable, '-m', *command], cwd=ROOT, env=env)
        for name in names:
            path = out / f'{name}.png'
            if run.returncode or not path.exists():
                results[name], failed = 'render failed', True
            else:
                results[name] = compare(path, args.compare / f'{name}.png') if args.compare else f'written to {path}'
    print('\nFigures:')
    for name in FIGURES:
        if name in results:
            print(f'  {name:32s} {results[name]}')
    if not args.no_numbers:
        from analysis.numbers import main as numbers
        numbers([])
    if failed or (args.compare and any(r != 'identical' and not r.startswith('skipped') for r in results.values())):
        sys.exit(1)


if __name__ == '__main__':
    main()
