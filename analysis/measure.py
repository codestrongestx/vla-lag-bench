"""Rebuild the derived tables in analysis/data/ from benchmark run records and traces.

    python -m analysis.measure --list
    python -m analysis.measure NAME [NAME ...] [--workers N] [--force]
    python -m analysis.measure --verify

Each measurement writes its full outputs (with trace paths, hashes, and provenance) under ANALYSIS_DERIVED, then
exports the columns the figures read to analysis/data/<folder>/<table>.csv.gz and records them in
analysis/data/MANIFEST.json. Run records and traces are read from VLALAGBENCH_ARTIFACTS (see analysis/paths.py).
A measurement that reads another's full outputs (plate-cabinet-replay, basket-releases, dynamicvla-replay) needs
that one first.
"""
import argparse
import importlib
import json
import sys

from analysis.paths import DATA, DERIVED, read_rows, sha, write_gzip_csv

# Columns that only locate raw inputs; the shipped tables leave them out.
LOCATORS = ('trace', 'trace_sha256', 'commit', 'study')
# name -> (module with ensure(), {derived table: shipped table}, columns left out)
MEASUREMENTS = {
    'libero-episodes': ('analysis.cohorts.libero',
                        {'libero_episodes/pi05.csv': 'libero_episodes/pi05', 'libero_episodes/gr00t.csv': 'libero_episodes/gr00t'}, ()),
    'plate-cabinet-episodes': ('analysis.cohorts.plate_cabinet.episodes',
                               {'plate_cabinet_episodes/episodes.csv': 'plate_cabinet_episodes/episodes'}, LOCATORS),
    'latency-grid-episodes': ('analysis.cohorts.latency_grid.episodes',
                              {'latency_grid_episodes/episodes.csv': 'latency_grid_episodes/episodes'}, LOCATORS),
    'molmoact2-episodes': ('analysis.cohorts.molmoact2.episodes',
                           {'molmoact2_episodes/episodes.csv': 'molmoact2_episodes/episodes'}, LOCATORS),
    'dynamicvla-episodes': ('analysis.cohorts.dynamicvla.episodes',
                            {'dynamicvla_episodes/episodes.csv': 'dynamicvla_episodes/episodes'}, LOCATORS),
    'dynamicvla-replay': ('analysis.cohorts.dynamicvla.replay',
                          {'dynamicvla_replay/attempts.csv': 'dynamicvla_replay/attempts'}, LOCATORS),
    'basket-alignment': ('analysis.cohorts.basket_alignment.measure',
                         {'basket_alignment_speed/episodes.csv': 'basket_alignment_speed/episodes'}, LOCATORS),
    'drawer-alignment': ('analysis.cohorts.drawer_alignment.measure',
                         {'drawer_alignment_speed/episodes.csv': 'drawer_alignment_speed/episodes'}, LOCATORS),
    'plate-cabinet-replay': ('analysis.cohorts.plate_cabinet.replay',
                             {'plate_cabinet_replay/alignment.csv': 'plate_cabinet_replay/alignment',
                              'plate_cabinet_replay/attempts.csv': 'plate_cabinet_replay/attempts'}, LOCATORS),
    'basket-releases': ('analysis.cohorts.basket_releases',
                        {'basket_releases/attempts.csv': 'basket_releases/attempts'}, LOCATORS),
}
MANIFEST = DATA / 'MANIFEST.json'


def manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def export(name, source=DERIVED):
    """Write the shipped tables of one measurement from a folder laid out like ANALYSIS_DERIVED."""
    _, tables, drop = MEASUREMENTS[name]
    entries = manifest()
    for derived, shipped in tables.items():
        rows = read_rows(source / derived)
        path = write_gzip_csv(DATA / f'{shipped}.csv.gz', rows, drop=drop)
        entries[f'{shipped}.csv.gz'] = dict(measurement=name, rows=len(rows), bytes=path.stat().st_size, sha256=sha(path))
        print(f'{name}: {len(rows)} rows -> {path}', flush=True)
    MANIFEST.write_text(json.dumps(dict(sorted(entries.items())), indent=2) + '\n')


def verify():
    """Check every shipped table against MANIFEST.json; returns the problems found."""
    problems = []
    for name, entry in manifest().items():
        path = DATA / name
        if not path.exists():
            problems.append(f'missing {name}')
        elif sha(path) != entry['sha256']:
            problems.append(f'{name} differs from MANIFEST.json')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', metavar='NAME')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        for name, (module, tables, _) in MEASUREMENTS.items():
            print(f'{name:24s} {module}  -> ' + ', '.join(f'{t}.csv.gz' for t in tables.values()))
        return
    if args.verify:
        problems = verify()
        print('\n'.join(problems) or f'{len(manifest())} shipped tables match MANIFEST.json')
        sys.exit(1 if problems else 0)
    unknown = set(args.names) - set(MEASUREMENTS)
    if unknown or not args.names:
        parser.error(f'choose measurements from: {", ".join(MEASUREMENTS)}')
    for name in args.names:
        importlib.import_module(MEASUREMENTS[name][0]).ensure(workers=args.workers, force=args.force)
        export(name)


if __name__ == '__main__':
    main()
