"""The shipped derived tables match their manifest and aggregate to the published success tables."""
import collections
import csv
import unittest

from analysis.measure import verify
from analysis.paths import RESULTS, data_rows
from analysis.sync_hold import replace_rows

LIBERO = RESULTS / 'libero'


def published(path, keep=lambda row: True):
    with path.open() as stream:
        return {(float(r['trajectory.speed_mps']), r['inference.mode'], int(r['inference.latency_ms'])):
                (int(r['successes']), int(r['episodes'])) for r in csv.DictReader(stream) if keep(r)}


def with_reruns(table):
    """A published table's cells and its folder's synchronous hold_pose reruns (<model>-sync-hold-pose), which replace
    the synchronous settings at a nonzero latency."""
    cells = published(table)
    hold = table.parent.parent / f'{table.parent.name.split("-")[0]}-sync-hold-pose/plot_data.csv'
    if table.parent.name == 'fcan04_native200_20261001':
        hold = table.parent / 'dynamicvla-sync-hold-pose/plot_data.csv'
    if hold.is_file():
        cells.update(published(hold, lambda r: r['idle_control'] == 'hold_pose'))
    return cells


def tally(rows, **match):
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            cell = out[(float(r['speed']), r['mode'], int(r['latency']))]
            cell[0] += int(r['success'])
            cell[1] += 1
    return {key: tuple(value) for key, value in out.items()}


class Data(unittest.TestCase):
    def test_manifest(self):
        self.assertEqual(verify(), [])

    def test_episode_tables_match_published_counts(self):
        cases = [
            ('pi0.5', data_rows('libero_episodes/pi05'), dict(task='basket'), LIBERO / '20260908_moving-basket/pi05/plot_data.csv'),
            ('pi0.5', data_rows('libero_episodes/pi05'), dict(task='drawer'), LIBERO / '20260926_moving-drawer-fixed/pi05/plot_data.csv'),
            ('GR00T N1.7', data_rows('libero_episodes/gr00t'), dict(task='basket'), LIBERO / '20260908_moving-basket/gr00t/plot_data.csv'),
            ('GR00T N1.7', data_rows('libero_episodes/gr00t'), dict(task='drawer'), LIBERO / '20260926_moving-drawer-fixed/gr00t/plot_data.csv'),
            ('pi0.5', data_rows('plate_cabinet_episodes/episodes'), dict(task='plate', model='pi0.5'),
             LIBERO / '20261001_task9-moving-plate/pi05-h100-b32/plot_data.csv'),
            ('pi0.5', data_rows('latency_grid_episodes/episodes'), dict(task='drawer'),
             LIBERO / '20260926_moving-drawer-fixed/pi05-latency-grid/plot_data.csv'),
            (None, data_rows('dynamicvla_episodes/episodes'), dict(task='box'), RESULTS / 'dynamicvla/fcan04_native200_20261001/plot_data.csv'),
        ]
        for policy, rows, match, table in cases:
            with self.subTest(table=str(table.relative_to(RESULTS)), **match):
                if policy:
                    rows = replace_rows(policy, match['task'], [r for r in rows if all(r[k] == v for k, v in match.items())])
                counts, expected = tally(rows, **match), with_reruns(table)
                self.assertLessEqual(set(published(table)), set(counts))
                self.assertEqual(counts, {key: expected.get(key) for key in counts})
        # The bowl combines the bowl-centered runs and the speed fill; its published speed-fill table holds the grid.
        bowl = with_reruns(LIBERO / '20261010_task9-moving-cabinet-bowl-centered/pi05-speed-fill/plot_data.csv')
        rows = [r for r in data_rows('plate_cabinet_episodes/episodes') if r['task'] == 'cabinet' and r['model'] == 'pi0.5']
        counts = tally(replace_rows('pi0.5', 'cabinet', rows))
        self.assertEqual(counts, {key: bowl.get(key) for key in counts})


if __name__ == '__main__':
    unittest.main()
