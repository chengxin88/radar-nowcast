"""Offline tests with real NumPy and synthetic translated echoes."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import numpy as np
import requests
import radar_nowcast as product
import radar_track

BJT = product.BJT
LATEST = datetime(2026, 10, 13, 15, 30, tzinfo=BJT)


def forbidden(*args, **kwargs):
    raise AssertionError('Network, EC retrieval and messaging are forbidden in these tests')


def grid(half=150):
    axis = np.arange(-half, half + 1, dtype=float)
    return np.meshgrid(axis, axis[::-1])


def echo(east, north, center=-40, value=40):
    return np.where((np.abs(east - center) <= 5) & (np.abs(north) <= 6), value, 0)


def translated_pair(latest=LATEST, half=150, center=-40, value=40):
    east, north = grid(half)
    return [(latest - timedelta(minutes=6), echo(east, north, center - 6, value), east, north),
            (latest, echo(east, north, center, value), east, north)]


def trusted_motion(speed=60):
    return {'status': 'ok', 'reason': None, 've': float(speed), 'vn': 0.0,
            'correlation': 1.0, 'peak_margin': 0.3}


def step(minute, status):
    return {'minute': minute, 'status': status, 'rain': True if status in ('rain', 'strong') else False if status == 'clear' else None}


class RadarNowcastTests(unittest.TestCase):
    def setUp(self):
        for method in ('get', 'head', 'post'):
            guard = patch.object(requests, method, side_effect=forbidden)
            guard.start()
            self.addCleanup(guard.stop)

    def analyse(self, frames=None, now=LATEST, speed=60, **kwargs):
        with patch.object(radar_track, 'motion', return_value=trusted_motion(speed)):
            return product.analyse(translated_pair() if frames is None else frames, now=now, **kwargs)

    def test_real_motion_recovers_synthetic_translation_and_rain_interval(self):
        result = product.analyse(translated_pair(), now=LATEST)
        self.assertTrue(result['forecast_reliable'])
        self.assertAlmostEqual(result['motion']['east_kmh'], 60)
        self.assertAlmostEqual(result['motion']['north_kmh'], 0)
        self.assertEqual(result['arrival_minutes'], 35)
        self.assertEqual(result['strong_arrival_minutes'], 35)
        self.assertEqual(result['end_minutes'], 50)
        self.assertEqual(len(result['steps']), 25)
        self.assertEqual([x['minute'] for x in result['steps']], list(range(0, 121, 5)))

    def test_frame_delay_is_included_in_arrival_relative_to_real_now(self):
        baseline = self.analyse()
        delayed = self.analyse(now=LATEST + timedelta(minutes=10))
        self.assertEqual(delayed['arrival_minutes'], baseline['arrival_minutes'] - 10)
        self.assertEqual(delayed['strong_arrival_minutes'], baseline['strong_arrival_minutes'] - 10)
        self.assertEqual(delayed['end_minutes'], baseline['end_minutes'] - 10)
        self.assertEqual(delayed['current']['age_minutes'], 10)
        self.assertEqual(delayed['steps'][0]['time'], (LATEST + timedelta(minutes=10)).isoformat())
        self.assertEqual(delayed['window_end'], (LATEST + timedelta(minutes=130)).isoformat())

    def test_latest_observation_is_distinct_from_age_compensated_step_zero(self):
        result = self.analyse(translated_pair(center=-10), now=LATEST + timedelta(minutes=10))
        self.assertFalse(result['current']['rain'])
        self.assertTrue(result['steps'][0]['rain'])
        self.assertEqual(result['arrival_minutes'], 0)
        self.assertEqual(result['current']['observed_at'], LATEST.isoformat())

    def test_empty_and_single_frame_never_create_forecast_times(self):
        empty = product.analyse([], now=LATEST)
        self.assertFalse(empty['publication_allowed'])
        single = product.analyse(translated_pair(center=0)[-1:], now=LATEST)
        self.assertTrue(single['publication_allowed'])
        self.assertTrue(single['current']['rain'])
        self.assertFalse(single['forecast_reliable'])
        self.assertTrue(all(item['status'] == 'unknown' for item in single['steps']))
        for key in ('arrival_minutes', 'strong_arrival_minutes', 'end_minutes'):
            self.assertIsNone(single[key])

    def test_blank_echoes_and_low_confidence_do_not_claim_two_hour_clear(self):
        east, north = grid()
        blank = np.zeros_like(east)
        frames = [(LATEST - timedelta(minutes=6), blank, east, north), (LATEST, blank, east, north)]
        result = product.analyse(frames, now=LATEST)
        self.assertFalse(result['forecast_reliable'])
        self.assertFalse(result['clear'])
        self.assertFalse(result['current']['rain'])
        with patch.object(radar_track, 'motion', return_value={'status': 'unavailable', 've': None, 'vn': None, 'reason': 'low correlation'}):
            result = product.analyse(translated_pair(), now=LATEST)
        self.assertFalse(result['forecast_reliable'])
        self.assertIsNone(result['arrival_minutes'])
        self.assertFalse(result['clear'])

    def test_repeated_or_reverse_timestamps_do_not_call_motion(self):
        frames = translated_pair()
        variants = [frames[::-1], [frames[-1], frames[-1]]]
        for variant in variants:
            with self.subTest(times=[frame[0] for frame in variant]), patch.object(radar_track, 'motion', side_effect=forbidden):
                result = product.analyse(variant, now=LATEST)
                self.assertFalse(result['forecast_reliable'])
                self.assertIsNone(result['arrival_minutes'])

    def test_stale_and_future_frames_are_not_publishable(self):
        for now in (LATEST + timedelta(minutes=21), LATEST - timedelta(minutes=1), LATEST - timedelta(minutes=6)):
            with self.subTest(now=now):
                result = self.analyse(now=now)
                self.assertFalse(result['publication_allowed'])
                self.assertFalse(result['forecast_reliable'])
                self.assertFalse(result['clear'])
                self.assertIsNone(result['arrival_minutes'])

    def test_slow_motion_reports_observation_without_forecast_times(self):
        result = self.analyse(speed=7)
        self.assertTrue(result['publication_allowed'])
        self.assertFalse(result['forecast_reliable'])
        self.assertIsNone(result['arrival_minutes'])
        self.assertIsNone(result['end_minutes'])
        self.assertFalse(result['clear'])
        self.assertIn('移动缓慢', result['external'])

    def test_future_coverage_edge_is_unknown_not_clear(self):
        result = self.analyse(translated_pair(half=60))
        self.assertTrue(result['publication_allowed'])
        self.assertFalse(result['forecast_reliable'])
        self.assertEqual(result['steps'][-1]['status'], 'unknown')
        self.assertFalse(result['steps'][-1]['coverage'])
        self.assertIsNone(result['arrival_minutes'])
        self.assertFalse(result['clear'])

    def test_whole_circle_required_at_current_and_future_edges(self):
        east, north = grid(10)
        dbz = np.zeros_like(east)
        config = product.Config()
        for center in ((8, 0), (0, 8), (8, 8)):
            with self.subTest(center=center):
                value = product.sample_target(dbz, east, north, center, config)
                self.assertEqual(value['status'], 'unknown')
                self.assertFalse(value['coverage'])

    def test_sparse_echo_is_unknown_and_three_supported_pixels_qualify(self):
        east, north = grid(10)
        dbz = np.zeros_like(east)
        indexes = np.argwhere(east ** 2 + north ** 2 <= 9)
        for y, x in indexes[:2]:
            dbz[y, x] = 40
        sparse = product.sample_target(dbz, east, north, (0, 0), product.Config())
        self.assertEqual(sparse['status'], 'unknown')
        self.assertIsNone(sparse['rain'])
        y, x = indexes[2]
        dbz[y, x] = 40
        supported = product.sample_target(dbz, east, north, (0, 0), product.Config())
        self.assertEqual(supported['status'], 'strong')
        self.assertEqual(supported['rain_pixels'], 3)
        self.assertGreaterEqual(supported['rain_fraction'], 0.10)
        high_fraction = product.sample_target(dbz, east, north, (0, 0), product.Config(support_fraction=0.20))
        self.assertEqual(high_fraction['status'], 'unknown')

    def test_distinct_rain_and_strong_thresholds(self):
        result = self.analyse(translated_pair(value=30))
        self.assertEqual(result['arrival_minutes'], 35)
        self.assertIsNone(result['strong_arrival_minutes'])
        self.assertNotIn('较强回波开始影响：', result['external'])

    def test_ending_needs_four_clear_samples_spanning_fifteen_minutes(self):
        short = [step(0, 'rain'), step(5, 'clear'), step(10, 'clear'), step(15, 'clear')]
        self.assertIsNone(product.ending_minutes(short, True))
        self.assertEqual(product.ending_minutes(short + [step(20, 'clear')], True), 5)
        unknown = short + [step(20, 'unknown')]
        self.assertIsNone(product.ending_minutes(unknown, True))

    def test_temporary_gap_and_later_return_cannot_be_final_ending(self):
        steps = [step(0, 'rain'), step(5, 'clear'), step(10, 'clear'), step(15, 'rain'),
                 step(20, 'clear'), step(25, 'clear'), step(30, 'clear'), step(35, 'clear')]
        self.assertEqual(product.ending_minutes(steps, True), 20)
        self.assertEqual(product.ending_minutes([step(t, 'clear') for t in (0, 5, 10, 15)], True), 0)
        self.assertIsNone(product.ending_minutes([step(t, 'clear') for t in (0, 5, 10, 15)], False))

    def test_cross_midnight_window_and_arrival(self):
        now = datetime(2026, 12, 31, 23, 55, tzinfo=BJT)
        result = self.analyse(translated_pair(latest=now), now=now)
        self.assertEqual(result['event_date'], '2026-12-31')
        self.assertEqual(result['window_end'], '2027-01-01T01:55:00+08:00')
        self.assertIn('次日00:30', result['external'])
        self.assertEqual(result['steps'][-1]['time'], result['window_end'])

    def test_invalid_or_reverse_coordinate_grid_cannot_predict(self):
        for axis in ('east', 'north', 'nan'):
            frames = translated_pair()
            if axis == 'east':
                frames = [(t, d, -e, n) for t, d, e, n in frames]
            elif axis == 'north':
                frames = [(t, d, e, -n) for t, d, e, n in frames]
            else:
                t, d, e, n = frames[-1]
                frames[-1] = (t, d.astype(float), e, n)
                frames[-1][1][0, 0] = np.nan
            with self.subTest(axis=axis):
                result = self.analyse(frames)
                self.assertFalse(result['publication_allowed'])
                self.assertFalse(result['forecast_reliable'])

    def test_reliable_no_arrival_is_clear_only_with_known_motion_and_full_window(self):
        result = self.analyse(translated_pair(center=30))
        self.assertTrue(result['forecast_reliable'])
        self.assertTrue(result['clear'])
        self.assertIsNone(result['arrival_minutes'])
        self.assertIsNone(result['end_minutes'])

    def test_observed_clear_allows_blank_latest_without_claiming_future_clear(self):
        east, north = grid()
        blank = np.zeros_like(east)
        frames = [(LATEST - timedelta(minutes=6), blank, east, north), (LATEST, blank, east, north)]
        result = product.analyse(frames, now=LATEST)
        self.assertTrue(result['observed_clear'])
        self.assertFalse(result['forecast_reliable'])
        self.assertFalse(result['clear'])
        self.assertTrue(all(item['status'] == 'unknown' for item in result['steps']))
        self.assertIn('不表示未来两小时必然无雨', result['internal'])
        self.assertIn('回波新生', result['external'])

    def test_observed_clear_requires_no_supported_rain_within_one_hundred_km(self):
        east, north = grid()
        for pixels, value, expected in ((3, 25, False), (2, 35, True), (3, 24, True)):
            with self.subTest(pixels=pixels, value=value):
                dbz = np.zeros_like(east)
                dbz[150, 100:100 + pixels] = value  # 48–50 km upstream, outside current 3 km circle.
                frames = [(LATEST - timedelta(minutes=6), dbz, east, north), (LATEST, dbz, east, north)]
                with patch.object(radar_track, 'motion', return_value={'status': 'unavailable', 've': None, 'vn': None, 'reason': 'unknown motion'}):
                    result = product.analyse(frames, now=LATEST)
                self.assertEqual(result['observed_clear'], expected)
                self.assertFalse(result['clear'])
                self.assertFalse(result['forecast_reliable'])

    def test_observed_clear_rejects_stale_current_or_incomplete_nearby_coverage(self):
        for half, now in ((150, LATEST + timedelta(minutes=21)), (2, LATEST), (60, LATEST)):
            with self.subTest(half=half, now=now):
                east, north = grid(half)
                frames = [(LATEST, np.zeros_like(east), east, north)]
                result = product.analyse(frames, now=now)
                self.assertFalse(result['observed_clear'])

    def test_frame_fetch_is_limited_to_latest_six_images(self):
        paths = [f'radar/cappi/{(LATEST - timedelta(minutes=6 * index)):%Y%m%d%H%M}.png' for index in range(19, -1, -1)]
        response = Mock(text=' '.join('"' + path + '"' for path in paths))
        e, n = grid()
        with patch.object(requests, 'get', return_value=response), patch.object(product, 'cached_frame', return_value=(np.zeros_like(e), e, n)) as reader:
            frames = product.load_frames(now=LATEST)
        self.assertEqual(len(frames), 6)
        self.assertEqual(reader.call_count, 6)
        self.assertEqual(reader.call_args_list[0].args[0], paths[-1])
        self.assertEqual(frames[-1][0], LATEST)

    def test_old_frame_failures_preserve_latest_observation_without_fake_motion(self):
        response = Mock(text='"radar/cappi/202610131524.png" "radar/cappi/202610131530.png"')
        east, north = grid()
        latest = echo(east, north, center=0)
        def reader(path, cache_dir, config=None):
            if '1530' in path:
                return latest, east, north
            raise requests.RequestException('old image unavailable')
        with patch.object(requests, 'get', return_value=response), patch.object(product, 'cached_frame', side_effect=reader):
            frames = product.load_frames(now=LATEST)
        result = product.analyse(frames, now=LATEST)
        self.assertTrue(result['publication_allowed'])
        self.assertTrue(result['current']['rain'])
        self.assertFalse(result['forecast_reliable'])
        self.assertIsNone(result['arrival_minutes'])
        self.assertTrue(any('历史雷达帧' in check for check in result['checks']))

    def test_stale_feed_does_not_download_images(self):
        response = Mock(text='"radar/cappi/202610131500.png"')
        with patch.object(requests, 'get', return_value=response), patch.object(product, 'cached_frame', side_effect=forbidden):
            frames = product.load_frames(now=LATEST)
        result = product.analyse(frames, now=LATEST)
        self.assertFalse(result['publication_allowed'])

    def test_cache_reuses_frame_and_recovers_corrupt_archive(self):
        e, n = grid()
        path = 'radar/cappi/202610131530.png'
        with tempfile.TemporaryDirectory() as folder, patch.object(product.sz_radar, 'frame', return_value=(np.zeros_like(e), e, n)) as read:
            value = product.cached_frame(path, Path(folder))
            cached = product.cached_frame(path, Path(folder))
            self.assertEqual(read.call_count, 1)
            np.testing.assert_array_equal(value[0], cached[0])
            target = Path(folder) / (product.frame_cache_key(path) + '.npz')
            target.write_bytes(b'broken archive')
            product.cached_frame(path, Path(folder))
            self.assertEqual(read.call_count, 2)

    def test_immutable_output_and_metadata_compatibility(self):
        result = self.analyse()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'version'
            product.save_result(result, target)
            metadata = json.loads((target / 'radar_result.json').read_text())
            required = {'notice', 'external', 'brief', 'messages', 'message_format', 'internal', 'checks', 'blocking_checks', 'generated_at', 'issue', 'event_date',
                        'publication_allowed', 'product', 'forecast_reliable', 'radar_time', 'current', 'steps',
                        'arrival_minutes', 'strong_arrival_minutes', 'end_minutes', 'target', 'horizon_minutes'}
            self.assertTrue(required <= metadata.keys())
            self.assertEqual(metadata['product'], 'radar_nowcast')
            with self.assertRaises(FileExistsError):
                product.save_result(result, target)

    def test_cli_dry_run_has_no_ec_or_push_call(self):
        import ec_event_point
        with tempfile.TemporaryDirectory() as folder, patch.object(product, 'load_frames', return_value=translated_pair()), \
                patch.object(radar_track, 'motion', return_value=trusted_motion()), \
                patch.object(ec_event_point, 'extract', side_effect=forbidden), patch.object(ec_event_point, 'latest_run', side_effect=forbidden), redirect_stdout(io.StringIO()):
            code = product.main(['--output-dir', str(Path(folder) / 'version'), '--now', LATEST.isoformat()])
            self.assertEqual(code, 0)
            metadata = json.loads((Path(folder) / 'version' / 'radar_result.json').read_text())
            self.assertEqual(metadata['generated_at'], LATEST.isoformat())

    def test_live_generation_rechecks_clock_after_frame_fetch(self):
        later = LATEST + timedelta(minutes=21)
        with tempfile.TemporaryDirectory() as folder, patch.object(product, 'load_frames', return_value=translated_pair()), \
                patch.object(product, 'datetime') as clock, redirect_stdout(io.StringIO()):
            clock.now.side_effect = [LATEST, later]
            # as_bjt needs datetime's actual type even when the wall clock is mocked.
            with patch.object(product, 'as_bjt', side_effect=lambda value: value if isinstance(value, datetime) else datetime.fromisoformat(value)):
                code = product.main(['--output-dir', str(Path(folder) / 'version')])
            self.assertEqual(code, 2)
            metadata = json.loads((Path(folder) / 'version' / 'radar_result.json').read_text())
            self.assertEqual(metadata['generated_at'], later.isoformat())
            self.assertFalse(metadata['publication_allowed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
