"""Offline monitoring geometry and freshness checks, independent of timing."""
import copy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import radar_monitor as monitor
import radar_nowcast as radar


NOW = datetime(2026, 10, 15, 11, 0, tzinfo=radar.BJT)


def fixture(config=None, half=160, at=NOW):
    config = config or radar.Config()
    axis = np.arange(-half, half + 1, dtype=float)
    east, north = np.meshgrid(axis, axis[::-1])
    cx, cy = radar.target_center(config)
    east, north = east + cx, north + cy
    field = np.zeros(east.shape)
    frames = [(at, field, east, north)]
    result = {
        'target': {key: getattr(config, key) for key in ('name', 'lat', 'lon', 'radius_km')},
        'fresh': True, 'publication_allowed': True, 'current': {'coverage': True},
        'radar_time': at.isoformat(), 'generated_at': NOW.isoformat(),
    }
    return config, frames, result


def add_echo(frames, center=120, northward=0, count=3):
    _, field, east, north = frames[-1]
    distance = np.hypot(east - center, north - northward)
    indexes = np.argsort(distance, axis=None)[:count]
    field.flat[indexes] = 20.


class RadarMonitorTests(unittest.TestCase):
    def test_complete_circle_is_clear_without_echo_and_does_not_mutate_inputs(self):
        config, frames, result = fixture()
        saved = copy.deepcopy(result)
        field = frames[0][1].copy()
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'clear')
        self.assertTrue(out['complete'])
        self.assertEqual((out['radius_km'], out['threshold_dbz'], out['echo_pixels']), (150., 20., 0))
        self.assertEqual(out['observed_at'], NOW.isoformat())
        self.assertIsNone(out['nearest'])
        self.assertEqual(result, saved)
        np.testing.assert_array_equal(frames[0][1], field)

    def test_confirmed_echo_between_100_and_150_km_is_reported_without_impact_claim(self):
        config, frames, result = fixture()
        add_echo(frames)
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'echo')
        self.assertEqual(out['echo_pixels'], 3)
        self.assertTrue(100 < out['nearest']['distance_km'] < 150)
        self.assertEqual(out['nearest']['bearing'], '东')
        self.assertNotIn('arrival_minutes', out)

    def test_echo_outside_monitoring_circle_does_not_count(self):
        config, frames, result = fixture()
        add_echo(frames, center=155)
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'clear')
        self.assertEqual(out['echo_pixels'], 0)

    def test_sparse_echo_is_not_clear(self):
        for count in (1, 2):
            with self.subTest(count=count):
                config, frames, result = fixture()
                add_echo(frames, count=count)
                out = monitor.estimate(frames, result, config)
                self.assertEqual(out['status'], 'sparse')
                self.assertTrue(out['complete'])
                self.assertEqual(out['echo_pixels'], count)
                self.assertIsNone(out['nearest'])

    def test_total_pixel_support_matches_existing_project_convention(self):
        config, frames, result = fixture()
        for e, n in ((40, 0), (80, 0), (120, 0)):
            add_echo(frames, center=e, northward=n, count=1)
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'echo')
        self.assertEqual(out['support_scope'], 'monitor_circle_total_pixels')
        self.assertEqual(out['nearest']['distance_km'], 40)

    def test_insufficient_circle_coverage_is_unavailable_not_clear(self):
        config, frames, result = fixture(half=149)
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'unavailable')
        self.assertFalse(out['complete'])
        self.assertIsNone(out['echo_pixels'])

    def test_stale_or_unapproved_result_is_unavailable(self):
        for key in ('fresh', 'publication_allowed'):
            with self.subTest(key=key):
                config, frames, result = fixture()
                result[key] = False
                self.assertEqual(monitor.estimate(frames, result, config)['status'], 'unavailable')
        for age in (21, -1):
            with self.subTest(age=age):
                config, frames, result = fixture(at=NOW-timedelta(minutes=age))
                self.assertEqual(monitor.estimate(frames, result, config)['status'], 'unavailable')

    def test_wrong_target_cannot_reuse_a_valid_result(self):
        config, frames, result = fixture()
        for key, value in (('lat', config.lat + .1), ('lon', config.lon + .1), ('radius_km', 8)):
            with self.subTest(key=key):
                changed = copy.deepcopy(result)
                changed['target'][key] = value
                self.assertEqual(monitor.estimate(frames, changed, config)['status'], 'unavailable')

    def test_shifted_target_uses_configured_center_and_relative_bearing(self):
        config = radar.Config(name='异地', lat=23., lon=115.)
        config, frames, result = fixture(config=config)
        cx, cy = radar.target_center(config)
        add_echo(frames, center=cx-125, northward=cy)
        out = monitor.estimate(frames, result, config)
        self.assertEqual(out['status'], 'echo')
        self.assertEqual(out['nearest']['bearing'], '西')
        self.assertGreaterEqual(out['nearest']['distance_km'], 124)
        self.assertLessEqual(out['nearest']['distance_km'], 125)

    def test_mismatched_frame_time_and_empty_batch_are_unavailable(self):
        config, frames, result = fixture()
        result['radar_time'] = (NOW-timedelta(minutes=6)).isoformat()
        self.assertEqual(monitor.estimate(frames, result, config)['status'], 'unavailable')
        self.assertEqual(monitor.estimate([], result, config)['status'], 'unavailable')

    def test_latest_frame_is_selected_even_when_input_is_unsorted(self):
        config, frames, result = fixture()
        add_echo(frames)
        old = (NOW-timedelta(minutes=6), np.zeros_like(frames[0][1]), *frames[0][2:])
        out = monitor.estimate(frames + [old], result, config)
        self.assertEqual(out['status'], 'echo')
        self.assertEqual(out['observed_at'], NOW.isoformat())

    def test_invalid_grid_values_shape_or_spacing_are_unavailable(self):
        for variant in ('missing', 'shape', 'spacing'):
            with self.subTest(variant=variant):
                config, frames, result = fixture()
                at, field, east, north = frames[0]
                if variant == 'missing':
                    field[0, 0] = np.nan
                elif variant == 'shape':
                    east = east[:, :-1]
                else:
                    east[:, 20] += .5
                self.assertEqual(monitor.estimate([(at, field, east, north)], result, config)['status'], 'unavailable')

    def test_invalid_radius_or_support_never_produces_clear(self):
        config, frames, result = fixture()
        for radius in (0, -150, float('nan'), float('inf'), True, None):
            with self.subTest(radius=radius):
                self.assertEqual(monitor.estimate(frames, result, config, radius)['status'], 'unavailable')
        config = radar.Config(support_pixels=2.5)
        self.assertEqual(monitor.estimate(frames, result, config)['status'], 'unavailable')


class RadarOutlookTests(unittest.TestCase):
    def result(self):
        _, _, result = fixture()
        result.update(
            forecast_reliable=True, horizon_minutes=120, step_minutes=5,
            motion={'status': 'ok', 'speed_kmh': 60., 'east_kmh': -60., 'north_kmh': 0.},
            current={'coverage': True, 'status': 'clear', 'rain': False, 'strong': False},
            steps=[{'minute': minute, 'coverage': True, 'status': 'clear', 'rain': False, 'strong': False}
                   for minute in range(0, 121, 5)],
            rolling_radar={'current_rain': False, 'current_strong': False, 'current_weak': False,
                           'timing_reliable': False, 'arrival_minutes': None, 'strong_arrival_minutes': None},
        )
        return result

    def test_complete_reliable_clear_path_allows_only_conditional_no_impact(self):
        result = self.result()
        saved = copy.deepcopy(result)
        out = monitor.outlook(result)
        self.assertEqual(out['motion']['status'], 'estimated')
        self.assertEqual(out['motion']['direction_to'], '西')
        self.assertEqual(out['motion']['speed_kmh'], 60)
        self.assertEqual(out['impact']['status'], 'not_expected')
        self.assertIn('仅按现有回波整体平移', out['impact']['reason'])
        self.assertIn('新生', out['impact']['reason'])
        self.assertEqual(result, saved)

    def test_150km_clear_alone_does_not_prove_two_hour_non_impact(self):
        result = self.result()
        result['radar_monitor'] = {'status': 'clear', 'complete': True}
        result['motion'] = {'status': 'unavailable'}
        result['forecast_reliable'] = False
        result['steps'] = []
        out = monitor.outlook(result)
        self.assertEqual(out['motion']['status'], 'unknown')
        self.assertEqual(out['impact']['status'], 'unknown')

    def test_nearest_echo_distance_or_direction_cannot_decide_target_impact(self):
        result = self.result()
        result['radar_monitor'] = {'status': 'echo', 'complete': True,
                                   'nearest': {'distance_km': 138., 'bearing': '西'}}
        result['steps'][-1].update(status='unknown', rain=None, strong=None, coverage=False)
        result['forecast_reliable'] = False
        self.assertEqual(monitor.outlook(result)['impact']['status'], 'unknown')

    def test_any_valid_wet_step_preserves_possible_impact_with_later_missing_coverage(self):
        result = self.result()
        result['steps'][6].update(status='rain', rain=True)
        result['steps'][12].update(status='unknown', rain=None, strong=None, coverage=False)
        result['forecast_reliable'] = False
        out = monitor.outlook(result)
        self.assertEqual(out['impact']['status'], 'possible')
        self.assertEqual(out['impact']['arrival_minutes'], 30.)
        self.assertEqual(out['impact']['basis'], 'target_path_extrapolation')

    def test_partial_invalid_wet_step_does_not_count_as_positive_evidence(self):
        result = self.result()
        result['steps'][6].update(status='rain', rain=True, coverage=False)
        result['forecast_reliable'] = False
        self.assertEqual(monitor.outlook(result)['impact']['status'], 'unknown')

    def test_current_rain_strong_and_weak_are_separate_from_future_impact(self):
        for kind in ('rain', 'strong', 'weak'):
            with self.subTest(kind=kind):
                result = self.result()
                result['motion'] = {'status': 'unavailable'}
                if kind == 'weak':
                    result['rolling_radar']['current_weak'] = True
                else:
                    result['current'].update(status=kind, rain=True, strong=kind == 'strong')
                out = monitor.outlook(result)
                self.assertEqual(out['impact']['status'], 'ongoing')
                self.assertEqual(out['impact']['current_kind'], kind)
                self.assertEqual(out['motion']['status'], 'unknown')

    def test_nearby_rolling_rain_is_not_claimed_inside_the_target(self):
        result = self.result()
        result['rolling_radar'].update(current_rain=True, nearby_strong=True)
        out = monitor.outlook(result)
        self.assertEqual(out['impact']['status'], 'possible')
        self.assertEqual(out['impact']['basis'], 'nearby_observation')
        self.assertIsNone(out['impact']['arrival_minutes'])

    def test_reliable_corridor_arrival_survives_partial_target_path(self):
        result = self.result()
        result['steps'] = []
        result['forecast_reliable'] = False
        result['rolling_radar'].update(timing_reliable=True, arrival_minutes=45.)
        out = monitor.outlook(result)
        self.assertEqual(out['impact']['status'], 'possible')
        self.assertEqual(out['impact']['arrival_minutes'], 45.)
        self.assertEqual(out['impact']['basis'], 'upstream_corridor')

    def test_unreliable_or_out_of_horizon_corridor_time_is_not_used(self):
        for reliable, minute in ((False, 45), (True, 150), (True, float('nan'))):
            with self.subTest(reliable=reliable, minute=minute):
                result = self.result()
                result['forecast_reliable'] = False
                result['rolling_radar'].update(timing_reliable=reliable, arrival_minutes=minute)
                self.assertEqual(monitor.outlook(result)['impact']['status'], 'unknown')

    def test_invalid_or_ground_wind_like_motion_never_allows_no_impact(self):
        variants = (
            {'status': 'unavailable'},
            {'status': 'ok', 'speed_kmh': 0., 'east_kmh': 0., 'north_kmh': 0.},
            {'status': 'ok', 'speed_kmh': 300., 'east_kmh': 300., 'north_kmh': 0.},
            {'status': 'ok', 'speed_kmh': float('nan'), 'east_kmh': 60., 'north_kmh': 0.},
            {'status': 'ok', 'speed_kmh': 60., 'east_kmh': -30., 'north_kmh': 0.},
            {'status': 'ok', 'wind_direction': '东', 'wind_level': '3'},
        )
        for vector in variants:
            with self.subTest(vector=vector):
                result = self.result()
                result['motion'] = vector
                out = monitor.outlook(result)
                self.assertEqual(out['motion']['status'], 'unknown')
                self.assertEqual(out['impact']['status'], 'unknown')

    def test_anomalous_motion_preserves_positive_risk_but_drops_arrival_time(self):
        result = self.result()
        result['motion']['speed_kmh'] = 600.
        result['steps'][6].update(status='strong', rain=True, strong=True)
        out = monitor.outlook(result)
        self.assertEqual(out['motion']['status'], 'unknown')
        self.assertEqual(out['impact']['status'], 'possible')
        self.assertIsNone(out['impact']['arrival_minutes'])

    def test_missing_duplicate_or_unknown_steps_never_count_as_complete_clear(self):
        for variant in ('missing', 'duplicate', 'unknown'):
            with self.subTest(variant=variant):
                result = self.result()
                if variant == 'missing':
                    result['steps'].pop()
                elif variant == 'duplicate':
                    result['steps'][-1]['minute'] = 115
                else:
                    result['steps'][-1].update(status='unknown', rain=None, strong=None)
                self.assertEqual(monitor.outlook(result)['impact']['status'], 'unknown')

    def test_stale_or_missing_metadata_invalidates_even_positive_evidence(self):
        for variant in ('stale', 'missing_time', 'missing_target'):
            with self.subTest(variant=variant):
                result = self.result()
                result['steps'][6].update(status='rain', rain=True)
                if variant == 'stale':
                    result['fresh'] = False
                elif variant == 'missing_time':
                    del result['radar_time']
                else:
                    del result['target']
                out = monitor.outlook(result)
                self.assertEqual(out['motion']['status'], 'unknown')
                self.assertEqual(out['impact']['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
