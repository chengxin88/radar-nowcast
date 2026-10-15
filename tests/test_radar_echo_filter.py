"""Public presentation selection never changes original radar/alert evidence."""
import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import radar_echo_filter as echo_filter
import radar_monitor
import radar_nowcast as radar
import radar_track
import rolling_radar


NOW = datetime(2026, 10, 15, 13, 0, tzinfo=radar.BJT)


def make(field_fn=None, spacing=1., half=155., moving=True):
    axis = np.arange(-half, half+spacing/2, spacing)
    east, north = np.meshgrid(axis, axis[::-1])
    field = np.zeros(east.shape) if field_fn is None else field_fn(east, north)
    frames = [(NOW-timedelta(minutes=6), field.copy(), east, north),
              (NOW, field.copy(), east, north)]
    config = radar.Config()
    motion = {'status': 'ok', 've': 60., 'vn': 0., 'correlation': .9, 'peak_margin': .3}
    if not moving:
        motion = {'status': 'unavailable', 've': None, 'vn': None, 'reason': '移动不明'}
    with patch.object(radar_track, 'motion', return_value=motion):
        result = radar.analyse(frames, now=NOW, config=config)
    result['rolling_radar'] = rolling_radar.estimate(frames, result, config)
    result['radar_monitor'] = radar_monitor.estimate(frames, result, config)
    result['radar_outlook'] = radar_monitor.outlook(result)
    return frames, result, config


class PublicEchoFilterTests(unittest.TestCase):
    def test_truly_clear_complete_circle_stays_clear_without_fabricated_forecast(self):
        frames, result, config = make(moving=False)
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'clear')
        self.assertTrue(summary['complete'])
        self.assertEqual((summary['raw_pixels'], summary['kept_pixels'], summary['filtered_pixels']), (0, 0, 0))
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')
        json.dumps(view, allow_nan=False)

    def test_twenty_scattered_pixels_are_not_combined_into_a_publishable_echo(self):
        def field(e, n):
            out = np.zeros(e.shape)
            for longitude in range(20, 120, 10):
                out[(e == longitude) & ((n == 0) | (n == 20))] = 25.
            return out
        frames, result, config = make(field)
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'filtered_only')
        self.assertEqual((summary['raw_pixels'], summary['kept_pixels']), (20, 0))
        self.assertIsNone(summary['nearest'])
        self.assertEqual(view['radar_outlook']['motion']['status'], 'unknown')
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')
        self.assertFalse(view['rolling_radar']['timing_reliable'])

    def test_small_twenty_dbz_echo_and_single_extreme_pixel_are_filtered(self):
        for value in (20., 70.):
            with self.subTest(value=value):
                frames, result, config = make(lambda e,n: np.where((e == 40) & (n == 0), value, 0.))
                summary, view = echo_filter.apply(frames, result, config)
                self.assertEqual(summary['status'], 'filtered_only')
                self.assertEqual(summary['kept_pixels'], 0)
                self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_organized_twenty_five_dbz_component_survives(self):
        frames, result, config = make(lambda e,n: np.where((e >= 40) & (e <= 42) & (abs(n) <= 1), 25., 0.))
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'echo')
        self.assertEqual(summary['kept_pixels'], 9)
        self.assertEqual(summary['nearest']['bearing'], '东')
        self.assertEqual(view['radar_outlook']['motion']['status'], 'estimated')

    def test_new_strong_three_pixel_core_does_not_wait_for_previous_frame(self):
        frames, result, config = make(lambda e,n: np.where((e == 40) & (abs(n) <= 1), 35., 0.))
        frames[0][1][:] = 0
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'echo')
        self.assertEqual(summary['kept_pixels'], 3)
        self.assertFalse(summary['policy']['temporal_confirmation_required'])

    def test_area_uses_real_grid_spacing_not_a_fixed_pixel_count(self):
        for spacing, expected in ((.5, 'filtered_only'), (2., 'echo')):
            with self.subTest(spacing=spacing):
                def field(e, n):
                    return np.where((abs(e-40) <= spacing/10) & (abs(n) <= spacing+spacing/10), 25., 0.)
                frames, result, config = make(field, spacing=spacing, half=156.)
                summary, _ = echo_filter.apply(frames, result, config)
                self.assertEqual(summary['status'], expected)

    def test_actual_grid_edge_fragment_is_kept_and_cannot_create_non_impact(self):
        frames, result, config = make(lambda e,n: np.where((e == 150) & (n == 0), 25., 0.), half=150.)
        summary, view = echo_filter.apply(frames, result, config)
        self.assertTrue(summary['boundary_truncated'])
        self.assertEqual(summary['status'], 'echo')
        self.assertEqual(summary['kept_pixels'], 1)
        self.assertNotEqual(view['radar_outlook']['impact']['status'], 'not_expected')

    def test_incomplete_150km_circle_does_not_become_public_clear(self):
        frames, result, config = make(half=100.)
        summary, _ = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'unavailable')
        self.assertFalse(summary['complete'])
        self.assertIsNone(summary['raw_pixels'])

    def test_only_removed_noise_cannot_turn_raw_possible_into_no_impact(self):
        frames, result, config = make(lambda e,n: np.where((e == -40) & (abs(n) <= 1), 25., 0.))
        self.assertEqual(result['radar_outlook']['impact']['status'], 'possible')
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'filtered_only')
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_mixed_scene_drops_noise_arrival_but_keeps_real_incoming_component(self):
        def field(e,n):
            noise = (e == -20) & (abs(n) <= 1)
            rain = (e >= -65) & (e <= -60) & (abs(n) <= 4)
            return np.where(noise | rain, 30., 0.)
        frames, result, config = make(field)
        summary, view = echo_filter.apply(frames, result, config)
        self.assertGreater(summary['filtered_pixels'], 0)
        self.assertEqual(view['radar_outlook']['impact']['status'], 'possible')
        self.assertGreater(view['radar_outlook']['impact']['arrival_minutes'],
                           result['radar_outlook']['impact']['arrival_minutes'])
        self.assertGreater(view['rolling_radar']['arrival_minutes'], result['rolling_radar']['arrival_minutes'])

    def test_filter_does_not_introduce_non_impact_in_mixed_incoming_noise_and_outgoing_rain(self):
        frames, result, config = make(lambda e,n: np.where(
            ((e == -20) & (abs(n) <= 1)) | ((e >= 40) & (e <= 45) & (abs(n) <= 4)), 30., 0.))
        self.assertEqual(result['radar_outlook']['impact']['status'], 'possible')
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'echo')
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_original_unknown_path_remains_unknown_after_selection(self):
        frames, result, config = make(lambda e,n: np.where((e >= 40) & (e <= 45) & (abs(n) <= 4), 30., 0.))
        result['steps'][-1].update(coverage=False, status='unknown', rain=None, strong=None)
        result['forecast_reliable'] = False
        summary, view = echo_filter.apply(frames, result, config)
        self.assertFalse(view['steps'][-1]['coverage'])
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_recent_measured_rain_survives_the_weak_echo_display_gate(self):
        frames, result, config = make(lambda e,n: np.where((e == 0) & (abs(n) <= 1), 20., 0.))
        result['official'] = {'station_observation_requested': True, 'station_observation': {'rain': {
            'time': (NOW-timedelta(minutes=2)).isoformat(), 'station_name': '市民中心',
            'rain_5m': .2, 'rain_10m': .3}}}
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'filtered_only')
        self.assertEqual(view['radar_outlook']['impact']['status'], 'ongoing')
        self.assertEqual(view['radar_outlook']['impact']['basis'], 'station_rain_observation')
        result['official']['station_observation']['rain']['time'] = (NOW-timedelta(minutes=21)).isoformat()
        _, view = echo_filter.apply(frames, result, config)
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_original_result_frames_and_alarm_fields_are_unchanged_and_view_is_json(self):
        frames, result, config = make(lambda e,n: np.where((e == -20) & (abs(n) <= 1), 30., 0.))
        original = copy.deepcopy(result)
        original_frames = [tuple(v.copy() if isinstance(v, np.ndarray) else v for v in frame) for frame in frames]
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(result, original)
        for before, after in zip(original_frames, frames):
            for old, new in zip(before[1:], after[1:]):
                np.testing.assert_array_equal(old, new)
        json.dumps((summary, view), allow_nan=False)
        self.assertNotIn('field', view)
        self.assertEqual(view['observed_clear'], result['observed_clear'])

    def test_stale_and_wrong_time_inputs_are_unavailable(self):
        for variant in ('stale', 'time'):
            with self.subTest(variant=variant):
                frames, result, config = make()
                if variant == 'stale':
                    result['fresh'] = False
                else:
                    result['radar_time'] = (NOW-timedelta(minutes=6)).isoformat()
                summary, view = echo_filter.apply(frames, result, config)
                self.assertEqual(summary['status'], 'unavailable')
                self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_verified_slow_strong_echo_keeps_slow_state_without_future_timing(self):
        for speed in (0., 3.):
            with self.subTest(speed=speed):
                frames, result, config = make(lambda e,n: np.where((abs(e) < 5) & (abs(n) < 5), 40., 0.))
                result['motion'].update(speed_kmh=speed, east_kmh=speed, north_kmh=0.)
                result['forecast_reliable'] = False
                summary, view = echo_filter.apply(frames, result, config)
                self.assertEqual(summary['status'], 'echo')
                self.assertEqual(view['motion']['status'], 'ok')
                self.assertEqual(view['motion']['speed_kmh'], speed)
                self.assertEqual(view['rolling_radar']['status'], 'slow')
                self.assertIn('移动缓慢', view['rolling_radar']['reason'])
                self.assertIsNone(view['rolling_radar']['arrival_minutes'])
                self.assertIsNone(view['rolling_radar']['strong_end_minutes'])
                self.assertTrue(all(step['status'] == 'unknown' for step in view['steps']))
                self.assertFalse(view['forecast_reliable'])
                self.assertEqual(view['radar_outlook']['impact']['status'], 'ongoing')

    def test_slow_motion_exception_does_not_accept_inconsistent_vectors_or_noise(self):
        frames, result, config = make(lambda e,n: np.where((abs(e) < 5) & (abs(n) < 5), 40., 0.))
        result['motion'].update(speed_kmh=3., east_kmh=30., north_kmh=0.)
        _, view = echo_filter.apply(frames, result, config)
        self.assertEqual(view['motion']['status'], 'unavailable')
        self.assertNotEqual(view['rolling_radar']['status'], 'slow')
        frames, result, config = make(lambda e,n: np.where((e == 40) & (abs(n) <= 1), 20., 0.))
        result['motion'].update(speed_kmh=3., east_kmh=3., north_kmh=0.)
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'filtered_only')
        self.assertEqual(view['motion']['status'], 'unavailable')
        self.assertEqual(view['radar_outlook']['impact']['status'], 'unknown')

    def test_filtered_circle_does_not_hide_supported_incoming_echo_outside_150_km(self):
        def field(e, n):
            noise = (e == 40) & (n == 0)
            incoming = (abs(e-120) <= 3) & (abs(n-120) <= 3)
            return np.where(incoming, 40., np.where(noise, 20., 0.))
        frames, result, config = make(field, half=180.)
        speed = 100.
        component = -speed / np.sqrt(2)
        result['motion'].update(speed_kmh=speed, east_kmh=component, north_kmh=component)
        # Keep the original path's physical coverage while changing the
        # synthetic tracked vector to point from the northeast toward target.
        steps = []
        for minute in range(0, 121, 5):
            center = (-component*minute/60, -component*minute/60)
            sample = radar.sample_target(frames[-1][1], frames[-1][2], frames[-1][3], center, config)
            steps.append(dict(minute=minute, time=(NOW+timedelta(minutes=minute)).isoformat(), **sample))
        result['steps'] = steps
        result['forecast_reliable'] = all(step['status'] != 'unknown' for step in steps)
        result['rolling_radar'] = rolling_radar.estimate(frames, result, config)
        result['radar_outlook'] = radar_monitor.outlook(result)
        self.assertEqual(result['radar_outlook']['impact']['status'], 'possible')
        summary, view = echo_filter.apply(frames, result, config)
        self.assertEqual(summary['status'], 'filtered_only')
        self.assertEqual((summary['raw_pixels'], summary['kept_pixels']), (1, 0))
        self.assertIsNone(summary['nearest'])
        self.assertTrue(summary['outside_incoming'])
        self.assertEqual(view['radar_outlook']['impact']['status'], 'possible')
        self.assertEqual(view['radar_outlook']['impact']['basis'], 'target_path_extrapolation')
        self.assertTrue(90 <= view['radar_outlook']['impact']['arrival_minutes'] <= 110)
        self.assertEqual(view['radar_outlook']['motion']['status'], 'estimated')


if __name__ == '__main__':
    unittest.main()
