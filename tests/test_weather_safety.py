"""Offline safety regression tests; no meteorological or push endpoints are called."""
import copy
from contextlib import redirect_stdout
import importlib.util
import json
import io
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))


def forbidden(*args, **kwargs):
    raise AssertionError('This regression suite must not access a network or push endpoint')


# Tests use synthetic guidance, never a live GRIB download. These import-only
# substitutes permit offline regression checks on the bundled local runtime;
# a deployment must separately install and smoke-test the real dependencies.
IMPORT_SUBSTITUTES = []
if importlib.util.find_spec('requests') is None:
    module = types.ModuleType('requests')
    module.get = module.head = module.post = forbidden
    module.RequestException = RuntimeError
    module.Session = lambda: types.SimpleNamespace(post=forbidden, proxies={}, trust_env=True)
    sys.modules['requests'] = module
    IMPORT_SUBSTITUTES.append('requests')
if importlib.util.find_spec('eccodes') is None:
    module = types.ModuleType('eccodes')
    for name in ('codes_get', 'codes_get_values', 'codes_grib_new_from_file', 'codes_release'):
        setattr(module, name, forbidden)
    sys.modules['eccodes'] = module
    IMPORT_SUBSTITUTES.append('eccodes')
if importlib.util.find_spec('ecmwf') is None:
    package = types.ModuleType('ecmwf')
    package.__path__ = []
    module = types.ModuleType('ecmwf.opendata')
    module.Client = forbidden
    sys.modules['ecmwf'] = package
    sys.modules['ecmwf.opendata'] = module
    IMPORT_SUBSTITUTES.append('ecmwf-opendata')

import numpy as np
import ec_event_point as ec_point
import event_briefing as briefing
import radar_track
import sz_lightning
import sz_official
import sz_radar

BJT = ec_point.BJT
DAY = datetime(2026, 10, 13, tzinfo=BJT)
NOW = DAY + timedelta(hours=15.5)
ISSUE = DAY + timedelta(hours=16)


def official_fixture():
    result = {key: {'cache_time': NOW.isoformat(), 'data': {}} for key in sz_official.FEEDS}
    result['alarm']['data'] = {'ingnalNum': '0', 'subAlarm': [], 'alarmInfo': ''}
    result['observation']['data'] = {'date': (NOW - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M'),
                                    't': 25, 'u': 50, 'wf': 2, 'wd': 45, 'v': 20, 'h24r': 0, 'h01r': 0}
    result['futian']['data'] = {'hoursweather': [
        {'forecasttime': (DAY + timedelta(hours=h)).strftime('%Y-%m-%d %H:%M'),
         'pubTime': NOW.strftime('%Y-%m-%d %H:%M'), 'state': '晴天', 'r': 0,
         'minT': 24, 'maxT': 26, 'h': 50, 'wspeed': 2, 'ws': 2, 'wd': '东北'} for h in (20, 21, 22)]}
    result['tenday']['data'] = {'daynew': [{'reportTime': DAY.strftime('%Y-%m-%d'),
        'pubTime': NOW.strftime('%Y-%m-%d %H:%M'), 'report': '晴天', 'minT': 24, 'maxT': 26, 'wd': '东北', 'ws': '2'}]}
    return result


def ec_fixture():
    rows = []
    for h in (20, 23):
        rows.append({'valid_bjt': (DAY + timedelta(hours=h)).strftime('%Y-%m-%d %H:%M'),
                     'rain_point_mm': 0, 'rain_area_max_mm': 0, 'rain_area_ge1_fraction': 0,
                     'tcc_percent': 40, 'mucape_point': 0, 'mucape_area_max': 0,
                     'gust_point_ms': 3, 'gust_area_max_ms': 4, 't2m_c': 25, 'rh2m_percent': 50,
                     'wind_dir': '东北', 'wind_ms': 2, 'wind_level': 2})
    return {'rows': rows, 'run_bjt': '2026-10-13 08:00', 'site': {'lat': 22.543, 'lon': 114.059}}


def radar_fixture():
    return {'first': (NOW - timedelta(minutes=12)).strftime('%Y-%m-%d %H:%M'),
            'last': NOW.strftime('%Y-%m-%d %H:%M'), 'strong': None, 'moderate': None}


class WeatherSafetyTests(unittest.TestCase):
    def setUp(self):
        import requests
        self.network_guards = [patch.object(requests, method, side_effect=forbidden) for method in ('get', 'head', 'post')]
        for guard in self.network_guards:
            guard.start()
            self.addCleanup(guard.stop)
        self.push_guard = patch.object(briefing, 'send', side_effect=forbidden)
        self.push_guard.start()
        self.addCleanup(self.push_guard.stop)

    def render(self, official=None, ec=None, radar=None, now=NOW):
        return briefing.build(DAY, ISSUE, ec or ec_fixture(), official or official_fixture(),
                              radar=radar if radar is not None else radar_fixture(),
                              lightning={'disabled': True, 'error': 'cloud disabled'}, now=now)

    def test_post_18_nowcast_and_manual_target_current_activity(self):
        late = DAY + timedelta(hours=20)
        self.assertEqual(briefing.activity_day(late, 'nowcast'), DAY)
        self.assertEqual(briefing.activity_day(late, 'manual'), DAY)
        self.assertEqual(briefing.activity_day(late, 'scheduled'), DAY + timedelta(days=1))
        self.assertEqual(briefing.activity_day(ISSUE, 'scheduled'), DAY)
        with self.assertRaises(ValueError):
            briefing.activity_day(DAY + timedelta(hours=18), 'scheduled')

    def test_negative_weather_and_explicit_possibility(self):
        for text in ('无明显降雨', '没有暴雨', '暂无雷电', '无雷电、大风和降雨', '降雨可能性较小'):
            with self.subTest(text=text):
                self.assertFalse(briefing.has_rain(text))
                self.assertFalse(briefing.has_thunder(text))
        self.assertTrue(briefing.has_thunder('不排除局地雷阵雨'))
        self.assertTrue(briefing.has_rain('没有大雨，但有小雨'))
        self.assertEqual(briefing.official_rain_index({'state': '晴天无雨', 'r': 10}), 0)

    def test_negative_nowcast_does_not_upgrade(self):
        item = {'d': NOW.strftime('%Y-%m-%d %H:%M'), 'prompt': '【短临预报】今晚无明显降雨，无雷电和大风。'}
        cast = briefing.nowcast({'tips': {'important': item}}, DAY, ISSUE, NOW)
        self.assertEqual((cast['rain'], cast['thunder'], cast['gust']), (0, False, None))

    def test_explicit_thunder_with_dry_stable_ec_is_at_least_middle(self):
        official = official_fixture()
        official['tips']['data'] = {'important': {'d': NOW.strftime('%Y-%m-%d %H:%M'), 'prompt': '【短临预报】今晚可能伴有雷电。'}}
        external, _, _, _ = self.render(official)
        self.assertIn('降雨 低｜雷电 中', external)

    def test_original_heavy_rain_gust_and_cape_thresholds_preserved(self):
        ec = ec_fixture()
        official = official_fixture()
        official['futian']['data']['hoursweather'] = []
        for row in ec['rows']:
            row.update(rain_point_mm=3, rain_area_ge1_fraction=0.5, mucape_point=2000, gust_point_ms=13.9)
        external, _, _, _ = self.render(official, ec)
        self.assertIn('降雨 较高｜雷电 较高｜大风 较高', external)
        self.assertIn('不适宜开展灯光秀活动', external)

    def test_alarm_missing_or_old_never_claims_no_warning(self):
        for bad in ({'error': 'offline'}, {'cache_time': (NOW - timedelta(minutes=31)).isoformat(), 'data': {'ingnalNum': '0'}},
                    {'cache_time': NOW.isoformat(), 'data': {}}):
            with self.subTest(feed=bad):
                official = official_fixture()
                official['alarm'] = bad
                external, _, _, _ = self.render(official)
                self.assertNotIn('目前福田区无生效的气象预警信号', external)
                self.assertIn('预警资料缺测或过期', external)
                self.assertTrue(briefing.publication_checks(official, DAY, ISSUE, radar_fixture(), NOW))

    def test_actual_now_instead_of_nominal_issue_controls_freshness(self):
        official = official_fixture()
        actual_late = NOW + timedelta(hours=2)
        self.assertIn('error', sz_official.validate_feed('alarm', official['alarm'], actual_late))
        external, _, _, _ = self.render(official, now=actual_late)
        self.assertIn('预警资料缺测或过期', external)
        item = {'d': (actual_late + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M'), 'prompt': '【短临】今晚有暴雨。'}
        self.assertIsNone(briefing.nowcast({'tips': {'important': item}}, DAY, ISSUE, actual_late))

    def test_observation_time_checked_separately_from_cache(self):
        feed = official_fixture()['observation']
        feed['data']['date'] = (NOW - timedelta(minutes=91)).strftime('%Y-%m-%d %H:%M')
        self.assertIn('error', sz_official.validate_feed('observation', feed, NOW))
        feed['data']['date'] = (NOW + timedelta(minutes=6)).strftime('%Y-%m-%d %H:%M')
        self.assertIn('error', sz_official.validate_feed('observation', feed, NOW))

    def test_fresh_cache_cannot_hide_stale_forecast_publication(self):
        official = official_fixture()
        for item in official['futian']['data']['hoursweather']:
            item['pubTime'] = (NOW - timedelta(hours=19)).strftime('%Y-%m-%d %H:%M')
        self.assertIn('error', sz_official.validate_feed('futian', official['futian'], NOW))

    def test_old_radar_does_not_upgrade_and_blocks_publication(self):
        radar = radar_fixture()
        radar.update(last=(NOW - timedelta(minutes=21)).strftime('%Y-%m-%d %H:%M'),
                     strong={'km': 1, 'direction': '北', 'pixels': 50, 'max_dbz': 50})
        external, _, _, _ = self.render(radar=radar)
        self.assertIn('降雨 低｜雷电 低', external)
        self.assertIn('雷达资料缺测或过期', briefing.publication_checks(official_fixture(), DAY, ISSUE, radar, NOW))

    def test_cloud_lightning_disabled_makes_no_request(self):
        with patch.object(sz_lightning, 'routes', side_effect=forbidden):
            value = sz_lightning.read_lightning(enabled=False)
        self.assertFalse(sz_lightning.available(value))
        self.assertTrue(value['disabled'])
        self.assertIn('闪电定位资料缺测（云端已禁用）', self.render()[0])
        self.assertEqual(briefing.publication_checks(official_fixture(), DAY, ISSUE, radar_fixture(), NOW), [])

    def test_ec_missing_fields_and_window_are_clear_errors(self):
        value = ec_fixture()
        del value['rows'][0]['mucape_point']
        with self.assertRaisesRegex(ValueError, 'missing fields'):
            ec_point.validate_event_coverage(value, DAY)
        value = ec_fixture()
        value['rows'] = value['rows'][:1]
        with self.assertRaisesRegex(ValueError, 'activity-window coverage'):
            ec_point.validate_event_coverage(value, DAY)
        value['rows'][0]['gust_point_ms'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            ec_point.validate_event_coverage(value, DAY)

    def test_late_nowcast_still_downloads_full_event_window(self):
        with patch.object(briefing, 'extract', return_value=ec_fixture()) as extract:
            briefing.load_ec(DAY, DAY + timedelta(hours=21.25), '2026101300')
        self.assertEqual(extract.call_args.kwargs['begin'], DAY + timedelta(hours=19.5))

    def test_immutable_output_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            metadata = {'publication_allowed': False, 'generated_at': NOW.isoformat()}
            stem = briefing.save_briefing(out, DAY, ISSUE, 'first', 'brief', 'basis', metadata)
            with self.assertRaises(FileExistsError):
                briefing.save_briefing(out, DAY, ISSUE, 'second', 'brief', 'basis', metadata)
            self.assertEqual(stem.with_suffix('.txt').read_text(encoding='utf-8-sig'), 'first\n')
            self.assertFalse(json.loads((out / 'briefing_result.json').read_text())['publication_allowed'])

    def test_history_search_recurses(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'sent-version'
            out.mkdir()
            old = out / '灯光秀气象研判_20261013_0800_活动1013.txt'
            old.write_text('sent report', encoding='utf-8-sig')
            self.assertEqual(briefing.previous_briefing(Path(folder), DAY, ISSUE)[1], 'sent report')

    def test_strict_generation_saves_blocked_metadata_and_exits_without_push(self):
        official = official_fixture()
        official['alarm'] = {'error': 'offline'}
        with tempfile.TemporaryDirectory() as folder, patch.object(briefing, 'load_ec', return_value=(ec_fixture(), None)), \
                patch.object(briefing, 'fetch', return_value=official), patch.object(briefing, 'read_radar', return_value=radar_fixture()), \
                patch.object(briefing, 'datetime') as clock, patch.object(sys, 'argv', ['event_briefing.py', '--issue', '2026-10-13 16:00',
                '--mode', 'scheduled', '--no-lightning', '--require-publishable', '--output-dir', folder]), redirect_stdout(io.StringIO()):
            clock.now.return_value = NOW
            clock.strptime.side_effect = datetime.strptime
            with self.assertRaises(SystemExit) as exited:
                briefing.main()
            self.assertEqual(exited.exception.code, 2)
            metadata = json.loads((Path(folder) / 'briefing_result.json').read_text())
            self.assertFalse(metadata['publication_allowed'])
            self.assertTrue(metadata['blocking_checks'])
            self.assertEqual(metadata['generated_at'], NOW.isoformat())
            self.assertEqual(metadata['event_date'], '2026-10-13')

    def test_availability_checks_other_mirrors_and_head_fallback(self):
        import requests
        response = Mock(status_code=200)
        with patch.object(requests, 'head', side_effect=[Mock(status_code=404), Mock(status_code=404), response]):
            self.assertTrue(ec_point.available(datetime(2026, 10, 13), 24))
        streamed = Mock()
        streamed.__enter__ = Mock(return_value=response)
        streamed.__exit__ = Mock(return_value=False)
        with patch.object(requests, 'head', return_value=Mock(status_code=405)), patch.object(requests, 'get', return_value=streamed):
            self.assertTrue(ec_point.available(datetime(2026, 10, 13), 24))

    def test_download_rejects_bad_mirror_and_atomically_changes_source(self):
        tried = []
        def client_factory(source, **kwargs):
            tried.append(source)
            def download(**request):
                Path(request['target']).write_bytes(b'valid' if source == 'azure' else b'broken')
            return types.SimpleNamespace(retrieve=download)
        def validate(path):
            if path.read_bytes() != b'valid':
                raise ValueError('invalid GRIB')
        with tempfile.TemporaryDirectory() as folder, patch.object(ec_point, 'Client', side_effect=client_factory):
            target = Path(folder) / 'guidance.grib2'
            ec_point.retrieve(datetime(2026, 10, 13), [3], target, validator=validate)
            self.assertEqual(target.read_bytes(), b'valid')
            self.assertFalse(target.with_name(target.name + '.part').exists())
        self.assertEqual(tried, ['google', 'aws', 'azure'])

    def test_azure_availability_uses_public_read_token_when_required(self):
        import requests
        token_response = Mock()
        token_response.json.return_value = {'token': 'st=offline-test&sig=fixture'}
        with patch.object(requests, 'head', side_effect=[Mock(status_code=404), Mock(status_code=404),
              Mock(status_code=403), Mock(status_code=200)]) as head, patch.object(requests, 'get', return_value=token_response):
            self.assertTrue(ec_point.available(datetime(2026, 10, 13), 24))
            self.assertIn('?st=offline-test&sig=fixture', head.call_args.args[0])

    def test_covered_nowcasts_only_use_valid_tips(self):
        official = official_fixture()
        official['tips']['data'] = {'important': {'d': '2026-10-13 15:20', 'prompt': '【短临预报】今晚有雷雨。'},
                                   'more': [{'d': '2026-10-13 12:00', 'prompt': '【天气提示】今天多云。'}]}
        self.assertEqual(briefing.covered_nowcasts(official, NOW), ['2026-10-13 15:20'])
        official['tips']['cache_time'] = (NOW - timedelta(hours=37)).isoformat()
        self.assertEqual(briefing.covered_nowcasts(official, NOW), [])

    def test_radar_no_frames_single_frame_and_zero_correlation_are_unknown(self):
        empty = radar_track.estimate([], now=NOW)
        self.assertIn('error', empty)
        dbz = np.zeros((281, 281))
        east, north = np.meshgrid(np.arange(-140, 141), np.arange(140, -141, -1))
        frame = (NOW, dbz, east, north)
        single = radar_track.estimate([frame], now=NOW)
        self.assertIsNone(single['speed'])
        self.assertIn('无法可靠判断', radar_track.text(single, NOW))
        self.assertEqual(radar_track.best_shift(dbz, dbz), (None, None))
        self.assertEqual(radar_track.motion([frame, frame]), (None, None))

    def test_radar_colour_background_and_midnight_rounding(self):
        pixels = np.array([[[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 0, 255], [255, 0, 0, 0]]])
        self.assertEqual(sz_radar.decode_dbz(pixels).tolist(), [[0, 0, 35, 0]])
        year_end = datetime(2026, 12, 31, 23, 58, tzinfo=BJT)
        expected = datetime(2027, 1, 1, tzinfo=BJT)
        self.assertEqual(radar_track.q15(year_end, True), expected)
        self.assertEqual(radar_track.round_half_hour(year_end), expected)


if __name__ == '__main__':
    if IMPORT_SUBSTITUTES:
        print('Offline import substitutes: ' + ', '.join(IMPORT_SUBSTITUTES), file=sys.stderr)
    unittest.main(verbosity=2)
