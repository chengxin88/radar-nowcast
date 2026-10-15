from datetime import datetime
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import yjt_observation as yjt


STATION = {'name': '市民中心', 'subdistrict': '莲花街道', 'areaid': '5949302'}
NOW = datetime(2026, 10, 15, 10, 40, tzinfo=yjt.BJT)


def row(kind, **updates):
    value = dict(STATION, updateTime='2026-10-15 10:34:00')
    value.update({'r05m': '0.0', 'r10m': '0.0', 'r30m': '0.0', 'r01h': '0.0'}
                 if kind == 'rain' else {'w01h': '7.2'})
    value.update(updates)
    return value


def raw(rain=None, wind=None):
    return {kind: {'source': yjt.SOURCES[kind], 'fetched_at': NOW.isoformat(),
                   'data': value if value is not None else row(kind)}
            for kind, value in (('rain', rain), ('wind', wind))}


def response(data):
    result = Mock(status_code=200, content=json.dumps(data).encode())
    result.raise_for_status.return_value = None
    return result


class FetchTests(unittest.TestCase):
    def test_only_real_endpoints_and_exact_unique_station(self):
        def get(url, **kwargs):
            kind = 'rain' if url == yjt.SOURCES['rain'] else 'wind'
            return response([row(kind, name='深圳福田国家基本气象站'), row(kind)])
        with patch.object(yjt.requests, 'get', side_effect=get) as get_mock:
            result = yjt.fetch(STATION)
        self.assertEqual({call.args[0] for call in get_mock.call_args_list}, set(yjt.SOURCES.values()))
        for call in get_mock.call_args_list:
            self.assertIsInstance(call.kwargs['params']['time'], int)
            self.assertEqual(call.kwargs['headers']['User-Agent'], 'Mozilla/5.0')
            self.assertFalse(call.kwargs['allow_redirects'])
        self.assertEqual(result['rain']['data']['name'], '市民中心')
        self.assertEqual(result['wind']['data']['w01h'], '7.2')
        self.assertIn('fetched_at', result['rain'])

    def test_one_http_failure_does_not_erase_other_product(self):
        def get(url, **kwargs):
            if url == yjt.SOURCES['rain']:
                raise requests.HTTPError('403')
            return response([row('wind')])
        with patch.object(yjt.requests, 'get', side_effect=get):
            result = yjt.fetch(STATION)
        self.assertIn('error', result['rain'])
        self.assertNotIn('data', result['rain'])
        self.assertEqual(result['wind']['data']['w01h'], '7.2')

    def test_duplicate_missing_wrong_street_and_malformed_rows_are_rejected(self):
        cases = [[], [row('rain'), row('rain')], [row('rain', subdistrict='香蜜湖街道')],
                 [row('rain', areaid='5949301')], {'data': [row('rain')]}]
        for payload in cases:
            with self.subTest(payload=payload), patch.object(yjt.requests, 'get', return_value=response(payload)):
                result = yjt.fetch(STATION)
            self.assertIn('error', result['rain'])

    def test_bad_json_and_invalid_station_fail_closed(self):
        bad = Mock(status_code=200, content=b'<html>not json</html>')
        bad.raise_for_status.return_value = None
        with patch.object(yjt.requests, 'get', return_value=bad):
            self.assertIn('error', yjt.fetch(STATION)['rain'])
        with patch.object(yjt.requests, 'get') as get_mock:
            self.assertIn('error', yjt.fetch({'name': '市民中心'})['rain'])
            get_mock.assert_not_called()


class SummaryTests(unittest.TestCase):
    def test_zero_remains_zero_and_wind_is_one_hour_maximum(self):
        result = yjt.summarize(raw(), STATION, NOW)
        self.assertEqual(result['rain']['rain_5m'], 0.0)
        self.assertEqual(result['rain']['rain_1h'], 0.0)
        self.assertEqual(result['wind']['max_wind_1h_m_s'], 7.2)
        self.assertEqual(result['rain']['time'], '2026-10-15T10:34:00+08:00')
        self.assertFalse(result['checks'])

    def test_independent_times_reject_stale_future_and_missing_timestamp(self):
        for stamp in ('2026-10-15 10:19:59', '2026-10-15 10:42:01', '', None, '2026-10-15'):
            with self.subTest(stamp=stamp):
                result = yjt.summarize(raw(rain=row('rain', updateTime=stamp)), STATION, NOW)
                self.assertIsNone(result['rain'])
                self.assertIsNotNone(result['wind'])
                self.assertTrue(result['checks'])

    def test_age_boundaries_and_timezone_conversion(self):
        for stamp in ('2026-10-15 10:20:00', '2026-10-15 10:42:00', '2026-10-15T02:34:00Z'):
            with self.subTest(stamp=stamp):
                result = yjt.summarize(raw(rain=row('rain', updateTime=stamp)), STATION, NOW)
                self.assertIsNotNone(result['rain'])

    def test_missing_values_are_not_coerced_to_zero(self):
        for value in ('', '--', '-9999', '9999', '999.9', 'nan', 'inf', '-1', None, True):
            with self.subTest(value=value):
                result = yjt.summarize(raw(rain=row('rain', r01h=value), wind=row('wind', w01h=value)), STATION, NOW)
                self.assertIsNone(result['rain']['rain_1h'])
                self.assertIsNone(result['wind'])
                self.assertEqual(result['rain']['rain_5m'], 0.0)
                self.assertTrue(result['checks'])

    def test_all_rain_missing_is_not_dry(self):
        result = yjt.summarize(raw(rain=row('rain', r05m='', r10m='', r30m='', r01h='')), STATION, NOW)
        self.assertIsNone(result['rain'])
        self.assertIsNotNone(result['wind'])

    def test_contradictory_accumulations_are_rejected_even_across_missing_period(self):
        cases = [dict(r05m='2', r10m='1', r30m='3', r01h='4'),
                 dict(r05m='2', r10m='', r30m='1', r01h='4')]
        for values in cases:
            with self.subTest(values=values):
                result = yjt.summarize(raw(rain=row('rain', **values)), STATION, NOW)
                self.assertIsNone(result['rain'])
                self.assertIsNotNone(result['wind'])

    def test_station_and_source_rechecked_at_summary(self):
        for key, value in (('name', '另一个站'), ('subdistrict', '香蜜湖街道'), ('areaid', '5949301')):
            with self.subTest(key=key):
                result = yjt.summarize(raw(rain=row('rain', **{key: value})), STATION, NOW)
                self.assertIsNone(result['rain'])
        data = raw()
        data['rain']['source'] = 'https://example.com/data.json'
        self.assertIsNone(yjt.summarize(data, STATION, NOW)['rain'])

    def test_errors_missing_raw_and_invalid_now_fail_closed(self):
        data = raw()
        data['rain']['error'] = 'fetch failed'
        self.assertIsNone(yjt.summarize(data, STATION, NOW)['rain'])
        self.assertIsNone(yjt.summarize(None, STATION, NOW)['wind'])
        self.assertIsNone(yjt.summarize(raw(), STATION, 'invalid')['rain'])


if __name__ == '__main__':
    unittest.main()
