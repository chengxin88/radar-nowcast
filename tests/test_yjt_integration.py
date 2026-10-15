"""Station observations remain distinct from regional reference observations."""
import copy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import rolling_official as official

NOW = datetime(2026, 10, 15, 10, 40, tzinfo=official.BJT)


def feeds():
    station = {key: value for key, value in official.DEFAULT_PROFILE['yjt_station'].items()
               if key in ('name', 'subdistrict', 'areaid')}
    station['updateTime'] = '2026-10-15 10:34:00'
    rain = dict(station, r05m='0.0', r10m='0.0', r30m='0.0', r01h='0.0')
    wind = dict(station, w01h='7.2')
    return {
        'observation': {'cache_time': NOW.isoformat(), 'data': {
            'date': NOW.isoformat(), 't': 27.9, 'u': 79, 'v': 12.2, 'h01r': 99}},
        'yjt_observation': {
            'rain': {'data': rain, 'source': 'https://yjt.121.com.cn/Data/ForecstReal/JsonRain.json'},
            'wind': {'data': wind, 'source': 'https://yjt.121.com.cn/Data/ForecstReal/JsonWind.json'}},
    }


class StationIntegrationTests(unittest.TestCase):
    def text(self, raw):
        context = official.summarize(raw, official.DEFAULT_PROFILE, NOW)
        return official.observation_text(context), context

    def test_rainfall_and_maximum_wind_are_named_and_reference_site_is_separate(self):
        text, context = self.text(feeds())
        self.assertIn('市民中心站过去5分钟0毫米', text)
        self.assertIn('1小时0毫米', text)
        self.assertIn('过去1小时极大风速7.2米/秒', text)
        self.assertNotIn('平均风速', text)
        self.assertNotIn('另据区域参考实况', text)
        self.assertIn('（国家基本站）', text)
        self.assertIn('气温27.9℃', text)
        self.assertNotIn('99', text)
        self.assertEqual(context['station_observation']['rain']['rain_1h'], 0)
        self.assertIn('截至10:40，气温', text)  # Six-minute difference stays visible.
        for minutes in (0, 2, 5):
            with self.subTest(reference_minutes_later=minutes):
                raw = feeds()
                raw['observation']['data']['date'] = (NOW - timedelta(minutes=6-minutes)).isoformat()
                compact, _ = self.text(raw)
                self.assertEqual(compact.count('截至'), 1)
                self.assertIn('截至10:34，市民中心站', compact)

    def test_station_failure_does_not_silently_use_other_sites_rain(self):
        raw = feeds()
        raw['yjt_observation'] = {'rain': {'error': 'HTTPError'}, 'wind': {'error': 'HTTPError'}}
        text, context = self.text(raw)
        self.assertIn('市民中心站降水实况暂缺', text)
        self.assertIn('市民中心站风速实况暂缺', text)
        self.assertNotIn('记录到降水', text)
        self.assertNotIn('毫米', text)
        self.assertTrue(context['station_observation']['checks'])

    def test_fresh_wind_survives_stale_rain_and_has_its_own_time(self):
        raw = feeds()
        raw['yjt_observation']['rain']['data']['updateTime'] = '2026-10-15 09:00:00'
        text, context = self.text(raw)
        self.assertIn('降水实况暂缺', text)
        self.assertIn('截至10:34，过去1小时极大风速7.2米/秒', text)
        self.assertIsNone(context['station_observation']['rain'])

    def test_station_observations_work_when_reference_temperature_is_missing(self):
        raw = feeds()
        del raw['observation']
        text, _ = self.text(raw)
        self.assertIn('极大风速7.2米/秒', text)
        self.assertNotIn('气温', text)
        self.assertNotIn('区域参考实况', text)

    def test_fetch_includes_the_new_observation_source(self):
        with patch.object(official, 'read_feed', return_value={}) as old, patch.object(
                official.yjt_observation, 'fetch', return_value={'marker': True}) as new:
            raw = official.fetch(official.DEFAULT_PROFILE)
        self.assertEqual(old.call_count, 4)
        self.assertEqual(new.call_count, 1)
        self.assertEqual(raw['yjt_observation'], {'marker': True})

    def test_changing_target_does_not_reuse_the_old_station_mapping(self):
        profile = copy.deepcopy(official.DEFAULT_PROFILE)
        profile['target']['lat'] = 22.64
        with patch.object(official, 'read_feed', return_value={}), patch.object(
                official.yjt_observation, 'fetch') as new:
            raw = official.fetch(profile)
        new.assert_not_called()
        self.assertNotIn('yjt_observation', raw)
        context = official.summarize(feeds(), profile, NOW)
        self.assertNotIn('station_observation', context)
        self.assertTrue(any('未套用其他地区站点' in item for item in context['checks']))

    def test_profiles_without_station_mapping_keep_existing_reference_behavior(self):
        profile = copy.deepcopy(official.DEFAULT_PROFILE)
        del profile['yjt_station']
        context = official.summarize(feeds(), profile, NOW)
        text = official.observation_text(context)
        self.assertIn('该站过去1小时记录到降水', text)
        self.assertNotIn('市民中心站', text)


if __name__ == '__main__':
    unittest.main()
