"""Three-part weather presentation regressions; all network calls are forbidden."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import numpy as np
import requests
import radar_nowcast as product
import radar_track

NOW = datetime(2026, 10, 13, 15, 30, tzinfo=product.BJT)


def forbidden(*args, **kwargs):
    raise AssertionError('Live weather retrieval, EC retrieval and messaging are forbidden')


def frames(latest=NOW, value=40, center=-40):
    east, north = np.meshgrid(np.arange(-150, 151, dtype=float), np.arange(150, -151, -1, dtype=float))
    def field(c):
        return np.where((np.abs(east - c) <= 5) & (np.abs(north) <= 6), value, 0)
    return [(latest - timedelta(minutes=6), field(center - 6), east, north),
            (latest, field(center), east, north)]


def move():
    return {'status': 'ok', 'reason': None, 've': 60.0, 'vn': 0.0, 'correlation': 1.0, 'peak_margin': 0.3}


class RadarTemplateTests(unittest.TestCase):
    def setUp(self):
        for method in ('get', 'head', 'post'):
            guard = patch.object(requests, method, side_effect=forbidden)
            guard.start()
            self.addCleanup(guard.stop)

    def result(self, frame_list=None, now=NOW, config=None):
        with patch.object(radar_track, 'motion', return_value=move()):
            return product.analyse(frames() if frame_list is None else frame_list, now=now, config=config)

    def assert_three_parts(self, result):
        self.assertEqual(result['message_format'], 'notice-detail-brief-v1')
        self.assertEqual(result['messages'], [result['notice'], result['external'], result['brief']])
        self.assertEqual(len(set(result['messages'])), 3)
        self.assertTrue(all(isinstance(text, str) and text.strip() for text in result['messages']))
        forbidden_words = ('dBZ', '像元', '像素', '相关系数', '相关性', '门槛', '阈值', '采样', '35dBZ', '8公里/小时',
                           '低相关', 'ReadTimeout', 'ValueError', '雷电', '雷雨', '阵风', '℃', '预警信号生效', '风险等级', '适宜')
        for text in result['messages']:
            for word in forbidden_words:
                self.assertNotIn(word, text)

    def test_normal_vector_order_style_and_utf8_limits(self):
        result = self.result()
        self.assert_three_parts(result)
        self.assertIn('草稿', result['notice'])
        self.assertIn('后面两条依次为详细版和简版正文', result['notice'])
        self.assertIn('发出前请核对', result['notice'])
        headings = ('[市民中心雷达短临预报]', '[预报时效]', '[当前雷达]', '[分时段预报]', '[降水研判]', '[建议]')
        positions = [result['external'].index(label) for label in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('[建议]\n1. ', result['external'])
        self.assertIn('\n2. 若遇降水', result['external'])
        self.assertEqual(len(result['brief'].splitlines()), 2)
        self.assertIn('15:30—16:00', result['external'])
        self.assertNotIn('0–30分钟', result['external'])
        for text in result['messages']:
            self.assertLessEqual(len(text.encode('utf-8')), 2048)

    def test_known_arrival_strong_and_ending_are_relative_to_actual_issue(self):
        result = self.result(now=NOW + timedelta(minutes=10))
        self.assertEqual((result['arrival_minutes'], result['strong_arrival_minutes'], result['end_minutes']), (25, 25, 40))
        self.assertIn('16:05前后降水回波到达（实际出稿后25分钟左右）', result['external'])
        self.assertIn('16:05前后较强回波开始影响', result['external'])
        self.assertIn('本轮回波影响预计16:20前后结束（实际出稿后40分钟左右）', result['external'])
        self.assertIn('按回波移动推算，出稿时刻', result['external'])
        self.assert_three_parts(result)
        for text in result['messages']:
            self.assertLessEqual(len(text.encode('utf-8')), 2048)

    def test_cross_midnight_clocks_and_three_part_byte_limits(self):
        now = datetime(2026, 12, 31, 23, 55, tzinfo=product.BJT)
        result = self.result(frames(latest=now), now=now)
        self.assertIn('23:55—次日00:25', result['external'])
        self.assertIn('次日00:30前后降水回波到达', result['external'])
        self.assertIn('次日00:45前后结束', result['brief'])
        self.assert_three_parts(result)
        for text in result['messages']:
            self.assertLessEqual(len(text.encode('utf-8')), 2048)

    def test_all_unknown_merges_future_window_without_dry_weather_promise(self):
        result = product.analyse(frames(value=0), now=NOW)
        self.assertFalse(result['forecast_reliable'])
        self.assertTrue(result['observed_clear'])
        self.assert_three_parts(result)
        section = result['external'].split('[分时段预报]\n', 1)[1].split('\n[降水研判]', 1)[0]
        self.assertEqual(len(section.splitlines()), 1)
        self.assertIn('15:30—17:30', section)
        self.assertIn('回波移动趋势尚不明确', result['brief'])
        self.assertNotIn('未来两小时', result['brief'])
        for text in result['messages']:
            for wrong in ('未来两小时无雨', '天气平稳', '降雨可能性较小', '预计15:30前后降水回波到达'):
                self.assertNotIn(wrong, text)
        self.assertIn('不表示未来两小时必然无雨', result['internal'])

    def test_single_rain_frame_reports_observation_and_uncertain_future(self):
        result = product.analyse(frames(center=0)[-1:], now=NOW)
        self.assertTrue(result['current']['rain'])
        self.assertFalse(result['forecast_reliable'])
        self.assert_three_parts(result)
        self.assertIn('有较强回波', result['external'])
        self.assertIn('该时段降水变化', result['brief'])
        self.assertIn('本次暂不提供明确', result['external'])

    def test_weak_echo_and_sparse_echo_do_not_become_no_rain_guarantee(self):
        weak = self.result(frames(value=24))
        self.assert_three_parts(weak)
        self.assertIn('暂无明显降水回波', weak['external'])
        self.assertIn('回波新生', weak['brief'])
        t, dbz, east, north = frames(value=0)[-1]
        dbz[150, 150] = 50
        sparse = product.analyse([(t, dbz, east, north)], now=NOW)
        self.assertEqual(sparse['current']['status'], 'unknown')
        self.assertIn('零散回波', sparse['external'])
        self.assertIn('降水影响仍需观察', sparse['brief'])
        self.assert_three_parts(sparse)

    def test_stale_missing_and_partial_coverage_keep_uncertainty_in_all_three(self):
        old = self.result(now=NOW + timedelta(minutes=21))
        missing = product.analyse([], now=NOW)
        east, north = np.meshgrid(np.arange(-60, 61, dtype=float), np.arange(60, -61, -1, dtype=float))
        dbz = np.zeros_like(east)
        partial = self.result([(NOW - timedelta(minutes=6), dbz, east, north), (NOW, dbz, east, north)])
        for result in (old, missing, partial):
            self.assert_three_parts(result)
            self.assertFalse(result['forecast_reliable'])
            self.assertIn('暂不提供明确', result['external'])
            self.assertIn('尚待核实' if not result['publication_allowed'] else '其余时段尚待核实', result['brief'])
            self.assertIn('起止仍需观察', result['notice'])
        self.assertIn('尚不足以代表当前情况', old['external'])

    def test_place_names_and_long_names_are_preserved_without_truncation(self):
        config = product.Config(name='龙岗中心')
        result = self.result(config=config)
        self.assert_three_parts(result)
        for text in result['messages']:
            self.assertIn('龙岗中心', text)
            self.assertNotIn('市民中心', text)
        long_name = '很长的地区名称' * 150
        result = self.result(config=product.Config(name=long_name))
        for text in result['messages']:
            self.assertIn(long_name, text)
        self.assertGreater(len(result['external'].encode('utf-8')), 2048)

    def test_partial_known_wet_signal_is_retained_in_all_three_without_reliable_times(self):
        for kind in ('rain', 'strong'):
            with self.subTest(kind=kind):
                result = self.result()
                result.update(forecast_reliable=False, arrival_minutes=None, strong_arrival_minutes=None, end_minutes=None, clear=False)
                for item in result['steps']:
                    if item['minute'] == 0:
                        item.update(status='clear', rain=False, strong=False)
                    elif item['minute'] == 5:
                        item.update(status=kind, rain=True, strong=kind == 'strong')
                    else:
                        item.update(status='unknown', rain=None, strong=None, coverage=False)
                product.refresh_presentation(result)
                signal = '较强回波影响' if kind == 'strong' else '降水回波影响'
                for text in result['messages']:
                    self.assertIn(signal, text)
                    self.assertIn('尚待核实', text)
                    self.assertNotIn('16:05前后降水回波到达', text)
                    self.assertNotIn('16:20前后结束', text)
                self.assertIn('暂不提供明确', result['external'])
                self.assertIsNone(result['arrival_minutes'])
                self.assertIsNone(result['end_minutes'])
                self.assert_three_parts(result)

    def test_refresh_does_not_change_algorithm_and_never_copies_raw_checks(self):
        result = self.result()
        algorithm = {key: copy.deepcopy(value) for key, value in result.items() if key not in ('notice', 'external', 'brief', 'internal', 'messages', 'message_format')}
        result['checks'].append('ReadTimeout raw 35dBZ 像元3 相关系数0.3')
        original_checks = list(result['checks'])
        product.refresh_presentation(result)
        self.assertEqual(result['checks'], original_checks)
        for key, value in algorithm.items():
            if key != 'checks':
                self.assertEqual(result[key], value)
        self.assertIn('ReadTimeout raw 35dBZ', result['internal'])
        self.assert_three_parts(result)

    def test_main_fetch_failure_refreshes_vector_and_saves_all_outputs_as_unpublishable(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(product, 'load_frames', side_effect=requests.RequestException('raw url unavailable')), redirect_stdout(io.StringIO()):
            code = product.main(['--output-dir', str(Path(folder) / 'version'), '--now', NOW.isoformat()])
            self.assertEqual(code, 2)
            output = Path(folder) / 'version'
            result = json.loads((output / 'radar_result.json').read_text())
            self.assertFalse(result['publication_allowed'])
            self.assert_three_parts(result)
            for filename, key in (('雷达短临预报.txt', 'external'), ('雷达短临预报_提醒.txt', 'notice'),
                                  ('雷达短临预报_简版.txt', 'brief'), ('雷达短临预报_内部依据.txt', 'internal')):
                self.assertEqual((output / filename).read_text(), result[key] + '\n')

    def test_output_failure_rolls_back_all_created_files_and_immutable_versions_remain(self):
        result = self.result()
        real_open = Path.open
        def fail_brief(path, *args, **kwargs):
            if path.name == '雷达短临预报_简版.txt':
                raise OSError('simulated full disk')
            return real_open(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'version'
            with patch.object(Path, 'open', new=fail_brief), self.assertRaises(OSError):
                product.save_result(result, output)
            self.assertEqual(list(output.iterdir()), [])
            product.save_result(result, output)
            with self.assertRaises(FileExistsError):
                product.save_result(result, output)
            self.assertEqual(len(list(output.iterdir())), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
