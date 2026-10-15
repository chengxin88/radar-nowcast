"""Local fixtures for configurable target geometry, names and real-image crops."""
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import numpy as np
import requests
from PIL import Image
import radar_nowcast as product
import sz_radar

NOW = datetime(2026, 10, 13, 15, 30, tzinfo=product.BJT)
PATH = 'radar/cappi/202610131530.png'


def forbidden(*args, **kwargs):
    raise AssertionError('No live network or messaging is permitted')


def png_bytes(width=900, height=700, colour=(255, 0, 0, 255), target=None):
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    pixels[:] = colour
    if target:
        pixels[:] = (0, 0, 0, 0)
        x = int((target.lon - sz_radar.WEST) / (sz_radar.EAST - sz_radar.WEST) * width)
        y = int((sz_radar.NORTH - target.lat) / (sz_radar.NORTH - sz_radar.SOUTH) * height)
        pixels[max(0, y - 5):min(height, y + 6), max(0, x - 5):min(width, x + 6)] = (255, 255, 0, 255)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    return output.getvalue()


def fake_frame(config=None):
    config = config or product.Config()
    cx, cy = product.target_center(config)
    east, north = np.meshgrid(np.arange(-160, 161, dtype=float) + cx, np.arange(160, -161, -1, dtype=float) + cy)
    return np.zeros_like(east), east, north


class RadarRegionTests(unittest.TestCase):
    def setUp(self):
        for method in ('get', 'head', 'post'):
            guard = patch.object(requests, method, side_effect=forbidden)
            guard.start()
            self.addCleanup(guard.stop)

    def read_png(self, config, content=None, width=900, height=700):
        response = Mock(content=content if content is not None else png_bytes(width, height))
        with patch.object(requests, 'get', return_value=response):
            return sz_radar.frame(PATH, lat=config.lat, lon=config.lon, radius_km=config.radius_km)

    def test_default_name_and_trimmed_single_line_name(self):
        self.assertEqual(product.Config().name, '市民中心')
        config = product.Config(name='  龙岗中心  ')
        config.validate()
        self.assertEqual(config.name, '龙岗中心')
        frame = fake_frame(config)
        result = product.analyse([(NOW, *frame)], now=NOW, config=config)
        self.assertEqual(result['target']['name'], config.name)
        self.assertIn('[龙岗中心雷达短临预报]', result['external'])
        self.assertIn('龙岗中心周边', result['external'])
        self.assertIn('最新观测的龙岗中心100公里附近', result['internal'])
        self.assertNotIn('市民中心', result['external'])

    def test_name_coordinate_and_radius_invalid_values_rejected(self):
        invalid = [dict(name=''), dict(name='   '), dict(name='地区\n伪标题'), dict(name='地区\r'),
                   dict(name='地区\t'), dict(name='地区\u2028另一行'), dict(name='地区\x80'), dict(name='地区\x9f'), dict(name=None),
                   dict(lat=float('nan')), dict(lon=float('inf')), dict(lat=91), dict(lon=-181),
                   dict(radius_km=0), dict(radius_km=-1), dict(radius_km=500.01)]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                product.Config(**arguments).validate()
        product.Config(radius_km=500).validate()

    def test_target_position_changes_actual_echo_sampling_and_frame_center(self):
        target = product.Config(name='惠州目标', lat=23.5, lon=115.5)
        content = png_bytes(target=target)
        remote = self.read_png(target, content)
        default = self.read_png(product.Config(), content)
        remote_result = product.analyse([(NOW, *remote)], now=NOW, config=target)
        default_result = product.analyse([(NOW, *default)], now=NOW)
        self.assertTrue(remote_result['current']['strong'])
        self.assertFalse(default_result['current']['rain'])
        expected = product.target_center(target)
        for value, wanted in zip(remote[1:], expected):
            row, col = value.shape[0] // 2, value.shape[1] // 2
            spacing = abs(value[0, 1] - value[0, 0]) if wanted == expected[0] else abs(value[1, 0] - value[0, 0])
            self.assertLessEqual(abs(value[row, col] - wanted), spacing / 2 + 1e-8)
        self.assertGreater(abs(remote[1][remote[1].shape[0] // 2, remote[1].shape[1] // 2]), 100)
        self.assertLess(abs(default[1][default[1].shape[0] // 2, default[1].shape[1] // 2]), 1)

    def test_true_pixels_at_four_edges_and_corners_without_padding(self):
        width, height = 900, 700
        dlon, dlat = (sz_radar.EAST - sz_radar.WEST) / width, (sz_radar.NORTH - sz_radar.SOUTH) / height
        middle_lon, middle_lat = (sz_radar.WEST + sz_radar.EAST) / 2, (sz_radar.SOUTH + sz_radar.NORTH) / 2
        lons = (sz_radar.WEST + 3.25 * dlon, sz_radar.EAST - 3.25 * dlon)
        lats = (sz_radar.SOUTH + 3.25 * dlat, sz_radar.NORTH - 3.25 * dlat)
        positions = [(lon, middle_lat) for lon in lons] + [(middle_lon, lat) for lat in lats] + [(lon, lat) for lon in lons for lat in lats]
        content = png_bytes(width, height)
        x0 = (sz_radar.SITE_LON - sz_radar.WEST) / (sz_radar.EAST - sz_radar.WEST) * width
        y0 = (sz_radar.NORTH - sz_radar.SITE_LAT) / (sz_radar.NORTH - sz_radar.SOUTH) * height
        kx = dlon * 111.32 * math.cos(math.radians(sz_radar.SITE_LAT))
        ky = dlat * 110.57
        for lon, lat in positions:
            with self.subTest(lon=lon, lat=lat):
                dbz, east, north = self.read_png(product.Config(name='边缘目标', lat=lat, lon=lon), content)
                self.assertTrue(np.all(dbz == 50), 'Opaque red original pixels must never acquire transparent/zero padding')
                xs, ys = east / kx + x0, y0 - north / ky
                self.assertGreaterEqual(xs.min(), 0.5 - 1e-8)
                self.assertLessEqual(xs.max(), width - 0.5 + 1e-8)
                self.assertGreaterEqual(ys.min(), 0.5 - 1e-8)
                self.assertLessEqual(ys.max(), height - 0.5 + 1e-8)
                self.assertEqual(dbz.shape[0] % 2, 1)
                self.assertEqual(dbz.shape[1] % 2, 1)

    def test_mosaic_external_targets_rejected_before_http_without_site_fallback(self):
        for lat, lon in ((sz_radar.NORTH + 0.001, 114), (sz_radar.SOUTH - 0.001, 114),
                         (23, sz_radar.WEST - 0.001), (23, sz_radar.EAST + 0.001)):
            with self.subTest(lat=lat, lon=lon), self.assertRaisesRegex(ValueError, '超出当前雷达资料覆盖'):
                sz_radar.frame(PATH, lat=lat, lon=lon, radius_km=3)
        config = product.Config(name='外部目标', lat=30, lon=120)
        result = product.analyse([(NOW, *fake_frame(config))], now=NOW, config=config)
        self.assertFalse(result['publication_allowed'])
        self.assertEqual(result['target']['name'], '外部目标')
        self.assertTrue(any('超出当前雷达资料覆盖' in item for item in result['blocking_checks']))

    def test_exact_boundary_and_nearest_pixel_cannot_produce_false_current_coverage(self):
        for lon in (sz_radar.WEST, sz_radar.WEST + 0.001, sz_radar.EAST):
            with self.subTest(lon=lon), self.assertRaisesRegex(ValueError, '真实像元不足'):
                self.read_png(product.Config(name='边界目标', lat=23, lon=lon))

    def test_large_radius_is_not_silently_shrunk_to_available_crop(self):
        config = product.Config(name='大范围', radius_km=500)
        result = product.analyse([(NOW, *self.read_png(config))], now=NOW, config=config)
        self.assertEqual(result['target']['radius_km'], 500)
        self.assertFalse(result['publication_allowed'])
        self.assertFalse(result['current']['coverage'])
        self.assertFalse(result['observed_clear'])

    def test_odd_even_png_and_non_square_pixel_scales_keep_target_center(self):
        target = product.Config(name='异地', lat=23.511, lon=115.501)
        for width, height in ((900, 700), (899, 703), (800, 400)):
            with self.subTest(width=width, height=height):
                dbz, east, north = self.read_png(target, width=width, height=height)
                wanted_x, wanted_y = product.target_center(target)
                cy, cx = dbz.shape[0] // 2, dbz.shape[1] // 2
                self.assertLessEqual(abs(east[cy, cx] - wanted_x), abs(east[0, 1] - east[0, 0]) / 2 + 1e-8)
                self.assertLessEqual(abs(north[cy, cx] - wanted_y), abs(north[1, 0] - north[0, 0]) / 2 + 1e-8)

    def test_legacy_path_only_frame_crop_is_compatible(self):
        width, height = 900, 700
        response = Mock(content=png_bytes(width, height))
        with patch.object(requests, 'get', return_value=response):
            dbz, east, north = sz_radar.frame(PATH)
        kx = (sz_radar.EAST - sz_radar.WEST) / width * 111.32 * math.cos(math.radians(sz_radar.SITE_LAT))
        ky = (sz_radar.NORTH - sz_radar.SOUTH) / height * 110.57
        half = int(sz_radar.RADIUS_KM * 1.5 / min(kx, ky))
        self.assertEqual(dbz.shape, (2 * half + 1, 2 * half + 1))
        self.assertEqual(east.shape, dbz.shape)
        self.assertEqual(north.shape, dbz.shape)

    def test_region_radius_name_path_and_projection_versions_isolate_cache(self):
        base = product.Config()
        variants = [product.Config(name='其他名称'), product.Config(lat=22.6), product.Config(lon=114.1), product.Config(radius_km=4)]
        baseline = product.frame_cache_key(PATH, base)
        self.assertTrue(all(product.frame_cache_key(PATH, variant) != baseline for variant in variants))
        self.assertNotEqual(product.frame_cache_key(PATH + '?v=2', base), baseline)
        self.assertNotEqual(product.frame_cache_key(PATH), baseline)
        with patch.object(sz_radar, 'FRAME_PROJECTION_VERSION', 'future-projection'):
            self.assertNotEqual(product.frame_cache_key(PATH, base), baseline)
        with patch.object(sz_radar, 'FRAME_DECODE_VERSION', 'future-decode'):
            self.assertNotEqual(product.frame_cache_key(PATH, base), baseline)
        self.assertNotEqual(baseline, hashlib.sha256(PATH.encode()).hexdigest())

    def test_cache_reads_selected_geometry_and_does_not_reuse_original_site_archive(self):
        target = product.Config(name='异地', lat=23.5, lon=115.5)
        calls = []
        def frame_reader(path, lat=None, lon=None, radius_km=None):
            calls.append((lat, lon, radius_km))
            return fake_frame(product.Config(lat=lat, lon=lon, radius_km=radius_km))
        with tempfile.TemporaryDirectory() as folder, patch.object(sz_radar, 'frame', side_effect=frame_reader):
            old = Path(folder) / (hashlib.sha256(PATH.encode()).hexdigest() + '.npz')
            dbz, east, north = fake_frame()
            np.savez_compressed(old, dbz=dbz, east=east, north=north)
            product.cached_frame(PATH, folder, config=target)
            product.cached_frame(PATH, folder, config=target)
            product.cached_frame(PATH, folder, config=product.Config())
            self.assertEqual(calls, [(23.5, 115.5, 3.0), (22.543, 114.059, 3.0)])
            self.assertEqual(len(list(Path(folder).glob('*.npz'))), 3)

    def test_loader_and_cli_pass_target_config_without_wrong_place_report(self):
        target = product.Config(name='异地', lat=23.5, lon=115.5, radius_km=4)
        response = Mock(text='"radar/cappi/202610131530.png"')
        with patch.object(requests, 'get', return_value=response), patch.object(product, 'cached_frame', return_value=fake_frame(target)) as reader:
            product.load_frames(now=NOW, config=target)
            self.assertEqual(reader.call_args.kwargs['config'], target)
        with tempfile.TemporaryDirectory() as folder, patch.object(product, 'load_frames', return_value=[(NOW, *fake_frame(target))]) as loader, redirect_stdout(io.StringIO()):
            code = product.main(['--output-dir', str(Path(folder) / 'version'), '--now', NOW.isoformat(),
                                 '--target-name', ' 异地 ', '--lat', '23.5', '--lon', '115.5', '--radius-km', '4'])
            self.assertEqual(code, 0)
            self.assertEqual(loader.call_args.kwargs['config'], target)
            result = json.loads((Path(folder) / 'version' / 'radar_result.json').read_text())
            self.assertEqual(result['target'], {'name': '异地', 'lat': 23.5, 'lon': 115.5, 'radius_km': 4.0})
            self.assertNotIn('市民中心', result['external'])

    def test_outside_target_cli_records_explicit_blocker_and_retains_target(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(product, 'load_frames', side_effect=ValueError('目标经纬度超出当前雷达资料覆盖（雷达拼图范围）')), redirect_stdout(io.StringIO()):
            code = product.main(['--output-dir', str(Path(folder) / 'version'), '--now', NOW.isoformat(),
                                 '--target-name', '外部地区', '--lat', '30', '--lon', '120'])
            self.assertEqual(code, 2)
            result = json.loads((Path(folder) / 'version' / 'radar_result.json').read_text())
            self.assertEqual(result['target']['name'], '外部地区')
            self.assertEqual(result['target']['lat'], 30)
            self.assertTrue(any('超出当前雷达资料覆盖' in item for item in result['blocking_checks']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
