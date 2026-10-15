"""Read the Shenzhen radar mosaic around the Futian Civic Center."""
import io
import math
import re
from datetime import datetime

import numpy as np
import requests
from PIL import Image

from ec_event_point import BJT, DIRECTIONS, SITE_LAT, SITE_LON

BASE = 'https://weather.121.com.cn/data_cache/'
HEADERS = {'Referer': 'https://weather.sz.gov.cn/'}
WEST, SOUTH, EAST, NORTH = 108.505, 19.0419, 117.505, 26.0419
PALETTE = (((0, 236, 236), 5), ((1, 160, 246), 10), ((1, 0, 246), 15), ((0, 239, 0), 20), ((0, 200, 0), 25),
           ((0, 144, 0), 30), ((255, 255, 0), 35), ((231, 192, 0), 40), ((255, 144, 0), 45), ((255, 0, 0), 50),
           ((214, 0, 0), 55), ((192, 0, 0), 60), ((255, 0, 240), 65), ((150, 0, 180), 70), ((173, 144, 240), 75))
RADIUS_KM = 100
STRONG, MODERATE = 35, 20
MAX_FRAME_AGE_MINUTES = 20
MAX_FUTURE_MINUTES = 5
MAX_COLOUR_DISTANCE = 30
FRAME_PROJECTION_VERSION = 'fixed-site-km-target-symmetric-real-pixels-v2'
FRAME_DECODE_VERSION = 'rgba-palette-distance-v1'


def as_bjt(when):
    """Naive datetimes in these interfaces are interpreted as Beijing time."""
    return when.replace(tzinfo=BJT) if when.tzinfo is None else when.astimezone(BJT)


def freshness(when, now=None, max_age_minutes=MAX_FRAME_AGE_MINUTES):
    """Validate against wall-clock time, independently of the requested issue time."""
    if max_age_minutes < 0:
        raise ValueError('max_age_minutes must be nonnegative')
    now = as_bjt(now or datetime.now(BJT))
    age = (now - as_bjt(when)).total_seconds() / 60
    if age < -MAX_FUTURE_MINUTES:
        reason = f'雷达末帧晚于当前时间 {-age:.0f} 分钟，时次异常'
    elif age > max_age_minutes:
        reason = f'雷达末帧已过期（距当前 {age:.0f} 分钟，允许 {max_age_minutes:g} 分钟）'
    else:
        reason = None
    return {'fresh': reason is None, 'age_minutes': age, 'freshness_reason': reason}


def decode_dbz(pixels, max_colour_distance=MAX_COLOUR_DISTANCE):
    """Reject opaque backgrounds/labels that do not resemble a radar palette colour."""
    pixels = np.asarray(pixels).astype(np.int32)
    colours = np.array([colour for colour, _ in PALETTE])
    squared = ((pixels[..., None, :3] - colours) ** 2).sum(axis=-1)
    closest = squared.argmin(axis=-1)
    valid = (pixels[..., 3] == 255) & (squared.min(axis=-1) <= max_colour_distance ** 2)
    return np.where(valid, np.array([value for _, value in PALETTE])[closest], 0)


def match_shift(prev, last, maxs=25, half=110, min_pixels=30,
                min_correlation=0.5, min_margin=0.05):
    """Conservative binary echo correlation; an unavailable shift is represented by None.

    The shift maps last[y, x] to prev[y + dy, x + dx]. A distinct correlation
    peak is required, so empty frames and broad/ambiguous fields cannot create
    a default displacement at the edge of the search window.
    """
    prev, last = np.asarray(prev), np.asarray(last)
    result = {'dx': None, 'dy': None, 'status': 'unavailable', 'reason': None,
              'correlation': None, 'peak_margin': None}
    if prev.ndim != 2 or prev.shape != last.shape or min(prev.shape) < 5:
        result['reason'] = '雷达帧网格不一致或过小'
        return result
    height, width = last.shape
    cy, cx = height // 2, width // 2
    maxs = min(int(maxs), (height - 3) // 2, (width - 3) // 2)
    half = min(int(half), cy - maxs, height - cy - maxs,
               cx - maxs, width - cx - maxs)
    if maxs < 1 or half < 1:
        result['reason'] = '雷达帧不足以进行位移匹配'
        return result
    a, b = last >= MODERATE, prev >= MODERATE
    core = a[cy - half:cy + half, cx - half:cx + half]
    count_a = int(core.sum())
    if count_a < min_pixels:
        result['reason'] = '有效降水回波不足，无法估计移动'
        return result
    scores = []
    for dy in range(-maxs, maxs + 1):
        for dx in range(-maxs, maxs + 1):
            patch = b[cy - half + dy:cy + half + dy, cx - half + dx:cx + half + dx]
            count_b = int(patch.sum())
            if count_b < min_pixels:
                continue
            overlap = int((core & patch).sum())
            score = overlap / math.sqrt(count_a * count_b)
            scores.append((score, dx, dy))
    if not scores:
        result['reason'] = '前帧有效降水回波不足，无法估计移动'
        return result
    score, dx, dy = max(scores, key=lambda item: (item[0], -item[1] ** 2 - item[2] ** 2))
    tied = sum(math.isclose(item[0], score, rel_tol=0, abs_tol=1e-12) for item in scores) > 1
    other = max((item[0] for item in scores if abs(item[1] - dx) > 2 or abs(item[2] - dy) > 2),
                default=0.0)
    result.update(correlation=score, peak_margin=score - other)
    if score <= 0 or score < min_correlation:
        result['reason'] = '回波互相关不足，无法可靠估计移动'
    elif tied or score - other < min_margin:
        result['reason'] = '回波匹配峰不明确，无法可靠估计移动'
    elif abs(dx) == maxs or abs(dy) == maxs:
        result['reason'] = '回波位移达到搜索边界，无法可靠估计移动'
    else:
        result.update(dx=dx, dy=dy, status='ok')
    return result


def frame(path, lat=None, lon=None, radius_km=None):
    """Crop real pixels around the selected target; coordinate origin stays SITE.

    With no location arguments the legacy 150 km pixel crop is retained. At
    image edges crop symmetrically within the original image instead of PIL
    padding; a smaller actual extent remains visible to coverage validators.
    """
    legacy = lat is None and lon is None and radius_km is None
    lat = SITE_LAT if lat is None else lat
    lon = SITE_LON if lon is None else lon
    values = [lat, lon] + ([radius_km] if radius_km is not None else [])
    if not all(isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)) for value in values):
        raise ValueError('目标坐标或半径必须为有限数值')
    if not SOUTH <= lat <= NORTH or not WEST <= lon <= EAST:
        raise ValueError('目标经纬度超出当前雷达资料覆盖（雷达拼图范围）')
    if radius_km is not None and not 0 < radius_km <= 500:
        raise ValueError('目标半径必须大于0且不超过500公里')
    response = requests.get(BASE + path, headers=HEADERS, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert('RGBA')
    width, height = image.size
    x0 = (SITE_LON - WEST) / (EAST - WEST) * width
    y0 = (NORTH - SITE_LAT) / (NORTH - SOUTH) * height
    kx = (EAST - WEST) / width * 111.32 * math.cos(math.radians(SITE_LAT))
    ky = (NORTH - SOUTH) / height * 110.57
    target_x = (lon - WEST) / (EAST - WEST) * width
    target_y = (NORTH - lat) / (NORTH - SOUTH) * height
    center_x, center_y = min(width - 1, int(target_x)), min(height - 1, int(target_y))
    if legacy:
        requested_x = requested_y = int(RADIUS_KM * 1.5 / min(kx, ky))
    else:
        half_km = max(150, (radius_km or 0) + 150)
        requested_x, requested_y = int(half_km / kx), int(half_km / ky)
    half_x = min(requested_x, center_x, width - center_x - 1)
    half_y = min(requested_y, center_y, height - center_y - 1)
    if half_x < 1 or half_y < 1:
        raise ValueError('目标临近雷达拼图边界，真实像元不足，不能补边冒充有效覆盖')
    left, top = center_x - half_x, center_y - half_y
    right, bottom = center_x + half_x + 1, center_y + half_y + 1
    pixels = np.asarray(image.crop((left, top, right, bottom)))
    dbz = decode_dbz(pixels)
    east, north = np.meshgrid((np.arange(left, right) + 0.5 - x0) * kx,
                              (y0 - np.arange(top, bottom) - 0.5) * ky)
    return dbz, east, north


def bearing(east, north):
    return DIRECTIONS[int((math.degrees(math.atan2(east, north)) % 360 + 22.5) // 45) % 8]


def nearest(dbz, east, north, threshold, minimum):
    distance = np.hypot(east, north)
    mask = (dbz >= threshold) & (distance <= RADIUS_KM)
    if mask.sum() < minimum:
        return None
    index = np.where(mask, distance, np.inf).argmin()
    return {'km': float(distance.flat[index]), 'direction': bearing(east.flat[index], north.flat[index]),
            'pixels': int(mask.sum()), 'max_dbz': int(dbz[mask].max())}


def centroid(dbz, east, north):
    mask = (dbz >= MODERATE) & (np.hypot(east, north) <= RADIUS_KM * 1.5)
    if mask.sum() < 30:
        return None
    return float(east[mask].mean()), float(north[mask].mean())


def read_radar(now=None, max_age_minutes=MAX_FRAME_AGE_MINUTES):
    response = requests.get(BASE + 'contour/radar/radarContour.js', headers=HEADERS, timeout=30)
    response.raise_for_status()
    paths = re.findall(r'radar/cappi/[^"\']+?\.png[^"\']*', response.text)
    entries = []
    for path in paths:
        match = re.search(r'(\d{12})\.png', path)
        try:
            if match:
                entries.append((datetime.strptime(match.group(1), '%Y%m%d%H%M').replace(tzinfo=BJT), path))
        except ValueError:
            continue
    if not entries:
        return {'error': '雷达帧列表为空或无有效时次', 'fresh': False}
    times = [entry[0] for entry in entries]
    latest_time, latest_path = max(entries, key=lambda entry: entry[0])
    result = {'first': f'{min(times):%Y-%m-%d %H:%M}', 'last': f'{latest_time:%Y-%m-%d %H:%M}',
              'frame_count': len(entries), 'max_age_minutes': max_age_minutes,
              **freshness(latest_time, now, max_age_minutes)}
    if not result['fresh']:
        result['error'] = result['freshness_reason']
        return result
    last = frame(latest_path)
    result.update(strong=nearest(*last, STRONG, 10), moderate=nearest(*last, MODERATE, 30))
    result['motion_status'] = 'unavailable'
    if len(entries) < 2:
        result['motion_reason'] = '仅有一帧雷达资料，无法估计移动'
    elif any(end <= start for start, end in zip(times, times[1:])):
        result['motion_reason'] = '雷达帧时次重复或未递增，无法估计移动'
    else:
        first = frame(entries[0][1])
        shift = match_shift(first[0], last[0])
        result.update(motion_reason=shift['reason'], motion_correlation=shift['correlation'],
                      motion_peak_margin=shift['peak_margin'])
        if shift['status'] == 'ok':
            hours = (latest_time - times[0]).total_seconds() / 3600
            east, north = last[1:]
            shift_east = -shift['dx'] * abs(east[0, 1] - east[0, 0])
            shift_north = shift['dy'] * abs(north[1, 0] - north[0, 0])
            result.update(move_kmh=math.hypot(shift_east, shift_north) / hours,
                          move_direction=bearing(shift_east, shift_north), motion_status='ok')
    return result


def heading(direction):
    return direction if len(direction) == 2 else '偏' + direction


def place(echo):
    if echo['km'] < 10:
        return '市民中心附近'
    return f"{heading(echo['direction'])}方向约{max(5, round(echo['km'] / 5) * 5)}公里处"


def describe(radar):
    if not radar or 'error' in radar:
        return '（雷达资料读取失败，请人工补充）'
    strong, moderate = radar.get('strong'), radar.get('moderate')
    if not strong and not moderate:
        return '我市及周边暂无明显降水回波'
    parts = []
    if moderate and (not strong or moderate['km'] < strong['km'] - 15):
        parts.append(f'{place(moderate)}有降水回波')
    if strong:
        parts.append(f'{place(strong)}有较强对流回波')
    if 'move_kmh' not in radar:
        motion = '，回波移动情况暂无法可靠判断'
    elif radar['move_kmh'] < 10:
        motion = '，移动缓慢'
    else:
        motion = f"，回波总体向{heading(radar['move_direction'])}方向移动"
    return '，'.join(parts) + motion


def summary(radar):
    if not radar:
        return '未读取'
    if 'error' in radar:
        return f"读取失败：{radar['error']}"
    parts = [f"帧时次 {radar['first']}—{radar['last']}（北京时）"]
    for key, label in (('strong', f'≥{STRONG}dBZ'), ('moderate', f'≥{MODERATE}dBZ')):
        echo = radar.get(key)
        parts.append(f'{label}：' + (f"最近在{echo['direction']} {echo['km']:.0f} km，{RADIUS_KM} km 内 {echo['pixels']} 像素，"
                                     f"最强 {echo['max_dbz']} dBZ" if echo else f'{RADIUS_KM} km 内无'))
    if 'move_kmh' in radar:
        parts.append(f"回波移向{radar['move_direction']}，约 {radar['move_kmh']:.0f} km/h（首末帧互相关估算，仅供参考）")
    elif radar.get('motion_reason'):
        parts.append('移动不可判断：' + radar['motion_reason'])
    if radar.get('age_minutes') is not None:
        parts.append(f"末帧距当前 {radar['age_minutes']:.0f} 分钟")
    return '；'.join(parts)
