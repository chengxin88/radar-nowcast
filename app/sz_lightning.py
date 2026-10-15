"""Read lightning-location data from the Shenzhen lightning alert service on the bureau intranet (read-only)."""
import json
import math
import os
import re
import subprocess
from datetime import datetime, timedelta

import requests

from ec_event_point import BJT, SITE_LAT, SITE_LON
from sz_radar import bearing, place

URL = os.environ.get('LIGHTSHOW_LIGHTNING_URL', '')
PROXY = os.environ.get('LIGHTSHOW_LIGHTNING_PROXY', '')
WINDOWS_CURL = os.environ.get('LIGHTSHOW_WINDOWS_CURL', '')
KINDS = {1: '云地闪', 0: '云间闪'}
WINDOW_MIN = 60
RADIUS_KM = 50


def parse(text, kind):
    flashes = []
    for row in (text or '').strip().splitlines()[1:]:
        stamp, lon, lat, peak = row.split(',')[:4]
        east = (float(lon) - SITE_LON) * 111.32 * math.cos(math.radians(SITE_LAT))
        north = (float(lat) - SITE_LAT) * 110.57
        km = math.hypot(east, north)
        if km <= RADIUS_KM:
            moment = datetime.fromtimestamp(int(re.search(r'-?\d+', stamp).group()) / 1000, BJT)
            flashes.append({'time': moment, 'km': km, 'direction': bearing(east, north), 'kind': KINDS[kind],
                            'peak_ka': float(peak)})
    return flashes


def poster(proxy=None):
    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies = {'http': proxy}
    def post(body):
        response = session.post(URL, json=body, timeout=(4, 15))
        response.raise_for_status()
        return response.json()
    return post

def post_windows(body):
    result = subprocess.run([WINDOWS_CURL, '--noproxy', '*', '-sf', '-m', '15', '-H', 'Content-Type: application/json',
                             '--data-binary', '@-', URL], input=json.dumps(body).encode(), capture_output=True, timeout=25)
    if result.returncode:
        raise ConnectionError(f'curl exit {result.returncode}')
    return json.loads(result.stdout)

def routes():
    result = [('直连', poster())]
    if PROXY:
        result.append(('经代理', poster(PROXY)))
    if WINDOWS_CURL and os.path.isfile(WINDOWS_CURL):
        result.append(('经Windows', post_windows))
    return result

def query(post, kind, start, end):
    body = {'lng': SITE_LON, 'lat': SITE_LAT, 'radius': 100, 'starttime': f'{start:%Y-%m-%dT%H:%M:%S}',
            'endtime': f'{end:%Y-%m-%dT%H:%M:%S}', 'ftype': kind}
    return parse(post(body)['d'], kind)


def read_lightning(end=None, enabled=None):
    if enabled is None:
        enabled = os.environ.get('LIGHTSHOW_ENABLE_LIGHTNING', '').lower() in ('1', 'true', 'yes')
    if not enabled:
        return {'disabled': True, 'error': '云端配置已禁用闪电定位，资料缺测'}
    if not URL:
        return {'error': '未配置 LIGHTSHOW_LIGHTNING_URL，闪电资料缺测'}
    end = (end or datetime.now(BJT)).astimezone(BJT).replace(tzinfo=None, second=0, microsecond=0)
    start = end - timedelta(minutes=WINDOW_MIN)
    errors = []
    for route, post in routes():
        try:
            flashes = query(post, 1, start, end) + query(post, 0, start, end)
            flashes = [flash for flash in flashes if start.replace(tzinfo=BJT) <= flash['time'] <= end.replace(tzinfo=BJT)]
        except Exception as error:
            errors.append(f'{route} {type(error).__name__}')
            continue
        return {'start': f'{start:%Y-%m-%d %H:%M}', 'end': f'{end:%Y-%m-%d %H:%M}', 'route': route,
                'flashes': flashes}
    return {'error': '；'.join(errors)}


def available(lightning):
    return bool(lightning) and 'flashes' in lightning


def describe(lightning):
    flashes = lightning['flashes']
    if not flashes:
        return f'闪电定位显示，近1小时市民中心{RADIUS_KM}公里范围内无闪电'
    nearest = min(flashes, key=lambda flash: flash['km'])
    latest = max(flash['time'] for flash in flashes)
    ground = sum(flash['kind'] == '云地闪' for flash in flashes)
    return (f'闪电定位显示，近1小时市民中心{RADIUS_KM}公里范围内监测到闪电{len(flashes)}次（云地闪{ground}次），'
            f'距离最近的位于{place(nearest)}，最新一次在{latest.hour}时{latest.minute}分')


def summary(lightning):
    if not lightning:
        return '本时次未读取'
    if 'error' in lightning:
        return f"闪电资料缺测（{lightning['error']}）"
    flashes = lightning['flashes']
    parts = [f"{lightning['start']}—{lightning['end'][11:]}（{lightning['route']}），市民中心{RADIUS_KM}公里内"]
    for kind in KINDS.values():
        group = [flash for flash in flashes if flash['kind'] == kind]
        if group:
            nearest = min(group, key=lambda flash: flash['km'])
            parts.append(f"{kind} {len(group)} 次，最近 {nearest['direction']} {nearest['km']:.0f} km，"
                         f"最新 {max(flash['time'] for flash in group):%H:%M}，最大强度 {max(abs(flash['peak_ka']) for flash in group):.0f} kA")
        else:
            parts.append(f'{kind} 0 次')
    return '；'.join(parts)
