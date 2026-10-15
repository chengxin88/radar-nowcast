"""Fresh, region-bound official context. Never substitute Futian for a new site."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import hashlib
import math
import os
from pathlib import Path
import re

import requests
import sz_official
import yjt_observation

BJT = sz_official.BJT
DEFAULT_PROFILE = {
    'target': {'lat': 22.543, 'lon': 114.059, 'radius_km': 3.0},
    'area': '福田区', 'area_aliases': ['福田'], 'city': '深圳',
    'station_name': '深圳国家基本气象站（福田区，区域参考）',
    'yjt_station': {'name': '市民中心', 'subdistrict': '莲花街道', 'areaid': '5949302',
                    'target': {'lat': 22.543, 'lon': 114.059, 'radius_km': 3.0}},
    'feeds': {key: list(sz_official.FEEDS[source]) for key, source in
              (('hourly', 'futian'), ('observation', 'observation'), ('alarm', 'alarm'), ('tips', 'tips'))},
}


def event_stamp(item, target):
    payload = {'id': item['id'], 'time': item['time'], 'text': item['text'],
               'target': {k: float(target[k]) for k in ('lat', 'lon', 'radius_km')}}
    return 'official:' + hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def profile_for(target, path=None):
    path = path if path is not None else os.environ.get('OFFICIAL_PROFILE_FILE')
    profile = json.loads(Path(path).read_text()) if path else DEFAULT_PROFILE
    # Exact geometry binding is intentional: naming a different district alone
    # does not prove that its forecasts/warnings apply to the configured circle.
    if any(float(profile['target'][k]) != float(target[k]) for k in ('lat', 'lon', 'radius_km')):
        return None
    if not profile.get('area') or not profile.get('city') or not profile.get('area_aliases'):
        raise ValueError('官方资料配置缺少地区或预警范围对应关系')
    station = profile.get('yjt_station')
    if station is not None and (not isinstance(station, dict) or any(
            not isinstance(station.get(key), str) or not station[key].strip()
            or len(station[key]) > 80 or any(c in station[key] for c in '<>\r\n')
            for key in ('name', 'subdistrict', 'areaid'))):
        raise ValueError('降水风速实况站配置无效')
    for key, feed in profile['feeds'].items():
        if key not in ('hourly', 'observation', 'alarm', 'tips') or len(feed) != 2:
            raise ValueError('官方资料配置结构无效')
        path, variable = feed
        if not re.fullmatch(r'[A-Za-z0-9_./-]+\.js', path) or '..' in path or path.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_]+', variable):
            raise ValueError('官方资料路径无效')
    return profile


def read_feed(key, descriptor):
    path, variable = descriptor
    try:
        response = requests.get(sz_official.BASE + path, timeout=(5, 20), headers={
            'Referer': 'https://weather.sz.gov.cn/', 'Cache-Control': 'no-cache'})
        response.raise_for_status()
        text = response.content.decode('utf-8')
        match = re.search(r'var\s+' + re.escape(variable) + r'\s*=\s*(\{.*\})\s*;?\s*\}\s*catch', text, re.S)
        if not match:
            raise ValueError('官方资料格式异常')
        created = re.search(r'@cdate:([^*]+)\*', text)
        return {'data': json.loads(match.group(1)), 'cache_time': created.group(1).strip() if created else None,
                'source': sz_official.BASE + path}
    except (requests.RequestException, ValueError, UnicodeError) as error:
        return {'error': '官方资料获取失败：' + type(error).__name__}


def _yjt_station_for(profile):
    station = profile.get('yjt_station')
    if not isinstance(station, dict):
        return None
    try:
        if any(float(station['target'][key]) != float(profile['target'][key])
               for key in ('lat', 'lon', 'radius_km')):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return station


def fetch(profile):
    if profile is None:
        return {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {key: pool.submit(read_feed, key, descriptor) for key, descriptor in profile['feeds'].items()}
        station = _yjt_station_for(profile)
        if station:
            jobs['yjt_observation'] = pool.submit(yjt_observation.fetch, station)
        return {key: job.result() for key, job in jobs.items()}


def number(value, lower, upper):
    try:
        n = float(value)
        return n if math.isfinite(n) and lower <= n <= upper else None
    except (ValueError, TypeError):
        return None


def clean(value):
    return ' '.join(str(value or '').split())


def warning_scope(area, profile):
    area = clean(area)
    aliases = [profile['area']] + profile['area_aliases']
    if not area:
        return 'unknown'
    if any(re.search(r'(除|不含|不包括|不包含)\s*' + re.escape(alias), area) for alias in aliases):
        return 'other'
    if any(alias in area for alias in aliases):
        # Street-specific warnings are retained explicitly as partial coverage,
        # not automatically applied to the whole configured forecast circle.
        return 'partial' if any(word in area for word in ('街道', '部分', '局部', '高地', '沿海')) else 'local'
    if any(word in area for word in ('全市陆地', '全陆地')) or ('全市' in area and '海区' not in area):
        return 'local'
    if area == '陆地':
        return 'local'
    return 'other'


def recent_nowcast(feed, now):
    if not feed.get('fresh'):
        return None
    data = feed.get('data') or {}
    candidates = []
    for item in [data.get('important')] + list(data.get('more') or []):
        if not isinstance(item, dict):
            continue
        text = clean(item.get('prompt'))
        t = sz_official.parse_time(item.get('d'))
        if t is not None and timedelta(0) <= now - t <= timedelta(hours=3) and '短临' in (clean(item.get('t')) + text[:20]):
            candidates.append({'time': t.isoformat(), 'text': text, 'id': str(item.get('taskid') or t.isoformat())})
    return max(candidates, key=lambda row: row['time'], default=None)


def summarize(raw, profile, now):
    now = sz_official.parse_time(now)
    out = {'matched': profile is not None, 'checks': [], 'hourly': [], 'observation': None,
           'warnings': [], 'other_warnings': [], 'warning_unknown': True, 'nowcast': None,
           'forecast_complete': False, 'feeds': {}}
    if profile is None:
        out['checks'].append('该预报区域尚未绑定匹配的官方资料源，未套用福田区预报、实况或预警')
        return out
    out.update(area=profile['area'], station_name=profile.get('station_name', '配置的参考站'), city=profile['city'])
    station = _yjt_station_for(profile)
    if station:
        out['station_observation_requested'] = True
        out['station_observation_name'] = station['name']
        out['station_observation'] = yjt_observation.summarize(raw.get('yjt_observation'), station, now)
        out['checks'].extend(out['station_observation']['checks'])
    elif profile.get('yjt_station'):
        out['checks'].append('降水风速实况站尚未绑定该预报区域，未套用其他地区站点资料')
    feeds = {key: sz_official.validate_feed('futian' if key == 'hourly' else key, raw.get(key), now)
             for key in ('hourly', 'observation', 'alarm', 'tips')}
    out['feeds'] = feeds
    for key, feed in feeds.items():
        if not feed.get('fresh'):
            out['checks'].append({'hourly': '逐时预报', 'observation': '实况', 'alarm': '预警', 'tips': '短临'}[key] + '资料缺测或过期')
    obs = feeds['observation'].get('data')
    if obs:
        out['observation'] = {'time': obs['date'], 'temperature': number(obs.get('t'), -60, 60),
            'humidity': number(obs.get('u'), 0, 100), 'visibility': number(obs.get('v'), 0, 100),
            'rain_1h': number(obs.get('h01r'), 0, 500)}
    rows = (feeds['hourly'].get('data') or {}).get('hoursweather') or []
    start = now.replace(minute=0, second=0, microsecond=0)
    end = (now + timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    if now.minute or now.second or now.microsecond:
        end += timedelta(hours=1)
    selected = {}
    for row in rows:
        t = sz_official.parse_time(row.get('forecasttime'))
        published = sz_official.parse_time(row.get('pubTime'))
        area = clean(row.get('areaName') or row.get('area'))
        if (t is None or published is None or not timedelta(0) <= now - published <= timedelta(hours=18)
                or area not in [profile['area']] + profile['area_aliases'] or not start <= t <= end):
            continue
        state = clean(row.get('state') or row.get('report'))
        # Reject non-meteorological/malformed values instead of echoing HTML.
        if not state or len(state) > 80 or any(c in state for c in '<>'):
            continue
        selected[t] = {'time': t.isoformat(), 'published': published.isoformat(), 'state': state,
            'min_t': number(row.get('minT'), -60, 60), 'max_t': number(row.get('maxT'), -60, 60),
            'wind_direction': clean(row.get('wd')), 'wind_level': clean(row.get('ws'))}
    out['hourly'] = [selected[t] for t in sorted(selected)]
    expected = [start + timedelta(hours=k) for k in range(int((end-start).total_seconds()/3600)+1)]
    out['forecast_complete'] = all(t in selected for t in expected)
    if not out['forecast_complete']:
        out['checks'].append('逐时预报未完整覆盖未来两小时，不用缺失时段推断天气')
    alarm = feeds['alarm'].get('data')
    if alarm is not None:
        rows = alarm.get('subAlarm')
        out['warning_unknown'] = not isinstance(rows, list)
        if rows == [] and str(alarm.get('ingnalNum', '0')) not in ('0', ''):
            out['warning_unknown'] = True
        for row in rows or []:
            if not isinstance(row, dict):
                out['warning_unknown'] = True
                continue
            area = clean(row.get('alarmArea'))
            name = clean(row.get('alarmColor')) + clean(row.get('alarmType'))
            if not row.get('alarmType'):
                out['warning_unknown'] = True
                continue
            scope = warning_scope(area, profile)
            item = {'name': name, 'area': area, 'scope': scope, 'issued': row.get('date')}
            if scope in ('local', 'partial'):
                out['warnings'].append(item)
            elif scope == 'unknown':
                out['warning_unknown'] = True
            else:
                out['other_warnings'].append(item)
    out['nowcast'] = recent_nowcast(feeds['tips'], now)
    return out


def _reference_observation_text(context, include_rain=True, *, compact=False):
    obs = context.get('observation')
    if not obs:
        return '当地有效地面实况暂缺。'
    items = []
    for key, label, unit in (('temperature', '气温', '℃'), ('humidity', '相对湿度', '%'), ('visibility', '能见度', '公里')):
        if obs[key] is not None:
            items.append(label + f'{obs[key]:g}' + unit)
    if include_rain and obs['rain_1h'] is not None:
        items.append('该站过去1小时未记录到降水' if obs['rain_1h'] == 0 else '该站过去1小时记录到降水')
    if not items:
        return '当地实况要素暂缺。'
    if compact:
        station = context['station_name'].split('（', 1)[0]
        if station == '深圳国家基本气象站':
            station = '国家基本站'
        return '，'.join(items) + f'（{station}）。'
    t = sz_official.parse_time(obs['time'])
    return f'截至{t:%H:%M}，{context["station_name"]}：' + '，'.join(items) + '。'


def observation_text(context):
    if not context.get('station_observation_requested'):
        return _reference_observation_text(context)
    station = context['station_observation_name']
    station = station if station.endswith('站') else station + '站'
    observed = context.get('station_observation') or {}
    rain, wind = observed.get('rain'), observed.get('wind')
    parts = []
    displayed_time = None

    def time_prefix(value):
        nonlocal displayed_time
        moment = sz_official.parse_time(value)
        # Nearby observations share a display time; retain independent times
        # when sources diverge. Their original timestamps stay in the result.
        if displayed_time is None or abs(moment - displayed_time) > timedelta(minutes=5):
            displayed_time = moment
            return f'截至{moment:%H:%M}，'
        return ''

    if rain:
        values = []
        for key, label in (('rain_5m', '5分钟'), ('rain_10m', '10分钟'),
                           ('rain_30m', '30分钟'), ('rain_1h', '1小时')):
            if rain.get(key) is not None:
                values.append(label + f'{rain[key]:g}毫米')
        if values:
            parts.append(time_prefix(rain['time']) + station + '过去' + '、'.join(values) + '（累计雨量）。')
    if not parts:
        parts.append(station + '降水实况暂缺。')
    if wind and wind.get('max_wind_1h_m_s') is not None:
        parts.append(time_prefix(wind['time']) + f'过去1小时极大风速{wind["max_wind_1h_m_s"]:g}米/秒。')
    else:
        parts.append(station + '风速实况暂缺。')
    # The YJT pages supply rainfall and maximum wind, not temperature/humidity.
    # Keep other valid official elements tied to their reference site with a
    # compact source note, without repeating nearby observation timestamps.
    if context.get('observation'):
        reference = context['observation']
        prefix = time_prefix(reference['time']) if any(
            reference[key] is not None for key in ('temperature', 'humidity', 'visibility')) else ''
        parts.append(prefix + _reference_observation_text(context, include_rain=False, compact=True))
    return ''.join(parts)


def weather_text(context, rows=None):
    rows = context['hourly'] if rows is None else rows
    if not rows:
        return '当地逐时预报资料暂缺。'
    states = list(dict.fromkeys(row['state'] for row in rows))
    text = '转'.join(states) if len(states) <= 2 else '、'.join(states)
    temps = [r[k] for r in rows for k in ('min_t', 'max_t')]
    if all(t is not None for t in temps):
        text += '，气温' + (f'{min(temps):g}～{max(temps):g}' if min(temps) != max(temps) else f'{min(temps):g}') + '℃'
    winds = []
    for row in rows:
        d, force = row['wind_direction'], row['wind_level']
        if (re.fullmatch(r'[东南西北偏风静无持续向转]+', d)
                and re.fullmatch(r'\d{1,2}([～~—-]\d{1,2})?', force)
                and all(0 <= int(v) <= 17 for v in re.findall(r'\d+',force))):
            winds.append(d + ('' if d.endswith('风') else '风') + force + '级')
        else:
            winds.append(None)
    if all(winds):
        text += '，' + '转'.join(dict.fromkeys(winds))
    else:
        text += '，风力资料不完整'
    return text + '。'


def warning_text(context):
    if not context.get('matched'):
        return '当地预警资料尚未匹配，预警状态需核实。'
    names = list(dict.fromkeys(r['name'] for r in context['warnings'] if r['scope'] == 'local'))
    partial = [r['area'] + '：' + r['name'] for r in context['warnings'] if r['scope'] == 'partial']
    text = f'目前{context["area"]}' + '、'.join(names) + '预警信号生效。' if names else ''
    if partial:
        text += '分区域预警（请核对具体覆盖位置）：' + '；'.join(partial) + '。'
    if context['warning_unknown']:
        text += '部分预警资料或覆盖范围不明，需核实。'
    elif not names and not partial:
        text += f'截至本次取数，未查询到覆盖{context["area"]}的生效预警信号。'
    return text
