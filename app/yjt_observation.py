"""Station observations from the public Shenzhen rainfall and wind pages.

Only the ``Real`` JSON products used by those pages are read. The wind page's
``picture_Wind_Real_1.png`` labels its one-hour field as past-hour maximum gust
(过去1小时极大风速), in m/s; it is not a mean wind or a forecast.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
import time

import requests

BJT = timezone(timedelta(hours=8))
MAX_AGE = timedelta(minutes=20)
FUTURE_TOLERANCE = timedelta(minutes=2)
SOURCES = {
    'rain': 'https://yjt.121.com.cn/Data/ForecstReal/JsonRain.json',
    'wind': 'https://yjt.121.com.cn/Data/ForecstReal/JsonWind.json',
}
PAGES = {
    'rain': 'https://yjt.121.com.cn/1districtWeather.html',
    'wind': 'https://yjt.121.com.cn/1districtWeather2.html',
}
STATION_FIELDS = ('name', 'subdistrict', 'areaid')
RAIN_FIELDS = (
    ('r05m', 'rain_5m'), ('r10m', 'rain_10m'),
    ('r30m', 'rain_30m'), ('r01h', 'rain_1h'),
)


def _valid_station(station):
    return isinstance(station, dict) and all(
        isinstance(station.get(key), str) and station[key].strip()
        for key in STATION_FIELDS)


def _matches(row, station):
    return isinstance(row, dict) and all(
        row.get(key) == station[key] for key in STATION_FIELDS)


def _read(kind, station):
    result = {'source': SOURCES[kind]}
    try:
        response = requests.get(SOURCES[kind],
            params={'time': int(time.time() * 1000)}, timeout=(5, 20),
            allow_redirects=False, headers={
                'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json',
                'Referer': PAGES[kind],
            })
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('unexpected response status')
        if len(response.content) > 5_000_000:
            raise ValueError('observation response too large')
        rows = json.loads(response.content)
        if not isinstance(rows, list):
            raise ValueError('observation JSON must be a list')
        matches = [row for row in rows if _matches(row, station)]
        if len(matches) != 1:
            raise ValueError('station missing or duplicated')
        result['data'] = matches[0]
    except (requests.RequestException, ValueError, UnicodeError) as error:
        # Avoid echoing response bodies, proxy settings, or request internals.
        result['error'] = '实况获取失败：' + type(error).__name__
    result['fetched_at'] = datetime.now(BJT).isoformat()
    return result


def fetch(station):
    """Fetch the two fixed public products; failures remain independent."""
    if not _valid_station(station):
        return {kind: {'source': source, 'fetched_at': datetime.now(BJT).isoformat(),
                       'error': '实况站点配置无效'} for kind, source in SOURCES.items()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {kind: pool.submit(_read, kind, station) for kind in SOURCES}
        return {kind: job.result() for kind, job in jobs.items()}


def _timestamp(value):
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            # A date alone is not an observation timestamp.
            if len(value.strip()) < 16:
                return None
            moment = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
    else:
        return None
    return moment.replace(tzinfo=BJT) if moment.tzinfo is None else moment.astimezone(BJT)


def _number(value, upper):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and 0 <= number <= upper else None


def summarize(raw, station, now):
    """Validate each station row using its own observation timestamp.

    Missing values remain ``None``. A dry five-minute observation does not
    establish a dry hour, and neither a station nor its records cover a circle.
    """
    out = {'rain': None, 'wind': None, 'checks': []}
    current = _timestamp(now)
    if not _valid_station(station) or current is None:
        out['checks'].append('实况站点配置或当前时间无效')
        return out
    raw = raw if isinstance(raw, dict) else {}
    for kind, label in (('rain', '降水'), ('wind', '风速')):
        feed = raw.get(kind)
        row = feed.get('data') if isinstance(feed, dict) else None
        if (not isinstance(feed, dict) or feed.get('error')
                or not _matches(row, station)):
            out['checks'].append('目标站' + label + '实况缺测或站点不匹配')
            continue
        if feed.get('source', SOURCES[kind]) != SOURCES[kind]:
            out['checks'].append('目标站' + label + '实况来源不匹配')
            continue
        observed = _timestamp(row.get('updateTime'))
        if observed is None or not -FUTURE_TOLERANCE <= current - observed <= MAX_AGE:
            out['checks'].append('目标站' + label + '实况过期或时间异常')
            continue
        normalized = {'time': observed.isoformat(), 'source': SOURCES[kind],
                      'station_name': station['name']}
        if kind == 'rain':
            values = [_number(row.get(source), 500) for source, _ in RAIN_FIELDS]
            known = [value for value in values if value is not None]
            if not known:
                out['checks'].append('目标站降水实况要素缺测')
                continue
            if any(left > right + 1e-6 for left, right in zip(known, known[1:])):
                out['checks'].append('目标站降水累计时段相互矛盾，未使用该行')
                continue
            normalized.update({target: value for (_, target), value in zip(RAIN_FIELDS, values)})
            if any(value is None for value in values):
                out['checks'].append('目标站部分降水时段缺测')
        else:
            value = _number(row.get('w01h'), 120)
            if value is None:
                out['checks'].append('目标站过去1小时极大风速缺测')
                continue
            normalized['max_wind_1h_m_s'] = value
        out[kind] = normalized
    return out
