"""Read the public Shenzhen Meteorological Observatory feed behind weather.sz.gov.cn."""
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests

BASE = 'https://weather.121.com.cn/data_cache/'
BJT = timezone(timedelta(hours=8))
# Operational freshness limits, separate from the meteorological risk thresholds.
MAX_AGE_MINUTES = {'alarm': 30, 'observation': 90, 'futian': 1080,
                   'tenday': 2160, 'tips': 2160, 'commute': 2160, 'haze': 2160}
FUTURE_TOLERANCE = timedelta(minutes=5)
FEEDS = {
    'tips': ('szWeather/szImportantTips.js', 'SZ121_szImportantTips'),
    'futian': ('szWeather/fenquyubao/MESISarea_ft.js', 'WF_ft'),
    'tenday': ('szWeather/sz10day_new.js', 'SZ121_10dayWeather'),
    'alarm': ('szWeather/alarm/szAlarm.js', 'SZ121_AlarmInfo'),
    'observation': ('szWeather/szOssmo.js', 'SZ121_szOssmo'),
    'commute': ('szWeather/szShangxiabanTipsNew.js', 'SZ121_szShangxiabanTips'),
    'haze': ('szWeather/SZ121_Haze.js', 'SZ121_Haze'),
}


def parse_time(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=BJT) if value.tzinfo is None else value.astimezone(BJT)
    text = str(value or '').strip()
    try:
        moment = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        found = re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?', text)
        if not found:
            return None
        try:
            moment = datetime(*[int(x or 0) for x in found.groups()])
        except ValueError:
            return None
    return moment.replace(tzinfo=BJT) if moment.tzinfo is None else moment.astimezone(BJT)


def validate_feed(key, feed, now=None):
    """Return an invalid/missing feed as an error, never as an all-clear.

    Cache creation time is required: a fresh HTTP response can contain old data.
    Actual observation time is checked independently from cache creation time.
    ``now`` is injectable for offline tests and must be the actual collection time.
    """
    now = parse_time(now or datetime.now(BJT))
    result = dict(feed or {})
    if 'error' in result:
        result.pop('data', None)
        result['fresh'] = False
        return result
    data = result.get('data')
    reason = None
    if not isinstance(data, dict):
        reason = '资料缺失或结构无效'
    elif key == 'alarm' and not any(name in data for name in ('ingnalNum', 'subAlarm')):
        reason = '预警资料缺少状态字段，无法确认是否有生效信号'
    cache = parse_time(result.get('cache_time'))
    if reason is None and cache is None:
        reason = '资料缓存时间缺失或无法解析'
    if reason is None:
        age = now - cache
        if age < -FUTURE_TOLERANCE:
            reason = '资料缓存时间晚于实际当前时间'
        elif age > timedelta(minutes=MAX_AGE_MINUTES.get(key, 2160)):
            reason = f'资料过期（缓存时间 {cache:%Y-%m-%d %H:%M}）'
    if reason is None and key == 'observation':
        observed = parse_time(data.get('date'))
        if observed is None:
            reason = '实况时间缺失或无法解析'
        elif not -FUTURE_TOLERANCE <= now - observed <= timedelta(minutes=MAX_AGE_MINUTES[key]):
            reason = '实况资料过期或时间异常'
    if reason is None and key in ('futian', 'tenday'):
        items = (data.get('hoursweather') or []) + ([data['dayreport']] if data.get('dayreport') else []) if key == 'futian' else data.get('daynew') or []
        raw = [item['pubTime'] for item in items if item.get('pubTime')]
        moments = [parse_time(value) for value in raw]
        if raw and any(value is None for value in moments):
            reason = '预报发布时间无法解析'
        elif moments and not -FUTURE_TOLERANCE <= now - max(moments) <= timedelta(minutes=MAX_AGE_MINUTES[key]):
            reason = '预报发布时间过期或异常（不能仅凭新缓存视为新预报）'
    if reason is not None:
        result['error'] = reason
        result.pop('data', None)
    result['fresh'] = reason is None
    result['validated_at'] = now.isoformat()
    return result


def validate(official, now=None):
    return {key: validate_feed(key, (official or {}).get(key), now) for key in FEEDS}


def fetch(archive=None, now=None):
    if archive:
        archive.mkdir(parents=True, exist_ok=True)
    result = {}
    for key, (path, variable) in FEEDS.items():
        try:
            response = requests.get(BASE + path, timeout=(10, 30), headers={'Referer': 'https://weather.sz.gov.cn/',
                                                                         'Cache-Control': 'no-cache'})
            response.raise_for_status()
            text = response.content.decode('utf-8')
            match = re.search(r'var\s+' + variable + r'\s*=\s*(\{.*\})\s*;?\s*\}\s*catch', text, re.S)
            if not match:
                raise ValueError(f'{variable} not found')
            created = re.search(r'@cdate:([^*]+)\*', text)
            result[key] = {'cache_time': created.group(1).strip() if created else None,
                           'data': json.loads(match.group(1))}
            if archive:
                (archive / Path(path).name).write_text(text, encoding='utf-8')
        except Exception as error:
            result[key] = {'error': f'{type(error).__name__}: {error}'}
    return validate(result, now)
