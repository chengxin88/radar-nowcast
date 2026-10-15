"""每15分钟 市民中心区域夜间天气研判（16时模板 + 雨几时到、雨量、时长）。"""
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import requests

import sz_lightning
import sz_official
import sz_radar
import radar_track
from event_briefing import publication_checks, haze_for, brief_visibility
from ec_event_point import BJT

STRONG = 35
HEADERS = {'Referer': 'https://weather.sz.gov.cn/'}


def frames():
    return radar_track.frames()[-4:]


def rate(dbz):  # Z = 300 R^1.4
    return (10 ** (dbz / 10) / 300) ** (1 / 1.4)


def analyse(fr, now=None):
    """Approach of the >=35 dBZ front toward the site, band depth along motion, rain rate."""
    if not fr:
        return {'error': '雷达帧列表为空', 'max_here': None, 'max_10': None}
    fresh = sz_radar.freshness(max(f[0] for f in fr), now)
    if not fresh['fresh']:
        return {'error': fresh['freshness_reason'], 'max_here': None, 'max_10': None}
    near = []
    for when, dbz, east, north in fr:
        dist = np.hypot(east, north)
        mask = (dbz >= STRONG) & (dist <= 100)
        near.append((when, float(dist[mask].min()) if mask.sum() >= 10 else None))
    when0, dbz, east, north = fr[-1]
    dist = np.hypot(east, north)
    res = {'time': when0, 'max_here': int(dbz[dist <= 3].max()) if (dist <= 3).any() else 0,
           'max_10': int(dbz[dist <= 10].max()), 'near': near}
    pts = [(w, d) for w, d in near if d is not None]
    move = radar_track.motion(fr, return_details=True)
    res['motion_status'] = move['status']
    if len(pts) >= 2 and move['status'] == 'ok':
        h = np.array([(w - pts[0][0]).total_seconds() / 3600 for w, _ in pts])
        d = np.array([d for _, d in pts])
        res['approach_kmh'] = float(-np.polyfit(h, d, 1)[0]) if h[-1] > 0 else 0.0
    res['front_km'] = pts[-1][1] if pts else None
    m = (dbz >= STRONG) & (dist <= 100)
    if m.sum() >= 10:
        i = np.where(m, dist, np.inf).argmin()
        res['front_dir'] = sz_radar.bearing(east.flat[i], north.flat[i])
    if move['status'] == 'ok':
        ve, vn = move['ve'], move['vn']
        res['move'] = (ve, vn)
        speed = math.hypot(ve, vn)
        if speed > 5:
            ue, un = ve / speed, vn / speed
            along = east * ue + north * un          # >0 downstream of site
            across = np.abs(-east * un + north * ue)
            corridor = (across <= 8) & (along < 0) & (dbz >= STRONG)
            if corridor.sum() >= 5:
                res['depth_km'] = float(along[corridor].max() - along[corridor].min())
                res['corridor_max'] = int(dbz[corridor].max())
                res['corridor_mean_rate'] = float(rate(dbz[corridor]).mean())
            heavy = (across <= 8) & (along < 0) & (dbz >= 20)
            if heavy.sum() >= 5:
                res['rain_depth_km'] = float(along[heavy].max() - along[heavy].min())
            res['speed'] = speed
    return res


def aws_rain():
    url = os.environ.get('LIGHTSHOW_AWS_RAIN_URL', '')
    if not url:
        return None
    try:
        response = requests.get(url, timeout=(5, 20))
        response.raise_for_status()
        out = response.text
        rows = json.loads(out[out.find('['):out.rfind(']') + 1])
    except Exception:
        return None
    def f(v):
        try:
            return float(v)
        except Exception:
            return 0.0
    ft = [r for r in rows if '福田' in str(r.get('NAME', ''))]
    top = max(rows, key=lambda r: f(r.get('R01H')), default=None)
    return {'futian_max1h': max((f(r.get('R01H')) for r in ft), default=0.0),
            'top': (top.get('NAME'), top.get('SUBDISTRICT') or top.get('OBTNAME'), f(top.get('R01H'))) if top else None,
            'time': rows[0].get('DDATETIME') if rows else None}


def futian_alarm(official):
    data = (official.get('alarm') or {}).get('data') or {}
    names = []
    for a in data.get('subAlarm') or []:
        area = a.get('alarmArea', '')
        if '福田' in area or '全市' in area or '陆地' in area:
            names.append(f"{a.get('alarmColor', '')}{a.get('alarmType', '')}")
    return names


def latest_nowcast(official, now):
    tips = (official.get('tips') or {}).get('data') or {}
    items = [tips.get('important')] + list(tips.get('more') or [])
    items = [i for i in items if i and i.get('d') and i.get('prompt') and '短临' in i['prompt'][:12]]
    items.sort(key=lambda i: i['d'], reverse=True)
    if not items:
        return None
    i = items[0]
    t = datetime.strptime(i['d'][:16], '%Y-%m-%d %H:%M').replace(tzinfo=BJT)
    return (t, i['prompt']) if timedelta(0) <= now - t <= timedelta(hours=3) else None


def hm(t):
    return f'{t.hour}:{t.minute:02d}'


def build(now=None, no_lightning=True):
    now = now or datetime.now(BJT)
    official = sz_official.fetch(now=now)
    fr = frames()
    radar = analyse(fr, now)
    if 'error' in radar:
        raise RuntimeError('无法生成短临稿：' + radar['error'])
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    blockers = publication_checks(official, day, now, {'last': radar['time'].strftime('%Y-%m-%d %H:%M')}, now)
    if blockers:
        raise RuntimeError('无法生成可发布短临稿：' + '；'.join(blockers))
    lightning = sz_lightning.read_lightning(enabled=not no_lightning)
    aws = aws_rain()
    alarm = futian_alarm(official)
    nc = latest_nowcast(official, now)
    hours = {}
    for item in ((official.get('futian') or {}).get('data') or {}).get('hoursweather', []):
        t = datetime.strptime(item['forecasttime'][:16], '%Y-%m-%d %H:%M').replace(tzinfo=BJT)
        hours[t] = item
    window = [v for k, v in hours.items() if now <= k <= now.replace(hour=22, minute=0) + timedelta(minutes=1)]
    temps = [float(v['minT']) for v in window] + [float(v['maxT']) for v in window]
    obs = ((official.get('observation') or {}).get('data') or {})

    raining_here = radar['max_here'] >= 20 or bool(aws and aws['futian_max1h'] > 0.5)
    front, appr = radar.get('front_km'), radar.get('approach_kmh', 0)
    track = radar_track.estimate(fr=fr, now=now)
    arrive = radar_track.text(track, now)
    weak = track['raining'] and track['here'] < 35 and track.get('lead_km') is None and track['motion_status'] == 'ok'
    if weak:
        e20 = track.get('end20_t')
        if e20 is not None:
            end = e20 + timedelta(minutes=15)
            end = radar_track.round_half_hour(end)
            arrive = (f'目前市民中心区域以中小雨为主，{radar_track.hm(end, now)}前后降雨基本结束'
                      if end > now else '雷达外推的雨停时刻已过，请核对最新雷达和实况')
        else:
            arrive = '目前市民中心区域以中小雨为主，强降雨已基本结束，雨停时间请关注后续研判'
    rain = ''
    top = max(radar.get('corridor_max', 0), radar['max_here'])
    if not weak and track['motion_status'] == 'ok' and (track['raining'] or (track.get('lead_t') and timedelta(0) < track['lead_t'] - now <= timedelta(hours=3))):
        hourly = '小时雨量可达30-50毫米' if top >= 50 else '小时雨量可达20-30毫米' if top >= 45 else '小时雨量10-20毫米'
        rain = f'雨带影响时雨势较大，{hourly}'
    t_now = float(obs['t']) if obs.get('t') not in (None, '') else None
    temps = temps + ([t_now] if t_now is not None else [])
    t_text = f'{math.floor(min(temps)):.0f}-{math.ceil(max(temps)):.0f}℃' if temps else '（气温资料缺测，请人工补充）'
    wind = ((window[0]['wd'] if window[0]['wd'].endswith('风') else window[0]['wd'] + '风') + f"{window[0]['ws']}级") if window else '（风预报缺测，请人工补充）'
    start_h = min(max(now.hour, 18), 21)
    strong_case = track['raining'] or bool(track.get('lead_t') and timedelta(0) < track['lead_t'] - now <= timedelta(hours=2))
    sky = '阴天有小到中雨，雨势逐渐减弱' if weak else '阴天有雷阵雨，局地雨势较大' if strong_case else '以多云为主，有短时（雷）阵雨'
    alarm_text = f'目前福田区{"、".join(alarm)}预警信号生效中。' if alarm else ''
    impact = ('雷阵雨对市民中心区域户外活动影响较大，请提前做好人员疏散和防雷避险准备，并关注最新天气预报和预警信号。'
              if strong_case else '该时段可能出现短时（雷）阵雨，对市民中心区域户外活动有一定影响，请关注最新天气预报和预警信号。')
    parts = [f'预计{start_h}-22时，市民中心区域{sky}。', arrive + '。']
    if rain:
        parts.append(rain + '；雷雨时可能伴有雷电和短时大风。')
    parts.append(f'气温在{t_text}之间，{wind}。' if weak else f'气温在{t_text}之间，{wind}，雷雨时局地阵风较大。')
    haze = (official.get('haze') or {})
    haze_item = haze_for(haze.get('data'), day + timedelta(hours=19.5))
    parts.append(brief_visibility('未读取' if 'error' in haze else (haze_item or {}).get('haze_level', '无'), 2 if strong_case else 1, False) + '。')
    if not sz_lightning.available(lightning):
        parts.append('闪电定位资料缺测（云端已禁用），请人工核对。' if lightning.get('disabled') else '闪电定位资料缺测，请人工核对。')
    parts.append((alarm_text.rstrip('。') + '，请继续关注最新天气预报和预警信号。' if alarm_text else '请继续关注最新天气预报和预警信号。') if weak else alarm_text + impact)
    header = f'【市民中心区域夜间天气研判】{now.month}月{now.day}日{hm(now)}'
    return header + '\n' + ''.join(parts), radar


def radar_dir(radar):
    return radar.get('front_dir', '北')


def round5(x):
    return max(5, int(round(x / 5) * 5))


def span(a, b):
    def fmt(h):
        m = int(round(h * 60 / 10) * 10) if h < 1 else int(round(h * 2) * 30)
        return f'{m}分钟' if m < 60 else f'{m / 60:g}小时'
    x, y = fmt(a), fmt(b)
    if x == y:
        return x
    if x.endswith('小时') and y.endswith('小时'):
        return f'{x[:-2]}-{y}'
    return f'{x}至{y}'


def window_text(t):
    t = t.replace(second=0, microsecond=0)
    now = datetime.now(BJT)
    if t <= now:
        return '外推到达时刻已过，请核对最新雷达'
    if t - now <= timedelta(minutes=20):
        return '未来20分钟内'
    center = radar_track.q15(t, True)
    return f'{radar_track.hm(center - timedelta(minutes=15), now)}—{radar_track.hm(center + timedelta(minutes=15), now)}'


if __name__ == '__main__':
    text, radar = build()
    print(text)
    print('#radar', {k: v for k, v in radar.items() if k != 'time'}, file=sys.stderr)
