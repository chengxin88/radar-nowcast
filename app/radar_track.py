"""雷达外推：互相关求回波移动，沿移动方向取上游剖面，估算市民中心降雨开始/强降雨结束/雨停时间。"""
import re
from datetime import datetime, timedelta

import numpy as np
import requests

import sz_radar
from ec_event_point import BJT

HEADERS = {'Referer': 'https://weather.sz.gov.cn/'}


def frames():
    response = requests.get(sz_radar.BASE + 'contour/radar/radarContour.js', headers=HEADERS, timeout=30)
    response.raise_for_status()
    out = []
    for path in re.findall(r'radar/cappi/[^"\']+?\.png[^"\']*', response.text):
        match = re.search(r'(\d{12})\.png', path)
        if not match:
            continue
        try:
            when = datetime.strptime(match.group(1), '%Y%m%d%H%M').replace(tzinfo=BJT)
        except ValueError:
            continue
        out.append((when, *sz_radar.frame(path)))
    return out


def best_shift(prev, last, maxs=25, half=110, return_details=False):
    """Return (dx, dy), or (None, None) when no reliable match is available."""
    result = sz_radar.match_shift(prev, last, maxs=maxs, half=half)
    return result if return_details else (result['dx'], result['dy'])


def motion(fr, return_details=False):
    """Unknown motion is (None, None), never a fabricated zero or edge shift."""
    result = {'ve': None, 'vn': None, 'status': 'unavailable', 'reason': None}
    if len(fr) < 2:
        result['reason'] = '雷达资料为空或仅有一帧，无法估计移动'
        return result if return_details else (None, None)
    times = [sz_radar.as_bjt(item[0]) for item in fr]
    if any(end <= start for start, end in zip(times, times[1:])):
        result['reason'] = '雷达帧时次重复或未递增，无法估计移动'
        return result if return_details else (None, None)
    i0 = max(0, len(fr) - 6)
    (t0, a, e, n), (t1, b, _, _) = fr[i0][:4], fr[-1][:4]
    shift = best_shift(a, b, return_details=True)
    result.update(correlation=shift['correlation'], peak_margin=shift['peak_margin'])
    if shift['status'] != 'ok':
        result['reason'] = shift['reason']
        return result if return_details else (None, None)
    e, n = np.asarray(e), np.asarray(n)
    if e.shape != np.asarray(a).shape or n.shape != e.shape or not np.isfinite(e).all() or not np.isfinite(n).all():
        result['reason'] = '雷达坐标网格异常，无法估计移动'
        return result if return_details else (None, None)
    dx, dy = shift['dx'], shift['dy']
    kx, ky = abs(e[0, 1] - e[0, 0]), abs(n[1, 0] - n[0, 0])
    if kx <= 0 or ky <= 0:
        result['reason'] = '雷达坐标间距非正，无法估计移动'
        return result if return_details else (None, None)
    hours = (sz_radar.as_bjt(t1) - sz_radar.as_bjt(t0)).total_seconds() / 3600
    # feature at prev (x+dx, y+dy) moved to last (x, y): displacement (-dx, -dy) px; image y points south
    result.update(ve=float(-dx * kx / hours), vn=float(dy * ky / hours), status='ok')
    return result if return_details else (result['ve'], result['vn'])


def profile(dbz, east, north, ve, vn, width=5, step=2, reach=90):
    speed = np.hypot(ve, vn)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('a positive reliable speed is required for an upstream profile')
    ue, un = ve / speed, vn / speed
    along = east * ue + north * un
    across = np.abs(-east * un + north * ue)
    prof = []
    for k in range(0, reach, step):
        m = (across <= width) & (along <= -k) & (along > -k - step)
        prof.append(int(dbz[m].max()) if m.any() else 0)
    return prof


def first(prof, test, start=0):
    return next((i for i in range(start, len(prof)) if test(prof[i])), None)


def trailing(prof, thr, start, step=2):
    for i in range(start, len(prof) - 2):
        if all(p < thr for p in prof[i:i + 3]):
            return i * step
    return None


def estimate(fr=None, now=None, max_age_minutes=sz_radar.MAX_FRAME_AGE_MINUTES):
    fr = frames() if fr is None else list(fr)
    res = {'time': None, 'here': None, 'share35': None, 'speed': None, 've': None, 'vn': None,
           'raining': False, 'motion_status': 'unavailable', 'max_age_minutes': max_age_minutes}
    for key in ('lead_km', 'lead_rain_km', 'end35_km', 'end20_km',
                'lead_t', 'lead_rain_t', 'end35_t', 'end20_t'):
        res[key] = None
    if not fr:
        res.update(error='雷达帧列表为空', fresh=False, motion_reason='无雷达资料，无法估计移动')
        return res
    t, dbz, east, north = max(fr, key=lambda item: sz_radar.as_bjt(item[0]))[:4]
    t = sz_radar.as_bjt(t)
    res.update(time=t, **sz_radar.freshness(t, now, max_age_minutes))
    if not res['fresh']:
        res['error'] = res['freshness_reason']
        return res
    dbz, east, north = np.asarray(dbz), np.asarray(east), np.asarray(north)
    if dbz.ndim != 2 or not dbz.size or east.shape != dbz.shape or north.shape != dbz.shape:
        res['error'] = '雷达帧或坐标网格为空/不一致'
        return res
    dist = np.hypot(east, north)
    if not np.isfinite(dbz).all() or not np.isfinite(dist).all() or not (dist <= 3).any():
        res['error'] = '雷达帧或坐标网格异常/不覆盖市民中心'
        return res
    here = int(dbz[dist <= 3].max())
    share35 = float(((dbz >= 35) & (dist <= 10)).sum() / (dist <= 10).sum())
    move = motion(fr, return_details=True)
    res.update(here=here, share35=share35, raining=here >= 30 or share35 >= 0.2,
               motion_status=move['status'], motion_reason=move['reason'],
               motion_correlation=move.get('correlation'), motion_peak_margin=move.get('peak_margin'))
    if move['status'] != 'ok':
        return res
    ve, vn = move['ve'], move['vn']
    speed = float(np.hypot(ve, vn))
    res.update(speed=speed, ve=ve, vn=vn)
    if speed < 8:
        res['slow'] = True
        return res
    prof = profile(dbz, east, north, ve, vn)
    res['prof'] = prof
    lead = first(prof, lambda p: p >= 35)
    lead_any = first(prof, lambda p: p >= 25)
    res['lead_km'] = None if lead is None else lead * 2
    res['lead_rain_km'] = None if lead_any is None else lead_any * 2
    s = lead if lead is not None else 0
    res['end35_km'] = trailing(prof, 35, s) if lead is not None else None
    res['end20_km'] = trailing(prof, 20, lead_any if lead_any is not None else 0) if lead_any is not None else None
    hrs = lambda km: None if km is None else km / speed
    for key in ('lead_km', 'lead_rain_km', 'end35_km', 'end20_km'):
        res[key.replace('_km', '_t')] = None if res[key] is None else t + timedelta(hours=hrs(res[key]))
    return res


def q15(dt, up):
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (dt - midnight).total_seconds()
    steps = int(seconds // 900) + (1 if up and seconds % 900 else 0)
    return midnight + timedelta(minutes=steps * 15)


def round_half_hour(dt):
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + timedelta(minutes=round((dt - midnight).total_seconds() / 1800) * 30)


def hm(dt, reference=None):
    prefix = ''
    if reference is not None:
        day = (dt.date() - reference.date()).days
        prefix = '次日' if day == 1 else (f'{dt.month}月{dt.day}日' if day else '')
    return prefix + f'{dt.hour}时' + (f'{dt.minute}分' if dt.minute else '')


def text(res, now):
    """Sentence about arrival / how long it lasts."""
    now = sz_radar.as_bjt(now)
    reason = res.get('error')
    if not reason and res.get('time') is not None:
        reason = sz_radar.freshness(res['time'], now, res.get('max_age_minutes', sz_radar.MAX_FRAME_AGE_MINUTES))['freshness_reason']
    if reason or res.get('time') is None:
        return f'雷达资料不可用（{reason or "缺少帧时次"}），降雨到达及结束时刻无法判断，请人工核对'
    if res.get('motion_status') == 'unavailable' or res.get('speed') is None:
        observed = '雷达显示市民中心区域有降雨回波，' if res.get('raining') else ''
        return observed + '回波移动情况无法可靠判断，降雨到达及结束时刻请人工核对'
    if res.get('slow'):
        if res['raining']:
            return '目前雷雨正在影响市民中心区域，雨带移动缓慢，降雨可能持续较长时间'
        return '市民中心周边回波移动缓慢，未来1-2小时仍可能出现分散雷阵雨'
    parts = []
    lead_t, lead_rain_t = res.get('lead_t'), res.get('lead_rain_t')
    if not res['raining'] and any(t is not None and t <= now for t in (lead_t, lead_rain_t)):
        return '雷达外推的降雨到达时刻已过，实际影响尚无法确认，请核对最新雷达'
    if res['raining']:
        parts.append('目前雷雨带正在影响市民中心区域')
    elif lead_t is not None and timedelta(0) < lead_t - now <= timedelta(hours=3):
        if lead_t - now <= timedelta(minutes=20):
            parts.append('较强雷雨带即将影响市民中心区域，预计未来20分钟内开始')
        else:
            a, b = q15(lead_t - timedelta(minutes=10), False), q15(lead_t + timedelta(minutes=10), True)
            parts.append(f'较强雷雨带正向市民中心靠近，预计{hm(a, now)}至{hm(b, now)}前后开始影响市民中心区域')
    elif lead_rain_t is not None and timedelta(0) < lead_rain_t - now <= timedelta(hours=2):
        parts.append('市民中心上游有分散降雨回波，未来2小时可能出现短时阵雨')
        return '，'.join(parts)
    else:
        return '市民中心上游暂无明显降雨回波，未来2小时降雨可能性较小'
    e35, e20 = res.get('end35_t'), res.get('end20_t')
    if res['raining'] or lead_t is not None:
        if res.get('lead_km') is None and res['here'] < 35:
            parts.append('目前以中小雨为主，强降雨已基本结束')
        elif e35 is None:
            parts.append('上游仍有大片较强回波，强降雨还将持续1小时以上')
        elif e35 <= now:
            parts.append('强降雨外推结束时刻已过，请核对实际雨势')
        elif e35 - now <= timedelta(minutes=15):
            parts.append('强降雨将在短时间内减弱')
        else:
            dur = e35 - max(now, lead_t or now)
            a = q15(e35 - timedelta(minutes=5), False)
            b = q15(e35 + max(timedelta(minutes=15), dur * 0.3), True)
            parts.append(f'预计强降雨持续至{hm(a, now)}至{hm(b, now)}前后' if a != b else f'预计强降雨持续至{hm(a, now)}前后')
        if e20 is not None and e20 <= now:
            parts.append('降雨外推结束时刻已过，雨停情况请核对最新资料')
        elif e20 is not None and (e35 is None or e20 > e35):
            end = round_half_hour(e20 + timedelta(minutes=15))
            parts.append(f'之后雨势逐渐减弱，{hm(end, now)}前后降雨基本结束')
        elif e20 is None:
            parts.append('之后仍有降雨，雨停时间请关注后续研判')
    return '，'.join(parts)


if __name__ == '__main__':
    r = estimate()
    print({k: (round(v, 2) if isinstance(v, float) else (v.strftime('%H:%M') if isinstance(v, datetime) else v))
           for k, v in r.items() if k != 'prof'})
    print('prof', r.get('prof'))
    print(text(r, datetime.now(BJT)))
