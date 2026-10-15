"""Target-centred upstream corridor used by the original rolling briefing.

The old 35/20 dBZ leading/trailing-edge method is retained. Missing coverage,
latency, a later band, and untrackable echoes cannot become an invented end time.
No conversion from PNG reflectivity to surface rain rate is made.
"""
import math
from datetime import timedelta

import numpy as np

import radar_nowcast as radar


def trailing(profile, threshold):
    """End of the FINAL band, followed by three observed 2-km clear bins."""
    wet = [i for i, value in enumerate(profile) if value is not None and value >= threshold]
    if not wet or any(value is None for value in profile):
        return None
    start = wet[-1] + 1
    return start * 2 if start + 2 < len(profile) and all(value < threshold for value in profile[start:start + 3]) else None


def estimate(frames, result, config):
    out = dict(status='unavailable', reason='雷达资料不可用', profile=[], complete=False,
               current_rain=None, current_strong=None, current_weak=None,
               arrival_minutes=None, strong_arrival_minutes=None,
               strong_end_minutes=None, end_minutes=None, timing_reliable=False,
               width_km=max(5., config.radius_km), reach_km=90., step_km=2.,
               rain_front_dbz=25, strong_dbz=35, rain_tail_dbz=20)
    if not radar._public_valid(result) or not frames:
        return out
    when, field, east, north = max(frames, key=lambda f: radar.as_bjt(f[0]))
    now = radar.as_bjt(result['generated_at'])
    age = (now - radar.as_bjt(when)).total_seconds() / 60
    x, y = radar.grid_axes(field, east, north)
    cx, cy = radar.target_center(config)
    e, n = east - cx, north - cy
    dist = np.hypot(e, n)
    here = dist <= config.radius_km
    near_radius = max(10., config.radius_km)
    near = dist <= near_radius
    near_covered = (cx - near_radius >= x.min() and cx + near_radius <= x.max()
                    and cy - near_radius >= y.min() and cy + near_radius <= y.max())
    counts = {threshold: int(((field >= threshold) & here).sum()) for threshold in (20, 30, 35)}
    share = float(((field >= 35) & near).sum() / near.sum()) if near.any() and near_covered else None
    local30, local35 = counts[30] >= config.support_pixels, counts[35] >= config.support_pixels
    nearby35 = share is not None and share >= .2
    out.update(here_max=float(field[here].max()), share35=share,
               current_rain=local30 or nearby35,
               current_strong=local35, nearby_strong=nearby35,
               current_weak=counts[20] >= config.support_pixels and not (local35 or nearby35),
               nearby_radius_km=near_radius,
               status='observation', reason=None)
    move = result['motion']
    speed = move.get('speed_kmh')
    if move.get('status') != 'ok' or speed is None:
        out['reason'] = ('目前未见可供追踪的明显降水回波，暂不能估计降水起止时间'
                         if result.get('observed_clear') and not (out['current_weak'] or out['current_rain'])
                         else '回波移动尚不能可靠追踪，暂不能估计降水起止时间')
        return out
    if speed < config.min_speed_kmh:
        out.update(status='slow', reason='雨带移动缓慢，降水起止时间暂不能可靠估计')
        return out
    ue, un = move['east_kmh'] / speed, move['north_kmh'] / speed
    along, across = -(e * ue + n * un), np.abs(-e * un + n * ue)
    profile = []
    width = out['width_km']
    # Check every complete physical cell, not only its surviving image pixels.
    for start in range(0, 90, 2):
        corners = [(cx - a * ue - b * un, cy - a * un + b * ue)
                   for a in (start, start + 2) for b in (-width, width)]
        covered = all(x.min() <= px <= x.max() and y.min() <= py <= y.max() for px, py in corners)
        mask = (along >= start) & (along < start + 2) & (across <= width)
        values = field[mask]
        # Third strongest pixel rejects isolated colour noise without converting
        # absent echo to absent coverage. Default is three spatial pixels.
        value = float(np.sort(values)[-config.support_pixels]) if covered and len(values) >= config.support_pixels else None
        profile.append(value)
    out.update(profile=profile, complete=all(v is not None for v in profile), speed_kmh=speed,
               direction_from=radar.sz_radar.bearing(-move['east_kmh'], -move['north_kmh']))
    if not out['complete']:
        out.update(reason='上游走廊覆盖不完整，不能据此确定完整的降水起止时间')
        return out
    # Radius reaches the approaching edge sooner and leaves the trailing edge
    # later. All clocks are based on the observation time, then corrected for age.
    first = lambda threshold: next((i * 2 for i, v in enumerate(profile) if v >= threshold), None)
    for key, threshold in (('arrival_minutes', 25), ('strong_arrival_minutes', 35)):
        distance = first(threshold)
        raw = None if distance is None else max(0., distance - config.radius_km) / speed * 60 - age
        out[key + '_raw'] = raw
        if raw is not None and 0 <= raw <= 120:
            out[key] = raw
        elif raw is not None and raw < 0:
            out['arrival_elapsed'] = True
    for key, threshold in (('strong_end_minutes', 35), ('end_minutes', 20)):
        distance = trailing(profile, threshold)
        # A band over the target whose upstream profile is already clear ends
        # only after the target's downstream edge also clears.
        active = out['current_strong'] if threshold == 35 else out['current_rain'] or out['current_weak']
        if distance is None and active and all(v < threshold for v in profile[:3]) and not any(v >= threshold for v in profile):
            distance = 0
        raw = None if distance is None else (distance + config.radius_km) / speed * 60 - age
        out[key + '_raw'] = raw
        if raw is not None and 0 <= raw <= 120:
            out[key] = raw
    out.update(status='tracked', timing_reliable=True, reason=None,
               corridor_max=max(profile), upstream_echo=any(v >= 20 for v in profile),
               upstream_strong=any(v >= 35 for v in profile))
    return out


def clock_window(now, minute, pad=10, unit=15):
    t = now + timedelta(minutes=minute)
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    lower = midnight + timedelta(minutes=math.floor(((t-midnight).total_seconds()/60-pad)/unit)*unit)
    upper = midnight + timedelta(minutes=math.ceil(((t-midnight).total_seconds()/60+pad)/unit)*unit)
    lower = max(now.replace(second=0, microsecond=0), lower)
    return radar.time_label(lower, now) + '—' + radar.time_label(upper, now)


def describe(rolling, result):
    now = radar.as_bjt(result['generated_at'])
    r = rolling
    if r['status'] == 'unavailable':
        return ['雷达资料尚待核实，降水到达、强降水结束和雨停时间暂不能确定。']
    if not r['timing_reliable']:
        text = r['reason'] + '。'
        if r['status'] == 'slow' and (r['current_rain'] or r['current_weak']):
            text += '降水回波可能继续影响，请关注后续变化。'
        return [text]
    parts = []
    wet = r['current_rain'] or r['current_weak']
    if r['current_strong']:
        parts.append('最新雷达显示较强回波已影响目标区域。')
    elif r['current_weak']:
        parts.append('最新雷达显示以较弱降水回波为主。')
    elif r['current_rain']:
        parts.append('最新雷达显示降水回波已影响目标区域附近。')
    if not wet:
        arrival = r['arrival_minutes']
        if arrival is not None:
            parts.append(f'上游雨带正从{r["direction_from"]}方向移近，预计{clock_window(now, arrival)}开始影响。')
        elif r.get('arrival_elapsed'):
            parts.append('外推的雨带到达时刻已过，是否开始降水需用更新雷达和现场情况确认。')
    if r['strong_arrival_minutes'] is not None and not r['current_strong']:
        parts.append(f'较强回波预计{clock_window(now, r["strong_arrival_minutes"])}开始影响，需关注雨势增强。')
    if r['strong_end_minutes'] is not None:
        parts.append(f'较强回波预计{clock_window(now, r["strong_end_minutes"])}移出，强降水可能随之减弱。')
    elif r.get('upstream_strong') or r['current_strong']:
        parts.append('上游较强回波尾部尚不足以确定未来两小时内的强降水结束时间。')
    if r['end_minutes'] is not None:
        parts.append(f'降水回波预计{clock_window(now, r["end_minutes"], pad=15, unit=30)}移出，雨停时间以现场实况为准。')
    elif wet or r.get('upstream_echo'):
        parts.append('雨停时间暂不能确定，需继续跟踪后续雷达。')
    if not wet and not r.get('upstream_echo'):
        parts.append('当前上游走廊未见明显降水回波向目标区域移近，仍需关注新生回波。')
    return parts
