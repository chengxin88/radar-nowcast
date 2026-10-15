"""Read-only radar observation around a target, separate from arrival timing.

Pixel support follows the existing project convention: count all echo pixels
inside the complete monitoring circle. A nearest pixel is a location reference,
not an independently confirmed rain cell or a prediction of target impact.
"""
import math

import numpy as np

import radar_nowcast as radar


def estimate(frames, result, config, radius_km=150.):
    """Inspect only the latest matching frame; never modify inputs or forecasts."""
    out = {
        'radius_km': None, 'observed_at': None, 'status': 'unavailable',
        'complete': False, 'threshold_dbz': 20.,
        'support_pixels': config.support_pixels, 'echo_pixels': None,
        'nearest': None, 'reason': '雷达监测资料不可用',
        'support_scope': 'monitor_circle_total_pixels',
    }
    try:
        config.validate()
        if (isinstance(config.support_pixels, bool)
                or not isinstance(config.support_pixels, (int, np.integer))):
            raise ValueError('监测像元支持门槛须为正整数')
        out['support_pixels'] = int(config.support_pixels)
        radius = float(radius_km)
        if isinstance(radius_km, bool) or not math.isfinite(radius) or radius <= 0:
            raise ValueError('监测半径须为有限正数')
        out['radius_km'] = radius
        if not radar._public_valid(result):
            raise ValueError('雷达资料未通过当前时效或覆盖校验')
        target = result.get('target') or {}
        if any(float(target[key]) != float(getattr(config, key))
               for key in ('lat', 'lon', 'radius_km')):
            raise ValueError('监测目标与雷达结果目标不一致')
        if not frames:
            raise ValueError('雷达监测帧为空')
        latest = max(frames, key=lambda frame: radar.as_bjt(frame[0]))
        when, field, east, north = latest
        when = radar.as_bjt(when)
        out['observed_at'] = when.isoformat()
        if when != radar.as_bjt(result['radar_time']):
            raise ValueError('监测最新帧时次与雷达结果不一致')
        age = (radar.as_bjt(result['generated_at']) - when).total_seconds() / 60
        if not 0 <= age <= config.max_age_minutes:
            raise ValueError('监测雷达帧过期或晚于生成时间')
        field, east, north = np.asarray(field), np.asarray(east), np.asarray(north)
        x, y = radar.grid_axes(field, east, north)
        cx, cy = radar.target_center(config)
        # Require the full physical circle, not only surviving image pixels.
        if (cx - radius < x.min() or cx + radius > x.max()
                or cy - radius < y.min() or cy + radius > y.max()):
            raise ValueError('目标周边监测圆域超出雷达真实网格覆盖')
        relative_east, relative_north = east - cx, north - cy
        distances = np.hypot(relative_east, relative_north)
        inside = distances <= radius + 1e-9
        if int(inside.sum()) < out['support_pixels']:
            raise ValueError('监测圆域内有效像元不足')
        echoes = inside & (field >= out['threshold_dbz'])
        count = int(echoes.sum())
        out.update(complete=True, echo_pixels=count, reason=None)
        if count == 0:
            out['status'] = 'clear'
        elif count < out['support_pixels']:
            out.update(status='sparse', reason='监测范围内有零散回波，像元支持不足，不能确认无回波')
        else:
            index = int(np.where(echoes, distances, np.inf).argmin())
            out.update(status='echo', nearest={
                'distance_km': float(distances.flat[index]),
                'bearing': radar.sz_radar.bearing(relative_east.flat[index], relative_north.flat[index]),
            })
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OverflowError) as error:
        out.update(status='unavailable', complete=False, echo_pixels=None,
                   nearest=None, reason=str(error))
    return out


def outlook(result):
    """Summarize existing motion and target-impact evidence without re-tracking.

    The vector describes the existing echo field as a whole. It does not prove
    that the monitoring circle's nearest pixel has independently been tracked.
    ``not_expected`` is conditional on a complete reliable model path; neither
    an empty monitoring circle nor a distant/receding nearest pixel establishes
    that outcome. The 180-km/h limit is a conservative presentation quality gate,
    not a physical bound on storm motion.
    """
    motion = {
        'status': 'unknown', 'direction_to': None, 'speed_kmh': None,
        'basis': 'existing_echo_field_translation', 'reason': '回波整体移动趋势尚不能可靠判断',
    }
    impact = {
        'status': 'unknown', 'arrival_minutes': None, 'current_kind': None,
        'basis': None, 'reason': '未来两小时是否影响目标尚不能可靠判断',
    }
    out = {'motion': motion, 'impact': impact}
    try:
        if not radar._public_valid(result):
            raise ValueError('雷达资料未通过当前时效或覆盖校验')
        target = result['target']
        if not all(math.isfinite(float(target[key])) for key in ('lat', 'lon', 'radius_km')):
            raise ValueError('雷达目标坐标或范围缺失')
        if radar.as_bjt(result['radar_time']) > radar.as_bjt(result['generated_at']):
            raise ValueError('雷达观测时次晚于生成时间')
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as error:
        motion['reason'] = impact['reason'] = str(error)
        return out

    vector = result.get('motion') or {}
    try:
        if vector.get('status') != 'ok':
            raise ValueError('回波整体移动尚未通过追踪校验')
        speed, east, north = (float(vector[key]) for key in ('speed_kmh', 'east_kmh', 'north_kmh'))
        if not all(math.isfinite(value) for value in (speed, east, north)):
            raise ValueError('回波整体移动速度或矢量无效')
        if not 8. <= speed <= 180.:
            raise ValueError('回波整体移动过慢或速度超出展示质控范围')
        if not math.isclose(speed, math.hypot(east, north), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('回波整体移动速度与矢量不一致')
        motion.update(status='estimated', direction_to=radar.sz_radar.bearing(east, north),
                      speed_kmh=speed, reason=None)
    except (KeyError, ValueError, TypeError, OverflowError) as error:
        motion['reason'] = str(error)

    current = result['current']
    rolling = result.get('rolling_radar') or {}
    if current.get('status') in ('rain', 'strong'):
        impact.update(status='ongoing', arrival_minutes=0.,
                      current_kind=current['status'], basis='current_observation',
                      reason='目标区域当前已有降水回波影响')
        return out
    if rolling.get('current_weak'):
        impact.update(status='ongoing', arrival_minutes=0., current_kind='weak',
                      basis='current_observation', reason='目标区域当前有较弱回波，地面降水需结合实况核对')
        return out
    if rolling.get('current_strong'):
        impact.update(status='ongoing', arrival_minutes=0., current_kind='strong',
                      basis='current_observation', reason='目标区域当前已有较强回波影响')
        return out

    def minute_of(step):
        try:
            value = step['minute']
            minute = float(value)
            return minute if not isinstance(value, bool) and math.isfinite(minute) and 0 <= minute <= 120 else None
        except (KeyError, ValueError, TypeError, OverflowError):
            return None

    steps = result.get('steps') or []
    wet_minutes = []
    for step in steps:
        if not isinstance(step, dict) or step.get('coverage') is not True:
            continue
        minute = minute_of(step)
        if (minute is not None and step.get('status') in ('rain', 'strong')
                and (step.get('rain') is True or step.get('strong') is True)):
            wet_minutes.append(minute)
    if wet_minutes:
        impact.update(status='possible', basis='target_path_extrapolation',
                      arrival_minutes=min(wet_minutes) if motion['status'] == 'estimated' else None,
                      reason='已覆盖的目标外推路径出现降水回波，后续覆盖不足不消除这一影响可能')
        return out

    corridor_arrivals = []
    if rolling.get('timing_reliable'):
        for key in ('arrival_minutes', 'strong_arrival_minutes'):
            minute = minute_of({'minute': rolling.get(key)})
            if minute is not None:
                corridor_arrivals.append(minute)
    if corridor_arrivals:
        impact.update(status='possible', basis='upstream_corridor',
                      arrival_minutes=min(corridor_arrivals) if motion['status'] == 'estimated' else None,
                      reason='现有上游剖面提示未来两小时可能有回波移近目标')
        return out
    if rolling.get('current_rain'):
        # This flag can come from strong echoes over the surrounding 10 km;
        # it alone does not establish rainfall inside the actual target circle.
        impact.update(status='possible', basis='nearby_observation',
                      reason='目标附近已有降水回波，是否影响目标仍需核对')
        return out

    all_clear = (len(steps) == 25 and all(isinstance(step, dict) for step in steps)
                 and sorted(minute_of(step) for step in steps if minute_of(step) is not None)
                 == list(range(0, 121, 5))
                 and all(step.get('coverage') is True and step.get('status') == 'clear'
                         and step.get('rain') is False and step.get('strong') is False for step in steps))
    if (motion['status'] == 'estimated' and result.get('forecast_reliable') is True
            and result.get('horizon_minutes') == 120 and result.get('step_minutes') == 5
            and current.get('status') == 'clear' and current.get('rain') is False
            and current.get('strong') is False and all_clear):
        impact.update(status='not_expected', basis='target_path_extrapolation',
                      reason='仅按现有回波整体平移估计，完整两小时目标路径未见明显回波影响；新生、增强或消散可能改变结论')
    elif motion['status'] != 'estimated':
        impact['reason'] = '回波整体移动趋势未通过校验，无法可靠判断未来两小时影响'
    else:
        impact['reason'] = '完整两小时外推覆盖或可靠性不足，无法排除回波影响'
    return out
