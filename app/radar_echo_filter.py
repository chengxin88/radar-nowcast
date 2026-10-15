"""Conservative public echo selection; original monitoring and alerts stay raw.

This is a presentation policy, not a false-echo classifier. It does not wait
for an earlier frame, re-estimate motion, or turn removed pixels into evidence
of no future impact. The same sampling/corridor functions receive selected
pixels only in the returned public view.
"""
import json
import math

import numpy as np

import radar_monitor
import radar_nowcast as radar
import rolling_radar


POLICY = {'connectivity': 8, 'rain_threshold_dbz': 25., 'minimum_area_km2': 5.,
          'core_threshold_dbz': 35., 'core_support_pixels': 3,
          'raw_threshold_dbz': 20., 'temporal_confirmation_required': False}
_METADATA = ('product', 'generated_at', 'issue', 'event_date', 'fresh',
             'publication_allowed', 'radar_time', 'target', 'horizon_minutes',
             'step_minutes', 'window_start', 'window_end', 'thresholds',
             'motion', 'checks', 'blocking_checks', 'algorithm_version',
             'observed_clear')


def _copy_metadata(result):
    view = {}
    for key in _METADATA:
        if key in result:
            try:
                view[key] = json.loads(json.dumps(result[key], allow_nan=False))
            except (TypeError, ValueError):
                pass
    return view


def _components(mask):
    height, width = mask.shape
    flat, visited = mask.ravel(), np.zeros(mask.size, dtype=bool)
    for seed in np.flatnonzero(flat):
        seed = int(seed)
        if visited[seed]:
            continue
        visited[seed] = True
        pending, indexes = [seed], []
        while pending:
            index = pending.pop()
            indexes.append(index)
            row, col = divmod(index, width)
            for nr in range(max(0, row-1), min(height, row+2)):
                for nc in range(max(0, col-1), min(width, col+2)):
                    adjacent = nr * width + nc
                    if flat[adjacent] and not visited[adjacent]:
                        visited[adjacent] = True
                        pending.append(adjacent)
        yield indexes


def _has_core(indexes, field, width, height):
    remaining = {index for index in indexes if field.flat[index] >= POLICY['core_threshold_dbz']}
    while remaining:
        pending, count = [remaining.pop()], 0
        while pending:
            index = pending.pop()
            count += 1
            if count >= POLICY['core_support_pixels']:
                return True
            row, col = divmod(index, width)
            for nr in range(max(0, row-1), min(height, row+2)):
                for nc in range(max(0, col-1), min(width, col+2)):
                    adjacent = nr * width + nc
                    if adjacent in remaining:
                        remaining.remove(adjacent)
                        pending.append(adjacent)
    return False


def _unknown_steps(result, reason):
    return [dict(minute=step.get('minute'), time=step.get('time'), **radar.unknown_sample(reason))
            for step in result.get('steps', []) if isinstance(step, dict)]


def _disable_motion(view, reason):
    view['motion'] = {'status': 'unavailable', 'speed_kmh': None, 'east_kmh': None,
                      'north_kmh': None, 'reason': reason}


def _known_slow_motion(vector, minimum_speed):
    """Preserve verified slow motion for the existing corridor's slow state."""
    try:
        if vector.get('status') != 'ok':
            return False
        speed, east, north = (float(vector[key]) for key in ('speed_kmh', 'east_kmh', 'north_kmh'))
        return (all(math.isfinite(value) for value in (speed, east, north))
                and 0 <= speed < minimum_speed
                and math.isclose(speed, math.hypot(east, north), rel_tol=1e-6, abs_tol=1e-6))
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def _surface_rain(result):
    """Use the already bound official station, with its own recent observation."""
    try:
        official = result['official']
        if not official.get('station_observation_requested'):
            return None
        rain = official['station_observation']['rain']
        when, now = radar.as_bjt(rain['time']), radar.as_bjt(result['generated_at'])
        if not 0 <= (now-when).total_seconds() <= 20 * 60:
            return None
        if not any(isinstance(rain.get(key), (int, float))
                   and math.isfinite(rain[key]) and rain[key] > 0
                   for key in ('rain_5m', 'rain_10m')):
            return None
        return {'status': 'ongoing', 'arrival_minutes': None, 'current_kind': 'surface_rain',
                'basis': 'station_rain_observation', 'station_name': rain['station_name'],
                'observation_time': when.isoformat(), 'reason': '站点近期有实测降水，背景回波筛选不删除地面实况'}
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def apply(frames, result, config):
    """Return JSON-only (summary, public_view), leaving all input data untouched."""
    summary = {
        'status': 'unavailable', 'complete': False, 'observed_at': None,
        'radius_km': 150., 'nearest': None, 'raw_pixels': None,
        'kept_pixels': None, 'filtered_pixels': None,
        'boundary_truncated': False, 'outside_incoming': False, 'policy': dict(POLICY),
        'reason': '公开回波筛选资料不可用',
    }
    view = _copy_metadata(result)
    view.update(current=radar.unknown_sample(summary['reason']),
                steps=_unknown_steps(result, summary['reason']), forecast_reliable=False,
                clear=False, rolling_radar={'status': 'unavailable', 'timing_reliable': False})
    try:
        config.validate()
        if not radar._public_valid(result):
            raise ValueError('雷达资料未通过当前时效或覆盖校验')
        if any(float(result['target'][key]) != float(getattr(config, key))
               for key in ('lat', 'lon', 'radius_km')):
            raise ValueError('公开筛选目标与原始雷达目标不一致')
        if not frames:
            raise ValueError('雷达帧为空')
        latest = max(frames, key=lambda frame: radar.as_bjt(frame[0]))
        when, field, east, north = latest
        when = radar.as_bjt(when)
        summary['observed_at'] = when.isoformat()
        if when != radar.as_bjt(result['radar_time']):
            raise ValueError('公开筛选帧时次与原始雷达结果不一致')
        age = (radar.as_bjt(result['generated_at'])-when).total_seconds() / 60
        if not 0 <= age <= config.max_age_minutes:
            raise ValueError('公开筛选帧过期或晚于生成时间')
        field, east, north = np.asarray(field), np.asarray(east), np.asarray(north)
        x, y = radar.grid_axes(field, east, north)
        pixel_area = float(abs((x[1]-x[0]) * (y[1]-y[0])))
        kept = np.zeros(field.shape, dtype=bool)
        height, width = field.shape
        for indexes in _components(field >= POLICY['rain_threshold_dbz']):
            edge = any(index // width in (0, height-1) or index % width in (0, width-1)
                       for index in indexes)
            if (edge or len(indexes) * pixel_area >= POLICY['minimum_area_km2']
                    or _has_core(indexes, field, width, height)):
                kept.flat[indexes] = True
                summary['boundary_truncated'] |= edge
        public_field = np.where(kept, field, 0.)
        cx, cy = radar.target_center(config)
        distances = np.hypot(east-cx, north-cy)
        circle = distances <= summary['radius_km'] + 1e-9
        complete = (cx-150 >= x.min() and cx+150 <= x.max()
                    and cy-150 >= y.min() and cy+150 <= y.max())
        summary['complete'] = bool(complete)
        if complete:
            raw = circle & (field >= POLICY['raw_threshold_dbz'])
            public = circle & kept
            raw_count, kept_count = int(raw.sum()), int(public.sum())
            summary.update(raw_pixels=raw_count, kept_pixels=kept_count,
                           filtered_pixels=raw_count-kept_count, reason=None)
            if kept_count:
                index = int(np.where(public, distances, np.inf).argmin())
                summary.update(status='echo', nearest={
                    'distance_km': float(distances.flat[index]),
                    'bearing': radar.sz_radar.bearing(east.flat[index]-cx, north.flat[index]-cy)})
            elif raw_count:
                summary.update(status='filtered_only', reason='仅有低于公开显著回波门槛的弱小片段，未判定其真伪')
            elif summary['boundary_truncated']:
                summary.update(status='unavailable', reason='实际网格边界存在截断回波，不据此作清空判断')
            else:
                summary['status'] = 'clear'
        else:
            summary['reason'] = '150公里监测圆域覆盖不完整，不能确认完整背景分布'

        raw_outlook = radar_monitor.outlook(result)
        slow_motion = _known_slow_motion(result.get('motion') or {}, config.min_speed_kmh)
        motion_valid = raw_outlook['motion']['status'] == 'estimated' and not slow_motion
        if not motion_valid and not slow_motion:
            _disable_motion(view, raw_outlook['motion']['reason'])
        current = radar.sample_target(public_field, east, north, (cx, cy), config, (x, y))
        current.update(observed_at=when.isoformat(), age_minutes=age)
        view['current'] = current
        steps = []
        for original in result.get('steps', []):
            if not isinstance(original, dict):
                continue
            sample = radar.unknown_sample('原始路径覆盖或回波整体移动依据不足')
            minute = original.get('minute')
            if (motion_valid and original.get('coverage') is True
                    and isinstance(minute, (int, float)) and not isinstance(minute, bool)
                    and math.isfinite(minute) and 0 <= minute <= 120):
                hours = (age+minute) / 60
                center = (cx-view['motion']['east_kmh']*hours, cy-view['motion']['north_kmh']*hours)
                sample = radar.sample_target(public_field, east, north, center, config, (x, y))
            steps.append(dict(minute=minute, time=original.get('time'), **sample))
        view['steps'] = steps
        view['forecast_reliable'] = bool(result.get('forecast_reliable')
                                         and len(steps) == 25
                                         and all(step['status'] != 'unknown' for step in steps))
        view['clear'] = bool(result.get('clear') and view['forecast_reliable']
                             and all(step['status'] == 'clear' for step in steps))
        supported_path_echo = any(
            step.get('coverage') is True and step.get('status') in ('rain', 'strong')
            and (step.get('rain') is True or step.get('strong') is True) for step in steps)
        # The crop is a rectangle: significant echoes outside the 150-km
        # reporting circle can still enter the target within two hours. Do not
        # let quiet/filtered background inside that circle suppress this risk.
        summary['outside_incoming'] = bool(
            summary['complete'] and summary['kept_pixels'] == 0 and supported_path_echo)
        if summary['status'] == 'filtered_only' and not supported_path_echo:
            _disable_motion(view, '仅有未纳入公开描述的弱小回波，不发布其运动或起止判断')
            view['steps'] = _unknown_steps(result, '弱小回波筛选不能作为影响判断依据')
            view['forecast_reliable'] = False
            view['clear'] = False
        view['rolling_radar'] = rolling_radar.estimate([(when, public_field, east, north)], view, config)
        view['radar_outlook'] = radar_monitor.outlook(view)
        if (view['radar_outlook']['impact']['status'] == 'not_expected'
                and (raw_outlook['impact']['status'] != 'not_expected'
                     or summary['boundary_truncated'])):
            view['radar_outlook']['impact'].update(
                status='unknown', arrival_minutes=None, basis=None,
                reason='弱小回波筛选或边界截断不能新增未来无影响结论')
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OverflowError) as error:
        summary.update(status='unavailable', complete=False, reason=str(error))
        _disable_motion(view, str(error))
        view['current'] = radar.unknown_sample(str(error))
        view['steps'] = _unknown_steps(result, str(error))
        view['forecast_reliable'] = view['clear'] = False
        view['rolling_radar'] = {'status': 'unavailable', 'timing_reliable': False, 'reason': str(error)}
        view['radar_outlook'] = radar_monitor.outlook(view)
    surface = _surface_rain(result)
    if surface:
        view['radar_outlook']['impact'] = surface
    view['radar_monitor'] = summary
    return summary, view
