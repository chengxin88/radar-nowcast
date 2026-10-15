"""Radar-only 0–2 hour translation nowcast around a configured target.

No ECMWF retrieval, lightning inference, wind inference, calibrated rain-rate
claim, or message transport is used. Unknown coverage is never clear weather.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

import radar_track
import sz_radar

BJT = radar_track.BJT


class FrameBatch(list):
    def __init__(self, frames=(), checks=()):
        super().__init__(frames)
        self.checks = list(checks)


@dataclass(frozen=True)
class Config:
    lat: float = 22.543
    lon: float = 114.059
    radius_km: float = 3.0
    rain_threshold: float = 25.0
    strong_threshold: float = 35.0
    support_fraction: float = 0.10
    support_pixels: int = 3
    min_speed_kmh: float = 8.0
    max_age_minutes: float = 20.0
    horizon_minutes: int = 120
    step_minutes: int = 5
    name: str = '市民中心'

    def __post_init__(self):
        if isinstance(self.name, str) and not any(ord(char) < 32 or 127 <= ord(char) <= 159 or char in '\u2028\u2029' for char in self.name):
            object.__setattr__(self, 'name', self.name.strip())

    def validate(self):
        if not isinstance(self.name, str) or not self.name.strip() or any(ord(char) < 32 or 127 <= ord(char) <= 159 or char in '\u2028\u2029' for char in self.name):
            raise ValueError('目标地区名称必须非空、单行且不含控制字符')
        numeric = (self.lat, self.lon, self.radius_km, self.rain_threshold,
                   self.strong_threshold, self.support_fraction, self.min_speed_kmh, self.max_age_minutes)
        if not all(isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)) for value in numeric):
            raise ValueError('配置数值必须有限')
        if not -90 <= self.lat <= 90 or not -180 <= self.lon <= 180 or not 0 < self.radius_km <= 500:
            raise ValueError('目标经纬度或半径无效')
        if self.strong_threshold < self.rain_threshold or not 0 < self.support_fraction <= 1 or self.support_pixels < 1:
            raise ValueError('回波阈值或空间支持门槛无效')
        if self.min_speed_kmh <= 0 or self.max_age_minutes < 0 or self.horizon_minutes != 120 or self.step_minutes != 5:
            raise ValueError('本产品采用0–120分钟、5分钟步长，速度和时效门槛必须有效')


def as_bjt(value):
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return value.replace(tzinfo=BJT) if value.tzinfo is None else value.astimezone(BJT)


def unknown_sample(reason):
    return {'status': 'unknown', 'rain': None, 'strong': None, 'coverage': False,
            'rain_fraction': None, 'rain_pixels': None, 'strong_fraction': None,
            'strong_pixels': None, 'target_pixels': None, 'reason': reason}


def grid_axes(dbz, east, north):
    dbz, east, north = np.asarray(dbz), np.asarray(east), np.asarray(north)
    if dbz.ndim != 2 or min(dbz.shape, default=0) < 2 or east.shape != dbz.shape or north.shape != dbz.shape:
        raise ValueError('雷达图和坐标网格为空或尺寸不一致')
    if not np.isfinite(dbz).all() or not np.isfinite(east).all() or not np.isfinite(north).all():
        raise ValueError('雷达图或坐标包含缺失值')
    x, y = east[0, :], north[:, 0]
    dx, dy = np.diff(x), np.diff(y)
    if not np.allclose(east, x[None, :]) or not np.allclose(north, y[:, None]):
        raise ValueError('雷达坐标不是规则经纬度投影网格')
    if dx[0] <= 0 or dy[0] >= 0 or not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError('雷达网格有反向坐标轴、重复坐标或空间缺口')
    return x, y


def target_center(config):
    east = (config.lon - sz_radar.SITE_LON) * 111.32 * math.cos(math.radians(sz_radar.SITE_LAT))
    north = (config.lat - sz_radar.SITE_LAT) * 110.57
    return east, north


def sample_target(dbz, east, north, center, config, axes=None):
    x, y = axes or grid_axes(dbz, east, north)
    cx, cy = center
    radius = config.radius_km
    # Entire physical circle must be covered, not merely the available pixels.
    if cx - radius < min(x) or cx + radius > max(x) or cy - radius < min(y) or cy + radius > max(y):
        return unknown_sample('上游目标圆域超出雷达网格覆盖')
    mask = (east - cx) ** 2 + (north - cy) ** 2 <= radius ** 2 + 1e-9
    total = int(mask.sum())
    if total < config.support_pixels:
        return unknown_sample('目标圆域内有效像元不足')
    rain_pixels = int(((dbz >= config.rain_threshold) & mask).sum())
    strong_pixels = int(((dbz >= config.strong_threshold) & mask).sum())
    rain_fraction, strong_fraction = rain_pixels / total, strong_pixels / total
    rain = rain_pixels >= config.support_pixels and rain_fraction >= config.support_fraction
    strong = strong_pixels >= config.support_pixels and strong_fraction >= config.support_fraction
    uncertain = rain_pixels > 0 and not rain
    return {'status': 'unknown' if uncertain else 'strong' if strong else 'rain' if rain else 'clear',
            'rain': None if uncertain else rain, 'strong': strong,
            'coverage': True, 'rain_fraction': rain_fraction, 'rain_pixels': rain_pixels,
            'strong_fraction': strong_fraction, 'strong_pixels': strong_pixels, 'target_pixels': total,
            'reason': '有零散回波，但空间支持不足，不能确认降水影响或无雨' if uncertain else None}


def ending_minutes(steps, current_rain):
    """Confirm a process ending only with four clear samples spanning 15 min.

    Use the final wet episode, so a temporary gap or a later return cannot be
    presented as the end of all rain within the two-hour horizon.
    """
    wet = [i for i, step in enumerate(steps) if step['rain'] is True]
    if not wet and current_rain is not True:
        return None
    start = wet[-1] + 1 if wet else 0
    if start + 3 >= len(steps):
        return None
    confirm = steps[start:start + 4]
    if all(step['status'] == 'clear' for step in confirm) and confirm[-1]['minute'] - confirm[0]['minute'] >= 15:
        return confirm[0]['minute']
    return None


def analyse(frames, now=None, config=None):
    config = config or Config()
    config.validate()
    now = as_bjt(now or datetime.now(BJT))
    result = {'product': 'radar_nowcast', 'generated_at': now.isoformat(), 'issue': now.isoformat(),
              'event_date': now.strftime('%Y-%m-%d'), 'fresh': False, 'publication_allowed': False,
              'forecast_reliable': False, 'clear': False, 'observed_clear': False, 'radar_time': None,
              'target': {'name': config.name, 'lat': config.lat, 'lon': config.lon, 'radius_km': config.radius_km},
              'horizon_minutes': config.horizon_minutes, 'step_minutes': config.step_minutes,
              'window_start': now.isoformat(), 'window_end': (now + timedelta(minutes=120)).isoformat(),
              'thresholds': {'rain_dbz': config.rain_threshold, 'strong_dbz': config.strong_threshold,
                             'support_fraction': config.support_fraction, 'support_pixels': config.support_pixels},
              'arrival_minutes': None, 'strong_arrival_minutes': None, 'end_minutes': None,
              'motion': {'status': 'unavailable', 'speed_kmh': None, 'east_kmh': None, 'north_kmh': None},
              'current': unknown_sample('尚未获取有效最新雷达帧'), 'steps': [], 'checks': [], 'blocking_checks': []}
    result['checks'].extend(getattr(frames, 'checks', []))
    frames = list(frames or [])
    reason = None if sz_radar.WEST <= config.lon <= sz_radar.EAST and sz_radar.SOUTH <= config.lat <= sz_radar.NORTH else '目标经纬度超出当前雷达资料覆盖（雷达拼图范围），不能使用其他地区资料兜底'
    normalized = []
    try:
        for frame in frames:
            if len(frame) != 4:
                raise ValueError('雷达帧结构无效')
            normalized.append((as_bjt(frame[0]), *frame[1:]))
    except (ValueError, TypeError, IndexError):
        reason = '雷达帧结构或时间无效'
    if not normalized and reason is None:
        reason = '雷达帧列表为空'
    if normalized:
        latest = max(normalized, key=lambda frame: frame[0])
        when, dbz, east, north = latest
        age = (now - when).total_seconds() / 60
        result['radar_time'] = when.isoformat()
        result['current'].update(observed_at=when.isoformat(), age_minutes=age)
        if age < 0:
            reason = '最新雷达帧晚于实际生成时间，不能反向外推'
        elif age > config.max_age_minutes:
            reason = f'最新雷达帧已过期（延迟{age:.1f}分钟，允许{config.max_age_minutes:g}分钟）'
        elif reason is None:
            try:
                dbz, east, north = np.asarray(dbz), np.asarray(east), np.asarray(north)
                axes = grid_axes(dbz, east, north)
                current = sample_target(dbz, east, north, target_center(config), config, axes)
                current.update(observed_at=when.isoformat(), age_minutes=age)
                result.update(current=current, fresh=True)
                if not current['coverage']:
                    reason = current['reason']
                elif current['rain'] is False and current['strong'] is False:
                    cx, cy = target_center(config)
                    x, y = axes
                    nearby_covered = cx - 100 >= min(x) and cx + 100 <= max(x) and cy - 100 >= min(y) and cy + 100 <= max(y)
                    nearby = (east - cx) ** 2 + (north - cy) ** 2 <= 100 ** 2
                    nearby_rain_pixels = int(((dbz >= config.rain_threshold) & nearby).sum())
                    result['observed_clear'] = bool(nearby_covered and nearby_rain_pixels < 3)
            except (ValueError, TypeError, IndexError) as error:
                reason = str(error)
    if reason:
        result['blocking_checks'].append(reason)
    else:
        result['publication_allowed'] = True
        # Validate every frame and its common coordinate grid before correlation.
        move_reason = None
        if len(normalized) < 2:
            move_reason = '只有一帧雷达资料，无法估计移动'
        elif any(b[0] <= a[0] for a, b in zip(normalized, normalized[1:])):
            move_reason = '雷达帧时次重复或未递增'
        else:
            try:
                for _, previous, pe, pn in normalized:
                    grid_axes(previous, pe, pn)
                    if np.asarray(previous).shape != dbz.shape or not np.allclose(pe, east) or not np.allclose(pn, north):
                        raise ValueError('多帧雷达网格不一致')
                move = radar_track.motion(normalized, return_details=True)
                speed = math.hypot(move['ve'], move['vn']) if move['status'] == 'ok' else None
                if speed is None or not math.isfinite(speed):
                    move_reason = move.get('reason') or '回波移动无法可靠判断'
                else:
                    result['motion'].update(status='ok', speed_kmh=speed, east_kmh=move['ve'], north_kmh=move['vn'],
                                            correlation=move.get('correlation'), peak_margin=move.get('peak_margin'))
                    if speed < config.min_speed_kmh:
                        move_reason = f'回波移动缓慢（小于{config.min_speed_kmh:g}公里/小时），不给出可靠起止时刻'
            except (ValueError, TypeError, IndexError, ZeroDivisionError) as error:
                move_reason = '移动估计失败：' + str(error)
        if move_reason:
            result['checks'].append(move_reason)
        else:
            cx, cy = target_center(config)
            for lead in range(0, 121, 5):
                hours = (age + lead) / 60
                center = (cx - move['ve'] * hours, cy - move['vn'] * hours)
                sample = sample_target(dbz, east, north, center, config, axes)
                result['steps'].append({'minute': lead, 'time': (now + timedelta(minutes=lead)).isoformat(), **sample})
            result['forecast_reliable'] = all(step['status'] != 'unknown' for step in result['steps'])
            if not result['forecast_reliable']:
                result['checks'].append('部分外推路径覆盖不足或回波空间支持不足，不能给出可靠到达/结束时刻')
            else:
                result['arrival_minutes'] = next((step['minute'] for step in result['steps'] if step['rain']), None)
                result['strong_arrival_minutes'] = next((step['minute'] for step in result['steps'] if step['strong']), None)
                result['end_minutes'] = ending_minutes(result['steps'], result['current']['rain'])
                result['clear'] = result['current']['rain'] is False and all(step['status'] == 'clear' for step in result['steps'])
    if not result['steps']:
        message = reason or (result['checks'][0] if result['checks'] else '移动无法可靠判断')
        result['steps'] = [{'minute': lead, 'time': (now + timedelta(minutes=lead)).isoformat(), **unknown_sample(message)}
                           for lead in range(0, 121, 5)]
    if result['current'].get('reason') and result['current']['coverage']:
        result['checks'].append(result['current']['reason'])
    result['checks'].extend(result['blocking_checks'])
    result['checks'].append('仅按现有回波平移外推；回波发展、消散和新生会改变影响时间，须结合后续雷达核对')
    return refresh_presentation(result)


def time_label(moment, reference):
    moment, reference = as_bjt(moment), as_bjt(reference)
    offset = (moment.date() - reference.date()).days
    prefix = '次日' if offset == 1 else (f'{moment.month}月{moment.day}日' if offset else '')
    return prefix + moment.strftime('%H:%M')


def report(result):
    """Diagnostic report retained only as internal evidence."""
    now = as_bjt(result['generated_at'])
    target = result['target']
    parts = [f'【{target["name"]}雷达短临预报】',
             f'生成时间：{now:%m月%d日 %H:%M}；未来0–120分钟（{time_label(now, now)}—{time_label(result["window_end"], now)}）。']
    if result['radar_time']:
        radar_time = as_bjt(result['radar_time'])
        age = result['current'].get('age_minutes')
        parts.append(f'最新雷达帧：{radar_time:%m月%d日 %H:%M}，相对生成时间延迟{age:.1f}分钟。')
    else:
        parts.append('最新雷达帧：未获取。')
    status = result['current']['status']
    current_text = {'strong': '有较强回波', 'rain': '有降水回波',
                    'clear': f'未见达到{result["thresholds"]["rain_dbz"]:g}dBZ及空间支持门槛的降水回波',
                    'unknown': '资料或空间支持不足，无法确认降水影响'}[status]
    parts.append(f'最新帧观测：{target["name"]}周边{target["radius_km"]:g}公里{current_text}。')
    if result.get('observed_clear'):
        parts.append(f'最新观测的{target["name"]}100公里附近未见达到{result["thresholds"]["rain_dbz"]:g}dBZ、至少3像素支持的降水回波；这仅表示观测明确无附近回波，不表示未来两小时必然无雨。')
    for start, end in ((0, 30), (30, 60), (60, 120)):
        steps = [step for step in result['steps'] if start <= step['minute'] <= end]
        known = [step['status'] for step in steps if step['status'] != 'unknown']
        if not known:
            text = '无法可靠判断回波是否移入，请关注后续雷达'
        elif 'strong' in known:
            text = '按当前移动，可能有较强回波影响'
        elif 'rain' in known:
            text = '按当前移动，可能有降水回波影响'
        else:
            text = '按当前移动，暂未见达到降水门槛的回波移入'
        if len(known) < len(steps) and known:
            text += '；部分时段路径覆盖或空间支持不足，后续情况未知'
        parts.append(f'{start}–{end}分钟：{text}。')
    if result['forecast_reliable']:
        for key, name in (('arrival_minutes', '降水回波到达'), ('strong_arrival_minutes', '较强回波开始影响'), ('end_minutes', '本轮回波影响结束')):
            minute = result[key]
            if minute is not None:
                when = now + timedelta(minutes=minute)
                parts.append(f'平移外推{name}：{time_label(when, now)}前后（相对生成时间{minute}分钟，5分钟采样）。')
        if result['end_minutes'] is not None:
            parts.append('结束时刻以其后连续15分钟未见达到降水门槛的回波确认，仍需实况核对。')
    else:
        parts.append('到达、增强和结束时刻：当前不能可靠给出。')
    if result['checks']:
        parts.append('资料与不确定性：' + '；'.join(dict.fromkeys(result['checks'])) + '。')
    return '\n'.join(parts)


def _public_valid(result):
    return bool(result.get('fresh') and result.get('publication_allowed') and result.get('current', {}).get('coverage'))


def _public_current(result, now):
    name, radius = result['target']['name'], result['target']['radius_km']
    current = result['current']
    if not _public_valid(result):
        age = current.get('age_minutes')
        if age is not None and age < 0:
            return '雷达资料时次尚待核实，本次不作当前降水结论。'
        if result.get('radar_time') and not result.get('fresh'):
            return f'最新收到的雷达资料为{time_label(result["radar_time"], now)}，尚不足以代表当前情况，降水变化需核实。'
        return f'{name}目前尚无可用雷达资料，当前降水情况需核实。'
    prefix = f'截至{time_label(result["radar_time"], now)}，雷达显示{name}周边{radius:g}公里'
    text = {'strong': '有较强回波', 'rain': '有降水回波', 'clear': '暂无明显降水回波',
            'unknown': '有零散回波，是否形成降水影响仍需观察'}[current['status']]
    return prefix + text + '。'


def _trusted_public_forecast(result):
    return _public_valid(result) and bool(result.get('forecast_reliable'))


def _future_summary(result, now):
    if not _public_valid(result):
        return '雷达资料尚待核实，未来两小时的降水变化暂不能确定。'
    known = [step['status'] for step in result['steps'] if step['status'] != 'unknown']
    if not known:
        if any('移动缓慢' in text for text in result.get('checks', [])):
            return '回波移动缓慢，未来两小时降水的起止时间尚需继续观察。'
        return '回波移动趋势尚不明确，未来两小时降水变化需继续观察。'
    if not _trusted_public_forecast(result):
        affected = '已能研判的时段可能有较强回波影响' if 'strong' in known else '已能研判的时段可能有降水回波影响' if 'rain' in known else '暂未见明显降水回波移入已覆盖的时段'
        return affected + '，其余时段尚待核实，暂不能确定完整的降水起止时间。'
    timing = _timing_lines(result, now)
    if timing:
        return ''.join(timing)
    return '按现有回波移动，未来两小时暂未见明显降水回波移入，仍需关注回波新生和发展。'


def _timing_lines(result, now):
    if not _trusted_public_forecast(result):
        return []
    lines = []
    for key, noun in (('arrival_minutes', '降水回波到达'), ('strong_arrival_minutes', '较强回波开始影响')):
        minute = result.get(key)
        if minute is None:
            continue
        if minute == 0:
            text = '附近可能已有较强回波影响' if key == 'strong_arrival_minutes' else '附近可能已有降水回波影响'
            lines.append('按回波移动推算，出稿时刻' + text + '。')
        else:
            when = time_label(now + timedelta(minutes=minute), now)
            lines.append(f'预计{when}前后{noun}（实际出稿后{minute:g}分钟左右）。')
    end = result.get('end_minutes')
    if end is not None:
        if end == 0:
            lines.append('按回波移动推算，本轮回波可能已移出目标附近，实际降水是否结束仍需核实。')
        else:
            when = time_label(now + timedelta(minutes=end), now)
            lines.append(f'本轮回波影响预计{when}前后结束（实际出稿后{end:g}分钟左右），请结合最新雷达和现场情况核对。')
    return lines


def _human_checks(result):
    checks = []
    if not _public_valid(result):
        checks.append('该地区最新雷达资料的时次和覆盖是否可用')
    if not _trusted_public_forecast(result):
        checks.append('回波移动趋势及后续降水变化')
    if result['current']['status'] == 'unknown' and result['current'].get('coverage'):
        checks.append('附近零散回波是否继续发展')
    if result['current'].get('age_minutes', 0) and result['current'].get('age_minutes', 0) > 0:
        checks.append('最新雷达资料是否已更新')
    checks.extend(('最新雷达与现场降水是否一致', '回波新生、增强或消散是否改变影响时间'))
    return checks


def refresh_presentation(result):
    """Regenerate the complete notice/detail/brief vector after any data change."""
    now = as_bjt(result['generated_at'])
    name = result['target']['name']
    start, finish = time_label(now, now), time_label(result['window_end'], now)
    current = _public_current(result, now)
    future = _future_summary(result, now)
    uncertainty = '回波新生、发展或消散可能改变影响时间，请结合后续雷达和现场情况核对。'
    advice = '请持续关注最新雷达、当地气象台短临预报和现场降水情况；若遇降水，备好雨具，注意路面湿滑。'
    detail = [f'[{name}雷达短临预报]', f'[预报时效]{now:%m月%d日} {start}—{finish}；以实际出稿时刻为起点。',
              '[当前雷达]' + current, '[分时段预报]']
    all_unknown = not _public_valid(result) or all(step['status'] == 'unknown' for step in result['steps'])
    if all_unknown:
        detail.append(f'{start}—{finish}：' + future)
    else:
        for lower, upper in ((0, 30), (30, 60), (60, 120)):
            items = [step for step in result['steps'] if lower <= step['minute'] <= upper]
            states = [step['status'] for step in items]
            if 'strong' in states:
                text = ('可研判部分可能有较强回波影响；其余时段尚待核实。' if 'unknown' in states
                        else '按现有回波移动，可能有较强回波影响。')
            elif 'rain' in states:
                text = ('可研判部分可能有降水回波影响；其余时段尚待核实。' if 'unknown' in states
                        else '按现有回波移动，可能有降水回波影响。')
            elif 'unknown' in states:
                text = '该时段资料尚不充分，降水变化需继续观察。'
            else:
                text = '按现有回波移动，暂未见明显降水回波移入。'
            a, b = time_label(now + timedelta(minutes=lower), now), time_label(now + timedelta(minutes=upper), now)
            detail.append(f'{a}—{b}：' + text)
    detail.append('[降水研判]')
    if _trusted_public_forecast(result):
        zero = result['steps'][0]['status']
        zero_text = {'strong': '附近可能有较强回波影响', 'rain': '附近可能有降水回波影响',
                     'clear': '附近暂未见明显降水回波影响', 'unknown': '附近降水情况仍需观察'}[zero]
        detail.append('按回波移动推算，出稿时刻' + zero_text + '。')
        detail.extend(_timing_lines(result, now) or ['按现有回波移动，未来两小时暂未见明显降水回波移入。'])
    else:
        detail.append('本次暂不提供明确的降水到达、增强和结束时刻，请随最新资料继续研判。')
    detail.extend((uncertainty, '[建议]', '1. 持续关注最新雷达、当地气象台短临预报和现场降水情况。',
                   '2. 若遇降水，请备好雨具并注意路面湿滑；回波临近时及时补充研判。'))
    brief = (f'【{name}区域雷达短临研判】\n' + current + f'未来{start}—{finish}，' + future.replace('未来两小时', '该时段')
             + '请关注回波新生、发展或消散，降水变化以最新雷达和现场情况为准。' + advice)
    status = result['current']['status'] if _public_valid(result) else 'unavailable'
    conclusion = {'strong': '最新雷达观测目标区域有较强回波', 'rain': '最新雷达观测目标区域有降水回波',
                  'clear': '最新雷达观测目标区域暂无明显降水回波', 'unknown': '附近零散回波的降水影响尚待确认',
                  'unavailable': '雷达资料尚待核实，当前降水情况需核实'}[status]
    if not _trusted_public_forecast(result):
        known = [step['status'] for step in result['steps'] if step['status'] != 'unknown'] if _public_valid(result) else []
        affected = '已能研判时段可能有较强回波影响' if 'strong' in known else '已能研判时段可能有降水回波影响' if 'rain' in known else None
        conclusion += ('；' + affected + '，其余时段尚待核实，起止仍需观察' if affected
                       else '；未来两小时降水变化和起止仍需观察')
    elif result.get('arrival_minutes') is not None:
        minute = result['arrival_minutes']
        conclusion += ('；按回波移动推算，出稿时刻可能已有回波影响' if minute == 0
                       else f'；预计{time_label(now + timedelta(minutes=minute), now)}前后可能有回波影响')
    else:
        conclusion += '；按现有移动暂未见明显降水回波移入，仍需关注新生回波'
    notice = (f'【{name}雷达短临预报草稿】{now:%m月%d日 %H:%M}\n结论：' + conclusion
              + '\n后面两条依次为详细版和简版正文，审核后可转发。\n发出前请核对：' + '；'.join(_human_checks(result)) + '。')
    result.update(notice=notice, external='\n'.join(detail), brief=brief, internal=report(result),
                  message_format='notice-detail-brief-v1')
    result['messages'] = [result['notice'], result['external'], result['brief']]
    return result


def frame_cache_key(path, config=None):
    target = config or Config()
    target.validate()
    signature = {'path': path, 'target': {'name': target.name, 'lat': float(target.lat), 'lon': float(target.lon), 'radius_km': float(target.radius_km)},
                 'crop_mode': 'legacy' if config is None else 'target',
                 'projection_version': sz_radar.FRAME_PROJECTION_VERSION, 'decode_version': sz_radar.FRAME_DECODE_VERSION,
                 'bounds': [sz_radar.WEST, sz_radar.SOUTH, sz_radar.EAST, sz_radar.NORTH],
                 'reference': [sz_radar.SITE_LAT, sz_radar.SITE_LON], 'palette': sz_radar.PALETTE,
                 'colour_distance': sz_radar.MAX_COLOUR_DISTANCE}
    return hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def cached_frame(path, cache_dir=None, config=None):
    if config is not None:
        config.validate()
    def read():
        return sz_radar.frame(path) if config is None else sz_radar.frame(path, lat=config.lat, lon=config.lon, radius_km=config.radius_km)
    if cache_dir is None:
        return read()
    cache_dir = Path(cache_dir)
    key = frame_cache_key(path, config)
    target = cache_dir / (key + '.npz')
    if target.exists():
        try:
            with np.load(target, allow_pickle=False) as cached:
                value = tuple(cached[name].copy() for name in ('dbz', 'east', 'north'))
            grid_axes(*value)
            return value
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
    value = read()
    grid_axes(*value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.part')
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, dbz=value[0], east=value[1], north=value[2])
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def load_frames(now=None, cache_dir=None, max_frames=6, config=None):
    config = config or Config()
    config.validate()
    if not 1 <= max_frames <= 6:
        raise ValueError('每次仅允许获取最新1–6帧')
    now = as_bjt(now or datetime.now(BJT))
    response = requests.get(sz_radar.BASE + 'contour/radar/radarContour.js', headers=sz_radar.HEADERS, timeout=(5, 20))
    response.raise_for_status()
    entries = []
    for path in re.findall(r'radar/cappi/[^"\']+?\.png[^"\']*', response.text):
        match = re.search(r'(\d{12})\.png', path)
        if not match:
            continue
        try:
            when = datetime.strptime(match.group(1), '%Y%m%d%H%M').replace(tzinfo=BJT)
        except ValueError:
            continue
        entries.append((when, path))
    entries = sorted(entries, key=lambda item: item[0])[-max_frames:]
    if not entries:
        return []
    # An invalid latest timestamp needs no expensive image fetch to reject it.
    age = (now - entries[-1][0]).total_seconds() / 60
    if not 0 <= age <= config.max_age_minutes:
        return [(entries[-1][0], None, None, None)]
    # Keep a fresh latest observation even if historical images are unavailable.
    latest_when, latest_path = entries[-1]
    frames = [(latest_when, *cached_frame(latest_path, cache_dir, config=config))]
    checks = []
    for when, path in reversed(entries[:-1]):
        try:
            frames.append((when, *cached_frame(path, cache_dir, config=config)))
        except (requests.RequestException, OSError, ValueError) as error:
            checks.append(f'历史雷达帧{when:%H:%M}读取失败（{type(error).__name__}），该帧未参与移动估计')
    return FrameBatch(sorted(frames, key=lambda frame: frame[0]), checks)


def save_result(result, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('输出目录已有文件，请提供新的唯一版本目录，禁止覆盖')
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, text in ((output_dir / 'radar_result.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n'),
                           (output_dir / '雷达短临预报.txt', result['external'] + '\n'),
                           (output_dir / '雷达短临预报_提醒.txt', result['notice'] + '\n'),
                           (output_dir / '雷达短临预报_简版.txt', result['brief'] + '\n'),
                           (output_dir / '雷达短临预报_内部依据.txt', result['internal'] + '\n')):
            with path.open('x', encoding='utf-8') as handle:
                created.append(path)
                handle.write(text)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='指定地区雷达-only未来0–2小时短临预报；不发送消息')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--now', help='Inject actual generation time for dry-run, ISO8601 or Beijing YYYY-MM-DD HH:MM')
    parser.add_argument('--cache-dir', type=Path, default=Path(os.environ['LIGHTSHOW_DATA_DIR']) / 'radar_frames' if os.environ.get('LIGHTSHOW_DATA_DIR') else None)
    parser.add_argument('--radius-km', type=float, default=3)
    parser.add_argument('--target-name', default='市民中心')
    parser.add_argument('--lat', type=float, default=22.543)
    parser.add_argument('--lon', type=float, default=114.059)
    parser.add_argument('--rain-threshold', type=float, default=25)
    parser.add_argument('--strong-threshold', type=float, default=35)
    parser.add_argument('--support-fraction', type=float, default=0.10)
    parser.add_argument('--support-pixels', type=int, default=3)
    parser.add_argument('--require-publishable', action='store_true', help='Compatibility: invalid latest observations always exit 2')
    args = parser.parse_args(argv)
    now = as_bjt(args.now or datetime.now(BJT))
    config = Config(name=args.target_name, lat=args.lat, lon=args.lon, radius_km=args.radius_km, rain_threshold=args.rain_threshold,
                    strong_threshold=args.strong_threshold, support_fraction=args.support_fraction, support_pixels=args.support_pixels)
    config.validate()
    try:
        frames = load_frames(now=now, cache_dir=args.cache_dir, config=config)
        if args.now is None:
            now = datetime.now(BJT)
        result = analyse(frames, now=now, config=config)
    except (requests.RequestException, OSError, ValueError) as error:
        if args.now is None:
            now = datetime.now(BJT)
        result = analyse([], now=now, config=config)
        result['blocking_checks'] = ['雷达资料获取或解析失败：' + (str(error) if isinstance(error, ValueError) else type(error).__name__)]
        result['checks'] = result['blocking_checks'] + result['checks']
        refresh_presentation(result)
    save_result(result, args.output_dir)
    print(result['external'])
    return 0 if result['publication_allowed'] else 2


if __name__ == '__main__':
    sys.exit(main())
