"""Restored rolling briefing: radar corridor + fresh official context, three texts.

The existing area-based nowcast remains in the JSON for rain-alert decisions;
the independent rolling_radar block describes the original upstream algorithm.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys

import requests
import radar_nowcast as radar
import radar_monitor
import radar_echo_filter
import rolling_official as official
import rolling_radar

BJT = radar.BJT
VERSION = 'radar-rolling-v5'


def radar_text(result, now):
    """Describe the monitoring area, then a qualified target-impact outlook."""
    name = result['target']['name']
    if not radar._public_valid(result):
        return radar._public_current(result, now)
    rolling = result['rolling_radar']
    monitor = result.get('radar_monitor') or {}
    monitor_valid = (monitor.get('complete')
                     and monitor.get('observed_at') == result.get('radar_time'))
    outlook = result.get('radar_outlook') or radar_monitor.outlook(result)
    impact = outlook['impact']
    if not monitor_valid:
        current = f'{name}周边150公里雷达覆盖不足，整体回波分布尚不能确定。'
    else:
        radius = monitor['radius_km']
        local_echo = any(rolling.get(key) for key in ('current_rain', 'current_weak', 'current_strong'))
        if (monitor.get('status') == 'clear' and result['current'].get('status') == 'clear'
                and not local_echo):
            current = f'雷达显示{name}周边{radius:g}公里暂无明显降水回波。'
        elif monitor.get('status') == 'filtered_only':
            current = f'雷达显示，{name}周边{radius:g}公里暂未见明显成片降水回波。'
            if impact.get('basis') == 'station_rain_observation':
                station = impact.get('station_name') or name
                station = station if station.endswith('站') else station + '站'
                current += f'{station}近期有实测降水，请关注现场变化。'
                return current
            elif impact.get('status') == 'possible':
                # A significant echo beyond the monitoring circle can still
                # enter a fully covered target path within the forecast window.
                pass
            else:
                current += '继续关注后续天气变化。'
                return current
        elif monitor.get('status') == 'echo':
            nearest = monitor.get('nearest')
            current = f'雷达显示，{name}周边{radius:g}公里内'
            if nearest and nearest['distance_km'] > result['target']['radius_km']:
                current += (f'，{nearest["bearing"]}方向约'
                            f'{nearest["distance_km"]:.0f}公里处有降水回波。')
            else:
                current += '有降水回波。'
        elif monitor.get('status') == 'sparse':
            current = f'雷达显示{name}周边{radius:g}公里内有零散回波。'
        else:
            current = f'{name}周边{radius:g}公里回波分布需进一步核实。'
    if monitor_valid and monitor.get('status') in ('echo', 'sparse'):
        motion = outlook['motion']
        if motion['status'] == 'estimated':
            current += f'可追踪回波整体向{motion["direction_to"]}移动。'
        elif rolling.get('status') == 'slow':
            current += '回波移动缓慢。'
        else:
            current += '回波移动趋势尚不明确。'
    if impact['status'] == 'ongoing':
        if impact.get('basis') == 'station_rain_observation':
            station = impact.get('station_name') or name
            station = station if station.endswith('站') else station + '站'
            current += f'{station}近期有实测降水，请关注现场变化。'
        else:
            kind = {'strong': '较强降水', 'weak': '较弱降水'}.get(impact.get('current_kind'), '降水')
            current += f'目前已有{kind}回波影响{name}附近。'
    elif impact['status'] == 'possible':
        current += f'未来2小时可能影响{name}，需关注后续变化。'
    elif impact['status'] == 'not_expected':
        current += f'按现有回波移动外推，预计未来2小时对{name}暂无明显影响。'
    elif not (monitor_valid and monitor.get('status') == 'clear'):
        current += f'未来2小时是否影响{name}仍需继续监测。'
    return current


def present(result):
    now = radar.as_bjt(result['generated_at'])
    name = result['target']['name']
    public = result.get('radar_public') or result
    context, rolling = result['official'], public['rolling_radar']
    current = radar_text(public, now)
    # Presentation concerns the configured target, not the separate 100-km
    # all-clear used to reset alarm episodes. Retain any positive approach signal.
    approaching_echo = any(step.get('rain') is True or step.get('strong') is True
                           for step in public.get('steps', []))
    no_precip_echo = (radar._public_valid(public) and not approaching_echo
                      and public['current'].get('status') == 'clear'
                      and public['current'].get('rain') is False
                      and not any(rolling.get(key) for key in
                                  ('current_rain', 'current_weak', 'current_strong', 'upstream_echo')))
    filtered_only = (public.get('radar_monitor') or {}).get('status') == 'filtered_only'
    timing = '' if no_precip_echo or filtered_only else ''.join(rolling_radar.describe(rolling, public))
    obs = official.observation_text(context)
    warning = official.warning_text(context)
    complete = context['forecast_complete']
    weather = official.weather_text(context) if complete else '逐时预报未完整覆盖未来两小时，完整天气趋势暂不能确定。'
    span = radar.time_label(now, now) + '—' + radar.time_label(now + timedelta(hours=2), now)
    nc = context['nowcast']
    # Always distinguish a citywide source quotation from a point forecast. No
    # automatic extraction of future-day/coastal gusts into the local forecast.
    nc_text = (f'{context["city"]}市气象台{radar.time_label(nc["time"], now)}短临预报原文（全市参考，注意其中分区和时段）：' + nc['text']) if nc else ''
    storm = any(word in ' '.join(w['name'] for w in context['warnings']) for word in ('雷电', '雷雨', '暴雨', '大风', '台风'))
    storm = storm or any('雷' in row['state'] for row in context['hourly'])
    wet = (rolling.get('current_rain') or rolling.get('current_weak') or rolling.get('upstream_echo')
           or (public.get('radar_outlook') or {}).get('impact', {}).get('status') == 'possible')
    advice = ('降水回波临近或影响时备好雨具，注意路面湿滑；' if wet else '')
    if storm:
        advice += '按当地预警要求做好防御，雷雨时及时进入安全室内场所；'
    if any('森林火险' in w['name'] for w in context['warnings']):
        advice += '注意用火安全；'
    advice += '关注最新预报、预警和雷达变化。'
    uncertainty = '回波新生、增强或消散可能改变降水变化，请结合后续雷达和现场情况核对。'
    if any(rolling.get(key) is not None for key in ('arrival_minutes','strong_arrival_minutes','strong_end_minutes','end_minutes')):
        uncertainty = '到达和结束时间按现有回波移动估算。' + uncertainty
    detail = [f'[{name}雷达短临预报]', f'[预报时效]{now:%m月%d日} {span}',
              '[当前实况]' + obs + current, '[天气趋势]' + weather, '[分时段预报]']
    for begin, finish in ((0, 60), (60, 120)):
        a, b = now + timedelta(minutes=begin), now + timedelta(minutes=finish)
        left, right = a.replace(minute=0, second=0, microsecond=0), b.replace(minute=0, second=0, microsecond=0)
        if b != right:
            right += timedelta(hours=1)
        rows = [row for row in context['hourly'] if left <= radar.as_bjt(row['time']) <= right]
        expected = int((right-left).total_seconds()/3600)+1
        text = official.weather_text(context, rows) if len(rows) == expected else '逐时预报覆盖不足，请结合后续资料判断。'
        detail.append(radar.time_label(a, now) + '—' + radar.time_label(b, now) + '：' + text)
    if timing:
        detail.append('[降水过程研判]' + timing)
    detail.append('[预警信号]' + warning)
    if nc_text:
        detail.append('[官方短临]' + nc_text)
    detail.append('[建议]' + advice)
    if not no_precip_echo:
        detail.append(uncertainty)
    brief_current = current.replace('继续关注后续天气变化。', '')
    brief = (f'【{name}区域雷达短临研判】{now:%m月%d日 %H:%M}\n'
             + f'未来{span}，' + weather + brief_current + timing + warning + advice)
    checks = list(dict.fromkeys(context['checks']))
    if not no_precip_echo and not rolling['timing_reliable']:
        checks.append('降水起止尚不能可靠确定')
    checks.append('最新预报、预警和现场情况' if no_precip_echo else '回波变化及现场降水是否与研判一致')
    notice = (f'【{name}雷达短临预报草稿】{now:%m月%d日 %H:%M}\n'
              + '结论：' + current + weather + '\n'
              + '后面两条依次为详细版和简版正文。发出前请核对：' + '；'.join(checks) + '。')
    # Unify public brackets, including any quoted feed labels; the original
    # source text remains untouched in the official/internal metadata.
    brackets = str.maketrans({'[': '【', ']': '】', '［': '【', '］': '】'})
    notice, external, brief = (text.translate(brackets) for text in (notice, '\n'.join(detail), brief))
    result.update(notice=notice, external=external, brief=brief,
                  messages=[notice, external, brief], message_format='notice-detail-brief-v1')


def build(frames, raw, profile, config, now, radar_error=None):
    result = radar.analyse(frames, now=now, config=config)
    if radar_error:
        result['blocking_checks'].append(radar_error)
    result['algorithm_version'] = VERSION
    result['rolling_radar'] = rolling_radar.estimate(frames, result, config)
    result['radar_monitor'] = radar_monitor.estimate(frames, result, config)
    result['radar_outlook'] = radar_monitor.outlook(result)
    result['official'] = official.summarize(raw, profile, now)
    result['radar_public_summary'], result['radar_public'] = radar_echo_filter.apply(frames, result, config)
    if result['official']['nowcast']:
        result['covered_nowcasts'] = [official.event_stamp(result['official']['nowcast'], result['target'])]
    result['checks'].extend(result['official']['checks'])
    result['internal'] += '\n\n原滚动研判上游剖面（当前圆域算法另供降水临近报警）：\n' + json.dumps(result['rolling_radar'], ensure_ascii=False, indent=2)
    result['internal'] += '\n\n目标周边150公里雷达背景监测：\n' + json.dumps(result['radar_monitor'], ensure_ascii=False, indent=2)
    result['internal'] += '\n\n回波整体移动与目标影响依据：\n' + json.dumps(result['radar_outlook'], ensure_ascii=False, indent=2)
    result['internal'] += '\n\n公开回波筛选与影响依据（原始报警分析保留）：\n' + json.dumps({
        'selection': result['radar_public_summary'], 'outlook': result['radar_public']['radar_outlook']}, ensure_ascii=False, indent=2)
    result['internal'] += '\n\n官方资料时效、范围和原始依据：\n' + json.dumps(result['official'], ensure_ascii=False, indent=2)
    present(result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description='雷达滚动研判，未来两小时；仅生成提醒、详细版和简版，不发送')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--target-name', default='市民中心')
    p.add_argument('--lat', type=float, default=22.543)
    p.add_argument('--lon', type=float, default=114.059)
    p.add_argument('--radius-km', type=float, default=3)
    p.add_argument('--rain-threshold', type=float, default=25)
    p.add_argument('--strong-threshold', type=float, default=35)
    p.add_argument('--support-fraction', type=float, default=.1)
    p.add_argument('--support-pixels', type=int, default=3)
    p.add_argument('--now', help='仅用于离线验收的实际出稿时刻')
    args = p.parse_args(argv)
    config = radar.Config(name=args.target_name, lat=args.lat, lon=args.lon, radius_km=args.radius_km,
        rain_threshold=args.rain_threshold, strong_threshold=args.strong_threshold,
        support_fraction=args.support_fraction, support_pixels=args.support_pixels)
    config.validate()
    profile = official.profile_for({'lat': config.lat, 'lon': config.lon, 'radius_km': config.radius_km})
    now = radar.as_bjt(args.now or datetime.now(BJT))
    error = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        weather_job = pool.submit(official.fetch, profile)
        try:
            frames = radar.load_frames(now=now, cache_dir=args.cache_dir, config=config)
        except (requests.RequestException, OSError, ValueError) as exc:
            frames = []
            error = '雷达获取或解析失败：' + type(exc).__name__
        raw = weather_job.result()
    # Revalidate every feed after all requests complete, at actual issue time.
    now = radar.as_bjt(args.now or datetime.now(BJT))
    result = build(frames, raw, profile, config, now, error)
    radar.save_result(result, args.output_dir)
    print(result['external'])
    return 0 if result['publication_allowed'] else 2


if __name__ == '__main__':
    sys.exit(main())
