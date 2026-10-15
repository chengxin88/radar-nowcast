"""Draft the Futian Civic Center light-show weather briefing from official Shenzhen forecasts and EC guidance."""
import argparse
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from ec_event_point import BJT, DIRECTIONS, beaufort, extract, latest_run, validate_event_coverage
from sz_official import fetch, validate as validate_official
from sz_radar import describe, read_radar, summary
import sz_radar
import sz_lightning

LEVELS = ('低', '中', '较高')
SUITABILITY = ('非常适宜', '适宜', '较适宜', '不太适宜', '不适宜')
EVENT = (19.5, 21.5)
PERIOD_END = 20
RAIN_24H = ((0.1, '无降水'), (10, '小雨'), (25, '中雨'), (50, '大雨'), (100, '暴雨'), (250, '大暴雨'), (math.inf, '特大暴雨'))
HOURLY_HEAVY_MM = 5
RADAR_UPGRADE_KM = 50
HAZE_VISIBILITY = {'重度': (0, 2), '中度': (2, 3), '轻度': (3, 5), '微度': (5, 10), '无': (10, None)}
RISK_LINE = re.compile(r'风险等级：降雨 (\S+)｜雷电 (\S+)｜大风 (\S+)｜闷热 (\S+?)，(\S+?)开展灯光秀活动')
ALARM_COLOURS = {'blue': '蓝', 'yellow': '黄', 'orange': '橙', 'red': '红', 'white': '白'}
EVENING = re.compile(r'傍晚|前半夜|晚上|晚间|夜间|今晚|今夜|明晚|入夜|上半夜')
DAY_WORDS = re.compile(r'(\d{1,2})日|今天|今日|今晚|今夜|明天|明日|明晚')
HEAVY_WORDS = re.compile(r'暴雨|大雨|强降水|强降雨|强（雷）阵雨|强雷阵雨|强雷雨')
OTHER_AREAS = ('罗湖', '南山', '盐田', '宝安', '龙岗', '龙华', '坪山', '光明', '大鹏', '深汕', '海区', '海域')
ISSUE_HOURS = (8, 16, 20)
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('LIGHTSHOW_DATA_DIR', str(HERE)))
OUTPUT_DIR = Path(os.environ.get('LIGHTSHOW_OUTPUT_DIR', str(HERE.parent / 'outputs')))
WEATHER_WORDS = r'(?:强(?:雷)?阵雨|强雷雨|雷阵雨|雷雨|雷电|雷暴|强降雨|强降水|暴雨|大雨|降雨|降水|阵雨|小雨|雨|大风|阵风)'
NEGATIVE_WEATHER = re.compile(r'(?:暂无|没有|未见|未出现|不会出现|不出现|不发生|不再有|不再出现|无)(?:明显的?|较强的?)?'
                              + WEATHER_WORDS + r'(?:[、和及](?:明显的?|较强的?)?' + WEATHER_WORDS + r')*')


def send(text):
    # Generation does not import the push transport or require its dependencies.
    from wechat_push import send as transport_send
    return transport_send(text)


def positive_weather(text):
    """Remove explicit absence; keep possibilities such as '不排除雷雨'."""
    text = NEGATIVE_WEATHER.sub('', str(text or ''))
    return re.sub(WEATHER_WORDS + r'(?:的)?(?:可能性(?:较小|很小|低)|概率(?:较低|很低)|不明显)', '', text)


def has_rain(text):
    return bool(re.search(r'雨|降水', positive_weather(text)))


def has_thunder(text):
    return '雷' in positive_weather(text)


def activity_day(issue, mode='manual', explicit=None):
    if explicit:
        return datetime.strptime(explicit, '%Y-%m-%d').replace(tzinfo=BJT)
    if mode not in ('scheduled', 'nowcast', 'manual'):
        raise ValueError('Unknown briefing mode')
    if mode == 'scheduled' and (issue.hour not in ISSUE_HOURS or issue.minute):
        raise ValueError('scheduled 模式时次必须为 08:00、16:00 或 20:00')
    return issue.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=mode == 'scheduled' and issue.hour == 20)


def validated_radar(radar, now=None):
    if not radar or 'error' in radar:
        return radar
    try:
        status = sz_radar.freshness(at(radar['last']), now)
    except (KeyError, TypeError, ValueError):
        return {'error': '雷达末帧时间缺失或异常'}
    if not status['fresh']:
        return {'error': status['freshness_reason'], 'last': radar['last']}
    return radar


def at(text):
    return datetime.strptime(text[:16], '%Y-%m-%d %H:%M').replace(tzinfo=BJT)


def hours_since(moment, day):
    return (moment - day).total_seconds() / 3600


def number(value):
    return str(int(math.floor(float(value) + 0.5)))


def temperature_text(values):
    low, high = number(min(values)), number(max(values))
    return f'{low}℃' if low == high else f'{low}～{high}℃'


def max_force(text):
    found = [int(value) for value in re.findall(r'\d+', str(text))]
    return max(found) if found else 0


def apparent(t, rh, wind_ms):
    vapour = rh / 100 * 6.105 * math.exp(17.27 * t / (237.7 + t))
    return t + 0.33 * vapour - 0.70 * wind_ms - 4.0


def mentioned_days(clause):
    days = set()
    for first, last in re.findall(r'(\d{1,2})日?(?:[-—～~至到](\d{1,2}))?日', clause):
        days.update(range(int(first), int(last or first) + 1))
    return days


def trend(tip, daily, day, same_day):
    # 用户要求：天气趋势优先用10天预报，与简版一致；没有10天预报时才引用天气提示
    if daily:
        return f"预计{day.day}日{daily['report']}，气温{number(daily['minT'])}～{number(daily['maxT'])}℃。"
    if tip:
        text = re.sub(r'^【[^】]*】', '', tip['prompt'].strip())
        text = re.sub(r'深圳市气象台\d{4}年.*?发布\s*$', '', text)
        words = ('今天', '今日') if same_day else ('明天', '明日')
        clauses = [clause.strip() for clause in re.split(r'[；。]', text) if clause.strip()]
        picked = [clause for clause in clauses if day.day in mentioned_days(clause) or any(word in clause for word in words)]
        picked = [re.sub(r'^(?:其中|另外|此外)，?', '', clause) for clause in picked]
        if picked:
            return '；'.join(picked) + '。'
    if daily:
        return f"预计{day.day}日{daily['report']}，气温{number(daily['minT'])}～{number(daily['maxT'])}℃。"
    return '（未读取到深圳市气象台预报，发布前请人工补充。）'


def nowcast(data, day, issue, now=None):
    now = now or datetime.now(BJT)
    tips = data.get('tips') or {}
    candidates = [tips.get('important')] + list(tips.get('more') or [])
    fresh = sorted({item['d']: item for item in candidates
                    if item and item.get('d') and item.get('prompt') and timedelta(0) <= now - at(item['d']) <= timedelta(hours=36)}.values(),
                   key=lambda item: item['d'], reverse=True)
    for item in fresh:
        issued = at(item['d']).date()
        body = re.sub(r'深圳市气象台\d{4}年.*?发布\s*$', '', re.sub(r'^【[^】]*】', '', item['prompt'].strip()))
        parts = []
        for sentence in re.split(r'[。；]', body):
            target, evening = issued == day.date(), False
            for part in re.split(r'[，,]', sentence):
                found = DAY_WORDS.search(part)
                if found:
                    word = found.group(0)
                    offset = 0 if word.startswith('今') else 1 if word.startswith('明') else None
                    target = (int(found.group(1)) == day.day) if offset is None else (issued + timedelta(days=offset) == day.date())
                    evening = False
                if EVENING.search(part):
                    evening = True
                elif re.search(r'白天|上午|中午|下午|午后|早晨|凌晨|后半夜|未来\d', part):
                    evening = False
                if target and evening:
                    parts.append(part.strip())
        if not parts:
            continue
        joined = '，'.join(parts)
        positive = positive_weather(joined)
        gusts = []
        for part in parts:
            clean = positive_weather(part)
            if '风' in clean and not re.search(r'沿海|高地|海区|海面|海域', clean):
                gusts += [(int(low), int(high or low)) for low, high in re.findall(r'(\d{1,2})(?:[-—～~至到](\d{1,2}))?级', clean)]
        title = re.match(r'【([^】]*)】', item['prompt'].strip())
        return {'time': item['d'][:16], 'title': title.group(1) if title else '市气象台提示', 'text': joined,
                'rain': 2 if HEAVY_WORDS.search(positive) else 1 if has_rain(positive) else 0,
                'thunder': has_thunder(positive), 'gust': max(gusts, key=lambda gust: gust[1]) if gusts else None}
    return None

def ec_rain_index(row):
    if row['rain_point_mm'] >= 3 and row['rain_area_ge1_fraction'] >= 0.5:
        return 2
    if row['rain_point_mm'] >= 1 or row['rain_area_ge1_fraction'] >= 0.25:
        return 1
    return 0


def official_rain_index(item):
    if not has_rain(item['state']):
        return 0
    return 2 if float(item.get('r') or 0) >= HOURLY_HEAVY_MM else 1


def ec_sky(rows):
    cloud = sum(row['tcc_percent'] for row in rows) / len(rows)
    if cloud >= 80:
        return '阴天'
    if cloud >= 70:
        return '多云间阴天'
    if cloud >= 40:
        return '多云'
    if cloud >= 30:
        return '晴间多云'
    return '晴天'


def sky_base(state):
    base = re.split(r'[，,；]|间中|有', state or '')[0].strip()
    return '' if '雨' in base else base


def official_sky(items):
    bases = [base for base in (sky_base(item['state']) for item in items) if base]
    return Counter(bases).most_common(1)[0][0] if bases else ''


def overlapping(rows, hours):
    return [row for row in rows if any(row['start_h'] < hour <= row['end_h'] for hour in hours)]


def rain_name(thunder, showery=True):
    if showery:
        return '（雷）阵雨' if thunder else '阵雨'
    return '雷雨' if thunder else '降雨'


def period_text(start, end, hours, rows, offset, thunder_day, base_sky, showery):
    wanted = list(range(start + 1, end + 1))
    official = [hours[float(hour)] for hour in wanted if float(hour) in hours]
    missing = [hour for hour in wanted if float(hour) not in hours]
    ec_rows = overlapping(rows, missing) if missing else []
    all_rows = overlapping(rows, wanted)
    if not official and not ec_rows:
        return None
    rain = 0
    temps = []
    if official:
        rain = max(official_rain_index(item) for item in official)
        temps += [float(item['minT']) for item in official] + [float(item['maxT']) for item in official]
    if ec_rows:
        rain = max(rain, max(ec_rain_index(row) for row in ec_rows))
        temps += [row['t2m_c'] + offset for row in ec_rows]
    unstable = any(row['mucape_point'] >= 1000 for row in all_rows)
    thunder = any(has_thunder(item['state']) for item in official) or (thunder_day and rain >= 1 and unstable)
    sky = official_sky(official) or base_sky or (ec_sky(ec_rows or all_rows) if (ec_rows or all_rows) else '多云')
    if rain == 2:
        text = f'{sky}，有{rain_name(thunder, showery)}，雨势相对明显'
    elif rain == 1:
        text = f'{sky}，有{rain_name(thunder, showery)}'
    elif any(row['rain_area_max_mm'] >= 1 for row in ec_rows):
        text = f'{sky}，局地可能有短时阵雨'
    else:
        text = sky
    source = '市气象台' if not ec_rows else ('EC' if not official else '市气象台+EC')
    return f'{start % 24:02d}—{end % 24:02d}时', text, temperature_text(temps), source


def rain_timing(sequence, shower):
    indices = [index for _, index in sequence]
    if max(indices) == 0:
        return '降雨可能性较小'
    if indices[0] == 0 and indices[-1] > 0:
        hour = next(start for start, index in sequence if index > 0)
        return f'活动开始前后降雨可能性较小，{hour % 24}时以后出现{shower}的可能性增大'
    if indices[0] > 0 and indices[-1] == 0:
        hour = next(start for start, index in sequence if index == 0)
        return f'活动开始前后有{shower}可能，{hour % 24}时以后逐渐减弱'
    return f'有{shower}' + ('，局地雨势较大' if max(indices) == 2 else '')


def temperature_offset(rows, hours):
    matches = [(row['end_h'], float(hours[row['end_h']]['minT']) - row['t2m_c']) for row in rows if row['end_h'] in hours]
    if not matches:
        return 0.0, None
    hour, delta = max(matches)
    return max(-3.0, min(3.0, delta)), hour


def haze_for(haze, moment):
    choices = []
    for item in ((haze or {}).get('data') or {}).values():
        try:
            start = datetime.strptime(item['fsddatetime'], '%Y年%m月%d日 %H:%M').replace(tzinfo=BJT)
            end = datetime.strptime(item['feddatetime'], '%Y年%m月%d日 %H:%M').replace(tzinfo=BJT)
        except (KeyError, ValueError):
            continue
        if start <= moment <= end:
            choices.append((end, item))
    return min(choices, key=lambda choice: choice[0])[1] if choices else None


def visibility(rain, haze_item, unavailable=False):
    if unavailable:
        return '（灰霾预报未读取，能见度请人工补充）', 0, '未读取'
    level = (haze_item or {}).get('haze_level') or '无'
    low, high = HAZE_VISIBILITY.get(level, HAZE_VISIBILITY['无'])
    if high is None:
        text = '能见度良好，普遍在10公里以上'
    elif low == 0:
        text = f'受灰霾影响，能见度较差，局地低于{high}公里'
    else:
        text = f'受灰霾影响，能见度{low}～{high}公里'
    if rain == 2:
        text += '，强降雨时局地能见度可能降至2公里以下'
    elif rain == 1:
        text += '，雨时能见度下降'
    risk = 2 if level in ('重度', '中度') else 1 if level == '轻度' or rain == 2 else 0
    return text, risk, level


def join_names(names):
    return names[0] if len(names) == 1 else '、'.join(names[:-1]) + '和' + names[-1]


def brief_temperature(values):
    low, high = number(min(values)), number(max(values))
    return f'气温{low}℃左右' if low == high else f'气温在{low}-{high}℃之间'


def brief_visibility(haze_level, rain, fog):
    if fog:
        text = '受大雾或灰霾影响，能见度较差'
    elif haze_level == '未读取':
        return '（能见度请人工补充）'
    else:
        low, high = HAZE_VISIBILITY.get(haze_level, HAZE_VISIBILITY['无'])
        if high is None:
            text = '能见度普遍在10公里以上，视野较好'
        elif low == 0:
            text = f'受灰霾影响，能见度较差，局地低于{high}公里'
        else:
            text = f'受灰霾影响，能见度{low}-{high}公里' + ('，视野较差' if high <= 5 else '')
    if rain == 2:
        text += '；强降雨时局地能见度可能降至2公里以下'
    elif rain == 1:
        text += '；若遇短时阵雨，局地能见度可能短时略降'
    return text


def level_range(levels):
    levels = sorted({int(level) for level in levels if str(level).strip() != ''})
    if not levels:
        return ''
    return str(levels[0]) if len(levels) == 1 else f'{levels[0]}～{levels[-1]}'


def day_gust_text(rows, start, end):
    """EC 市民中心点阵风（全天）：分白天（18时前）和傍晚以后两段，返回（简版文字, 全天最大级数）。"""
    rows = [row for row in rows if row['end_h'] > start and row['start_h'] < end]
    if not rows:
        return '', 0
    day = [row['gust_point_ms'] for row in rows if row['start_h'] < 18]
    night = [row['gust_point_ms'] for row in rows if row['end_h'] > 18]
    day_level = beaufort(max(day)) if day else 0
    night_level = beaufort(max(night)) if night else 0
    top = max(day_level, night_level)
    if not day or not night or day_level == night_level:
        return f'阵风{top}级左右', top
    if day_level > night_level:
        return f'白天阵风{day_level}级左右，傍晚起阵风减弱到{night_level}级左右', top
    return f'白天阵风{day_level}级左右，傍晚起阵风加大到{night_level}级左右', top


def brief_tone(rain, thunder, wind, weather):
    if max(rain, thunder, wind) == 2:
        return '天气不稳定，'
    if thunder or wind or '雷' in weather or '雨势较大' in weather:
        return ''
    return '天气总体平稳，' if rain or '雨' in weather else '天气平稳，'


def brief_impact(rain, thunder, wind, heat, shower, warnings):
    high = [name for name, level in (('强降水', rain), ('雷电', thunder), ('大风', wind)) if level == 2]
    if high:
        text = f'该时段可能出现{join_names(high)}等高影响天气，对市民中心区域户外活动有较大不利影响，请密切关注最新预警信号。'
    elif warnings:
        text = f'目前福田区{warnings}生效中，对市民中心区域户外活动有较大不利影响，请密切关注最新预警信号。'
    else:
        middle = [f'短时{shower}'] if rain else ['局地雷阵雨'] if thunder else []
        if wind:
            middle.append('较大阵风')
        if middle:
            text = f'该时段可能出现{join_names(middle)}，对市民中心区域户外活动有一定影响，请关注最新天气预报和预警信号。'
        else:
            text = '该时段无强降水、雷电、大风等高影响天气，对市民中心区域户外活动无明显不利影响。'
    if heat == 2:
        text += '体感闷热，请注意防暑降温。'
    return text


def nominal_issue(now):
    for days in (-1, 0, 1):
        for hour in ISSUE_HOURS:
            slot = (now + timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
            if slot >= now - timedelta(hours=3):
                return slot


def edition(day, issue):
    if issue.date() < day.date():
        return 20
    return 8 if issue.hour < 12 else 16


def today_trend(text, day):
    if not re.match(rf'(?:预计)?(?:今[天日]|(?:周[一二三四五六日天])?（?{day.day}日)', text):
        return text
    body = re.sub(rf'^(?:预计)?(?:今[天日])?(?:周[一二三四五六日天])?(?:（?{day.day}日）?)?', '', text)
    return f'预计今天（{day.day}日）{body}'


def covers_futian(area):
    area = str(area or '')
    return any(word in area for word in ('福田', '全市', '陆地')) or not any(word in area for word in OTHER_AREAS)

def warning_label(item):
    kind = re.sub(r'预警(信号)?$', '', str(item.get('alarmType') or '')).strip()
    colour = str(item.get('alarmColor') or '').strip()
    colour = ALARM_COLOURS.get(colour.lower(), colour).rstrip('色')
    if kind:
        return f'{colour}色{kind}预警信号' if colour else f'{kind}预警信号'
    return str(item.get('str') or '')

def warning_names(alarm):
    futian, others = [], []
    for item in alarm.get('subAlarm') or []:
        label = warning_label(item)
        if not label:
            continue
        area = str(item.get('alarmArea') or '')
        if covers_futian(area):
            futian.append(label)
        else:
            others.append(f'{label}（{area}）')
    info = alarm.get('alarmInfo', '')
    if not futian and not others and '解除' not in info:
        match = re.search(r'【(?:深圳市)?(?:发布|更新|升级|降级|继续生效)?([^】]*预警信号)】', info)
        if match:
            (futian if covers_futian(info) else others).append(match.group(1))
    return list(dict.fromkeys(futian)), list(dict.fromkeys(others))


def rain_record(observation):
    if observation.get('h24r') is None:
        return '（过去24小时降水请人工补充）'
    amount = float(observation['h24r'])
    grade = next(name for upper, name in RAIN_24H if amount < upper)
    text = '过去24小时无降水' if grade == '无降水' else f'过去24小时出现{grade}量级降雨'
    if float(observation.get('h01r') or 0) >= 0.1:
        text += '，近1小时仍有降雨'
    return text


def current_conditions(kind, observation, warning, radar, lightning=None):
    if not observation:
        return '（未读取到实况，请人工补充）' + warning
    moment = at(observation['date']) if observation.get('date') else None
    when = (f'截至{moment.hour}时' + (f'{moment.minute}分' if moment.minute else '')) if moment else '目前'
    basic = f"{when}，福田区气温{number(observation['t'])}℃，相对湿度{number(observation['u'])}%"
    if kind == 8:
        return f'{basic}；{rain_record(observation)}。{warning}'
    level = beaufort(float(observation.get('wf') or 0))
    direction = DIRECTIONS[int((float(observation.get('wd') or 0) + 22.5) // 45) % 8]
    wind = f'{direction}风{level}级' if level else '静风'
    flashes = (f'{sz_lightning.describe(lightning)}。' if sz_lightning.available(lightning)
               else '闪电定位资料缺测（云端已禁用），请人工核对。' if (lightning or {}).get('disabled')
               else '闪电定位资料缺测，请人工核对。')
    return (f'{basic}，{wind}，能见度{number(observation["v"])}公里；{rain_record(observation)}。'
            f'雷达显示{describe(radar)}。{flashes}{warning}')


def previous_briefing(folder, day, issue):
    found = []
    for path in Path(folder).rglob(f'灯光秀气象研判_*_活动{day:%m%d}*.txt'):
        match = re.search(r'_(\d{8}_\d{4})_活动', path.name)
        if match and '简版' not in path.name:
            moment = datetime.strptime(match.group(1), '%Y%m%d_%H%M').replace(tzinfo=BJT)
            if issue - timedelta(days=2) < moment < issue:
                found.append((moment, path))
    if not found:
        return None
    moment, path = max(found)
    return moment, path.read_text(encoding='utf-8-sig'), path.name


def comparison(previous, risks, suitability, day):
    if not previous:
        return '与前次研判相比，（未找到前次研判稿，请人工补充）。'
    moment, text, _ = previous
    label = '昨晚' if moment.date() < day.date() else '上午' if moment.hour < 12 else '前次'
    match = RISK_LINE.search(text)
    if not match:
        return f'与{label}研判相比，（前次稿件未找到风险等级，请人工补充）。'
    changes = []
    for (name, level), old in zip(risks, match.groups()[:4]):
        if old in LEVELS and LEVELS.index(old) != level:
            changes.append(f'{name}风险由{old}{"上调" if level > LEVELS.index(old) else "下调"}为{LEVELS[level]}')
    if match.group(5) != SUITABILITY[suitability]:
        changes.append(f'适宜度由{match.group(5)}调整为{SUITABILITY[suitability]}')
    return f'与{label}研判相比，' + ('，'.join(changes) if changes else '无明显变化') + '。'


def build(day, issue, ec, official, previous=None, radar=None, lightning=None, now=None):
    now = now or datetime.now(BJT)
    validate_event_coverage(ec, day)
    official = validate_official(official, now)
    radar = validated_radar(radar, now)
    kind = edition(day, issue)
    data = {key: value.get('data') for key, value in official.items()}
    failures = [f'{key}：{value["error"]}' for key, value in official.items() if 'error' in value]
    futian = data.get('futian') or {}
    hours = {hours_since(at(item['forecasttime']), day): item for item in futian.get('hoursweather', [])}
    daily = next((item for item in (data.get('tenday') or {}).get('daynew', [])
                  if item['reportTime'][:10] == f'{day:%Y-%m-%d}'), None)
    daily_source = '市气象台10天预报'
    if not daily and (futian.get('dayreport') or {}).get('forecasttime', '')[:10] == f'{day:%Y-%m-%d}':
        daily, daily_source = futian['dayreport'], '福田区日预报（10天预报已不含活动日）'
    tip = (data.get('tips') or {}).get('important')
    if tip and not timedelta(0) <= now - at(tip['d']) <= timedelta(hours=36):
        tip = None
    same_day = issue.date() == day.date()
    alarm = data.get('alarm') or {}
    citywide = str(alarm.get('ingnalNum', '0')) not in ('0', '') or bool(alarm.get('subAlarm'))
    names, other_warnings = warning_names(alarm) if citywide else ([], [])
    info = alarm.get('alarmInfo', '')
    unknown = citywide and not names and not other_warnings
    active = bool(names) or unknown
    alarm_text = ' '.join(names) or (('' if '解除' in info else info) if unknown else '')
    if 'error' in official['alarm']:
        warning = '目前福田区预警资料缺测或过期，无法确认生效信号，请人工核对。'
    elif not active:
        warning = '目前福田区无生效的气象预警信号。'
    elif names:
        warning = f'目前福田区{"、".join(names)}生效中。'
    else:
        warning = '目前有气象预警信号生效（请核对信号名称和是否覆盖福田区）。'
    observation = data.get('observation') or {}
    commute = (data.get('commute') or {}).get('important')
    trend_text = trend(tip, daily, day, same_day)
    day_report = (daily or {}).get('report', '')
    thunder_day = has_thunder(day_report) or has_thunder(trend_text)
    base_sky = sky_base(day_report)
    showery = '阵雨' in positive_weather(day_report) or not has_rain(day_report)
    # “A转B”：活动时段（晚上）只看“转”后的B
    evening_report = day_report.split('转')[-1]
    turned = evening_report != day_report
    evening_sky = sky_base(evening_report)
    evening_thunder = has_thunder(evening_report) if turned else thunder_day

    rows = [dict(row) for row in ec['rows']]
    for row in rows:
        row['end_h'] = hours_since(at(row['valid_bjt']), day)
        row['start_h'] = row['end_h'] - 3
    offset, offset_hour = temperature_offset(rows, hours)

    periods = []
    start = int(math.floor(hours_since(issue, day)))
    while start < PERIOD_END:
        end = min(start + 6, PERIOD_END)
        period = period_text(start, end, hours, rows, offset, thunder_day, base_sky, showery)
        if period:
            periods.append(period)
        start = end

    event_hours = [hours.get(float(hour)) for hour in (20, 21, 22)]
    event_rows = [row for row in rows if row['start_h'] < EVENT[1] and row['end_h'] > EVENT[0]]
    force = max_force((daily or {}).get('ws', ''))
    wind_dir = (daily or {}).get('wd', '')
    wind_level = (daily or {}).get('ws', '').replace('-', '～')
    ec_window = not all(event_hours)
    if not ec_window:
        event_source = '市气象台福田区逐时预报'
        sequence = [(hour - 1, official_rain_index(item)) for hour, item in zip((20, 21, 22), event_hours)]
        sky = official_sky(event_hours) or evening_sky or ec_sky(event_rows)
        temps = [float(item['minT']) for item in event_hours] + [float(item['maxT']) for item in event_hours]
        humidity = [float(item['h']) for item in event_hours]
        feels = [apparent(float(item['maxT']), float(item['h']), float(item['wspeed'])) for item in event_hours]
        force = max([force] + [int(item['ws']) for item in event_hours])
        wind_dir = wind_dir or event_hours[0]['wd']
        if not wind_level:
            wind_level = level_range([int(item['ws']) for item in event_hours])
        thunder_text = any(has_thunder(item['state']) for item in event_hours)
    else:
        event_source = f'EC {ec["run_bjt"]}（北京时）起报，气温按市气象台逐时预报订正'
        sequence = [(int(row['start_h']), ec_rain_index(row)) for row in event_rows]
        sky = evening_sky or ec_sky(event_rows)
        temps = [row['t2m_c'] + offset for row in event_rows]
        humidity = [row['rh2m_percent'] for row in event_rows]
        feels = [apparent(row['t2m_c'] + offset, row['rh2m_percent'], row['wind_ms']) for row in event_rows]
        thunder_text = False
    if not wind_dir:
        wind_dir = event_rows[0]['wind_dir']
    if not wind_level:
        wind_level = level_range([row['wind_level'] for row in event_rows])

    cape = max((row['mucape_point'] for row in event_rows), default=0)
    gust = max((row['gust_point_ms'] for row in event_rows), default=0)
    gust_area = max((row['gust_area_max_ms'] for row in event_rows), default=0)
    gust_level = beaufort(gust)
    applied = []
    rain = max(index for _, index in sequence)
    if ec_window and has_rain(evening_report):
        rain = max(rain, 1)
    cast = nowcast(data, day, issue, now) if same_day else None
    cast_notes = []
    if cast and cast['rain'] > rain:
        rain = cast['rain']
        cast_notes.append(f'降雨→{LEVELS[rain]}')
    if cast and cast['thunder']:
        thunder_text = True
    unstable = cape >= 1000
    # Explicit official thunder signals are sufficient even when EC is stable/dry.
    thunder = 1 if evening_thunder or thunder_text or (rain >= 1 and unstable) else 0
    if rain >= 2 and cape >= 2000:
        thunder = 2
    wind = 2 if gust_level >= 7 or force >= 6 else 1 if gust_level >= 6 or force >= 5 else 0
    cast_wind, cast_gust_text = 0, ''
    if cast and cast['gust']:
        low, high = cast['gust']
        cast_wind = 2 if high >= 7 else 1 if high >= 6 else 0
        if cast_wind:
            cast_gust_text = ('雷雨时' if cast['rain'] else '') + (f'阵风可达{low}～{high}级' if high != low else f'阵风{low}级左右')
        if cast_wind > wind:
            wind = cast_wind
            cast_notes.append(f'大风→{LEVELS[wind]}')
    if cast_notes:
        applied.append(f'市气象台{cast["time"]}{cast["title"]}提到活动时段天气→{"、".join(cast_notes)}')
    feel = max(feels)
    heat = 2 if feel >= 35 else 1 if feel >= 30 else 0
    event_moment = day + timedelta(hours=EVENT[0])
    haze_item = haze_for(data.get('haze'), event_moment)
    if active and same_day:
        if '大风' in alarm_text or '台风' in alarm_text:
            wind = 2
            applied.append('大风/台风预警→大风较高')
        if '高温' in alarm_text:
            heat = 2
            applied.append('高温预警→闷热较高')
        if kind == 16 and '雷' in alarm_text:
            thunder = 2
            applied.append('雷电预警→雷电较高')
        if kind == 16 and '暴雨' in alarm_text:
            rain = 2
            applied.append('暴雨预警→降雨较高')
    storm = (radar or {}).get('strong')
    if kind == 16 and storm and storm['km'] <= RADAR_UPGRADE_KM:
        rain, thunder = max(rain, 1), max(thunder, 1)
        applied.append(f'雷达{RADAR_UPGRADE_KM}公里内有≥35dBZ回波→降雨、雷电至少为中')
    if kind == 16 and sz_lightning.available(lightning) and lightning['flashes']:
        thunder = max(thunder, 1)
        applied.append(f'闪电定位近1小时{sz_lightning.RADIUS_KM}公里内有闪电→雷电至少为中')
    visibility_text, sight, haze_level = visibility(rain, haze_item, 'error' in official.get('haze', {}))
    fog = active and same_day and kind == 16 and ('大雾' in alarm_text or '灰霾' in alarm_text)
    if fog:
        visibility_text, sight = '受大雾或灰霾影响，能见度较差', 2
        applied.append('大雾/灰霾预警→能见度较差')
    severe_words = ('台风', '大风', '暴雨', '橙色', '红色') if kind == 16 else ('台风', '大风')
    severe = active and same_day and any(word in alarm_text for word in severe_words)
    core = max(rain, thunder, wind)
    high_count = sum(level == 2 for level in (rain, thunder, wind))
    local_shower = rain == 0 and any(row['rain_area_max_mm'] >= 1 for row in event_rows)
    calm = max(core, heat) == 0 and haze_level == '无' and not (active and same_day) and not local_shower and 'error' not in official['alarm']
    if severe or high_count >= 2:
        suitability = 4
    elif core == 2:
        suitability = 3
    elif core == 1 or heat == 2 or sight >= 1:
        suitability = 2
    else:
        suitability = 0 if calm else 1

    shower = rain_name(thunder >= 1, showery)
    if same_day and max(index for _, index in sequence) >= rain:
        rain_text = rain_timing(sequence, shower)
    else:
        rain_text = {0: '降雨可能性较小', 1: f'活动期间有{shower}可能', 2: f'活动期间有{shower}，局地雨势较大'}[rain]
    if rain == 0 and thunder:
        rain_text += '，但大气不稳定，不排除局地雷阵雨'
    elif local_shower:
        rain_text += '，但不排除局地短时阵雨'
    storm_clause = thunder == 2 or (rain == 2 and thunder and cast_wind == 2)
    thunder_clause = '' if rain == 0 else {0: '', 1: f'{"阵雨" if showery else "降雨"}时可能伴有雷电。',
                                           2: '雷雨时可能伴有短时强降水、雷电和短时大风。'}[2 if storm_clause else thunder]
    humidity_text = '相对湿度90%以上' if min(humidity) >= 90 else f'相对湿度{number(min(humidity))}%～{number(max(humidity))}%'
    wind_text = f'{wind_dir}风{wind_level}级'
    if event_rows and '北' in event_rows[-1]['wind_dir'] and '北' not in wind_dir:
        wind_text += '，夜间逐渐转偏北风'
    if cast_gust_text:
        wind_text += '，' + cast_gust_text
    else:
        if gust_level:
            wind_text += f'，阵风{gust_level}级左右'
        if thunder:
            wind_text += '，雷雨时局地阵风较大'
    event_text = (f'{sky}，{rain_text}。{thunder_clause}气温{temperature_text(temps)}，{humidity_text}；'
                  f'{wind_text}；{visibility_text}。')
    risks = (('降雨', rain), ('雷电', thunder), ('大风', wind), ('闷热', heat))
    risk_line = '风险等级：' + '｜'.join(f'{name} {LEVELS[level]}' for name, level in risks) + \
                f'，{SUITABILITY[suitability]}开展灯光秀活动'
    if same_day:
        event_text += comparison(previous, risks, suitability, day)

    advice = []
    if suitability >= 3:
        advice.append('建议主办方做好活动延期或调整的预案，根据最新预警信号和现场天气及时决策。')
    if thunder:
        advice.append('密切关注雷电预警信号；如发布雷电预警或雷雨云临近，及时引导观众到室内或有防雷设施的场所避险。')
    if rain:
        advice.append('准备雨具，注意广场地面湿滑，降雨时做好观众疏散引导，防止拥挤踩踏。')
    if wind or thunder:
        advice.append('对围挡、隔离栏、指示牌等临时设施进行加固，注意高层建筑周边高空坠物。')
    if heat:
        advice.append('现场潮湿闷热，注意观众补水和防暑，特别关注老人和儿童。')
    if sight:
        advice.append('能见度偏低可能影响灯光观看效果和人员疏散，注意现场照明和交通安全。')
    if not advice:
        advice.append('天气条件总体有利，请关注最新天气预报。')

    lines = ['[福田区市民中心灯光秀气象研判]']
    if kind != 20:
        lines.append('[当前实况]' + current_conditions(kind, observation, warning, radar, lightning))
    if kind != 16:
        lines.append(f'[天气趋势]{today_trend(trend_text, day) if same_day else trend_text}')
    if periods:
        lines += ['[分时段预报]'] + [f'{name}：{text}，{temp}' for name, text, temp, _ in periods]
    lines += ['[活动时段（19:30—21:30）研判]', event_text, risk_line, '[建议]']
    lines += [f'{index}. {item}' for index, item in enumerate(advice, 1)]
    external = '\n'.join(lines)

    brief_level = wind_level.replace('～', '-')
    brief_wind = f'{wind_dir}风{brief_level}级'
    if event_rows and '北' in event_rows[-1]['wind_dir'] and '北' not in wind_dir:
        brief_wind = f'{wind_dir}风转偏北风{brief_level}级'
    if kind == 16:
        gust_start = max(18, min(21, issue.hour + (1 if issue.minute else 0))) if same_day else 18
    else:
        gust_start = max(0, hours_since(issue, day)) if same_day else 0
    brief_gust, brief_gust_top = day_gust_text(rows, gust_start, 22 if kind == 16 else 24)
    if cast_gust_text:
        brief_wind += '，' + cast_gust_text.replace('～', '-')
    else:
        if brief_gust:
            brief_wind += '，' + brief_gust
        if thunder:
            brief_wind += '，雷雨时局地阵风较大'
        elif not wind and brief_gust_top < 6:
            brief_wind += '，无明显大风'
    impact = brief_impact(rain, thunder, wind, heat, shower, ('、'.join(names) or '气象预警信号') if severe else '')
    sight_text = brief_visibility(haze_level, rain, fog)
    if kind == 16:
        brief_start = max(18, min(21, issue.hour + (1 if issue.minute else 0))) if same_day else 18
        window = [hours.get(float(hour)) for hour in range(brief_start, 23)]
        window_temps = [float(item[key]) for item in window for key in ('minT', 'maxT')] if all(window) else temps
        if rain == 2:
            weather = f'{sky}，有{shower}，局地雨势较大'
        elif rain == 1:
            weather = f'以{sky}为主，有短时{shower}'
        elif thunder:
            weather = f'以{sky}为主，不排除局地雷阵雨'
        elif any(row['rain_area_max_mm'] >= 1 for row in event_rows):
            weather = f'以{sky}为主，偶有短时阵雨或零星小雨'
        else:
            weather = f'以{sky}为主'
        brief = ('【市民中心区域夜间天气研判】\n'
                 f'预计{brief_start}-22时，市民中心区域{brief_tone(rain, thunder, wind, weather)}{weather}。'
                 f'{brief_temperature(window_temps)}，{brief_wind}。{sight_text}。{impact}')
    else:
        if daily:
            weather = daily['report']
            day_temperature = brief_temperature([float(daily['minT']), float(daily['maxT'])])
        else:
            weather, day_temperature = '（天气请人工补充）', '（气温请人工补充）'
        brief = ('【市民中心区域天气研判】\n'
                 f'预计{day.month}月{day.day}日全天市民中心区域{brief_tone(rain, thunder, wind, weather)}{weather}，'
                 f'{day_temperature}，{brief_wind}。18时至22时{sight_text}。{impact}')

    hourly = futian.get('hoursweather', [])
    if haze_level == '未读取':
        haze_note = '灰霾预报读取失败'
    elif haze_item:
        haze_note = f'市气象台灰霾预报（覆盖活动时刻）{haze_level}，{haze_item["ddatetime"]}发布，{haze_item["haze_des"]}'
    else:
        haze_note = '无覆盖活动时刻的灰霾预报，按无霾处理'
    pending = '；08时版雷电、暴雨、大雾/灰霾预警只写入实况，不自动上调，请人工判断' if kind == 8 and active else ''
    basis = ['## 内部依据（不对外）', '',
             f'- 生成时间：{now:%Y-%m-%d %H:%M}；研判时次：{issue:%Y-%m-%d %H:%M}（{kind}时版）；活动日期：{day:%Y-%m-%d}',
             f'- 市气象台天气提示：{tip["d"] + " " + tip["prompt"] if tip else "无（或超过36小时）"}',
             f'- {daily_source}：{daily["pubTime"] + " " + daily["report"] + "，" + str(daily["minT"]) + "～" + str(daily["maxT"]) + "℃，" + daily["wd"] + "风" + daily["ws"] + "级" if daily else "未找到活动日"}',
             f'- 福田区逐时预报：发布 {hourly[0]["pubTime"] if hourly else "无"}，覆盖至 {hourly[-1]["forecasttime"] if hourly else "无"}',
             f'- 预警（只取覆盖福田区的信号，按 subAlarm 的 alarmArea 判断）：{"预警资料缺测或过期，无法确认" if "error" in official["alarm"] else "生效中 " + alarm_text if active else "福田区无生效信号"}；'
             f'其他区域：{"、".join(other_warnings) if other_warnings else "无"}；最近一条：{alarm.get("alarmInfo", "")}；'
             f'subAlarm 原始值：{json.dumps(alarm.get("subAlarm") or [], ensure_ascii=False)}',
             f'- 福田区实况（取深圳国家基本气象站，位于福田区）原始值：{json.dumps(observation, ensure_ascii=False)}'
             '（24小时雨量取 h24r，与官网首页一致；按 GB/T 28592-2012 24小时降雨量等级描述）',
             f'- 短临/天气提示（活动时段）：{cast["time"] + " " + cast["title"] + "：" + cast["text"] if cast else ("无提到活动时段的当天提示" if same_day else "20时版不用")}',
             f'- 预警用于评级：{"；".join(applied) if applied else "无"}{pending}',
             f'- 雷达：{summary(radar) if kind == 16 else "本时次未读取"}',
             f'- 闪电定位：{sz_lightning.summary(lightning) if kind == 16 else "本时次未读取"}',
             f'- 前次研判：{previous[2] if previous else "未找到"}',
             f'- 能见度：{haze_note}；福田区实况能见度 {observation.get("v", "无")} 公里（{observation.get("date", "")}）',
             f'- 上下班提示：{(commute or {}).get("pubDate", "")} {(commute or {}).get("prompt", "")}',
             f'- 读取失败：{"；".join(failures) if failures else "无"}',
             '- 分时段来源：' + '；'.join(f'{name} {source}' for name, _, _, source in periods),
             f'- 活动时段依据：{event_source}',
             f'- EC 气温订正：{offset:+.1f}℃' + (f'（以官方 {offset_hour % 24:g} 时逐时预报对比）' if offset_hour is not None else '（无可对比时次，未订正）'),
             f'- 活动窗口：对流潜势(点)最大 {cape} J/kg；EC 阵风 市民中心 {gust} m/s（{gust_level}级），'
             f'周边最大 {gust_area} m/s（含海面，仅参考）；体感温度 {feel:.1f}℃',
             '- 适宜度：按“风险等级与适宜度判定标准.txt”评定', '',
             f'EC {ec["run_bjt"]}（北京时）起报，市民中心 {ec["site"]["lat"]}°N、{ec["site"]["lon"]}°E 双线性插值；周边为 22.25–22.75°N、113.75–114.5°E。', '',
             '| 时段 | 市民中心 3h 雨量 mm | 周边最大 mm | 周边≥1mm 比例 | 总云量 % | 对流潜势 J/kg（点/周边最大） | 阵风 m/s（点/周边最大） | 气温 ℃ | 湿度 % | 10m 风 |',
             '|---|---:|---:|---:|---:|---|---|---:|---:|---|']
    for row in rows:
        begin = at(row['valid_bjt']) - timedelta(hours=3)
        basis.append(f"| {begin:%m-%d %H}—{row['valid_bjt'][11:13]}时 | {row['rain_point_mm']:.1f} | {row['rain_area_max_mm']:.1f} | "
                     f"{row['rain_area_ge1_fraction']:.0%} | {row['tcc_percent']} | {row['mucape_point']} / {row['mucape_area_max']} | "
                     f"{row['gust_point_ms']} / {row['gust_area_max_ms']} | {row['t2m_c']} | {row['rh2m_percent']} | "
                     f"{row['wind_dir']} {row['wind_ms']} |")
    if 'error' in official['alarm']:
        brief += '目前福田区预警资料缺测或过期，无法确认生效信号，请人工核对。'
    markdown = ('# 灯光秀气象研判草稿\n\n> 草稿：发布前请与深圳市气象台最新预报、预警核对。\n\n```text\n'
                + external + '\n```\n\n## 简版\n\n```text\n' + brief + '\n```\n\n' + '\n'.join(basis) + '\n')
    checks = []
    if failures:
        checks.append('资料读取失败（' + '、'.join(key for key, value in official.items() if 'error' in value) + '）')
    if '人工' in external + brief:
        checks.append('稿中标注“人工补充”的内容')
    if kind == 16 and not sz_lightning.available(lightning):
        checks.append('闪电定位缺测（云端配置已禁用）' if (lightning or {}).get('disabled') else '闪电定位缺测（内网未连通）')
    if pending:
        checks.append('生效预警对活动时段的影响')
    if cast_notes:
        checks.append(f'按市气象台{cast["time"][11:]}{cast["title"]}上调了{"、".join(cast_notes)}，请与逐时预报和雷达核对')
    if other_warnings:
        checks.append('其他区域预警（未计入福田区）：' + '、'.join(other_warnings))
    if unknown:
        checks.append('预警信号名称和是否覆盖福田区')
    return external, brief, markdown, checks


def load_ec(day, issue, forced=None):
    begin = min(issue, day + timedelta(hours=EVENT[0]))
    if forced:
        return extract(datetime.strptime(forced, '%Y%m%d%H'), day, DATA_DIR / 'ec_event', begin=begin), None
    latest = latest_run(day)
    errors = []
    for back in range(3):
        run = latest - timedelta(hours=6 * back)
        try:
            ec = extract(run, day, DATA_DIR / 'ec_event', begin=begin)
        except Exception as error:
            errors.append(f'{run:%Y%m%d%H}Z {type(error).__name__}: {error}')
            continue
        return ec, (f'EC 最新起报读取失败，改用 {ec["run_bjt"]} 起报' if back else None)
    raise RuntimeError('EC 读取失败：' + '；'.join(errors))


def publication_checks(official, day, issue, radar=None, now=None):
    valid = validate_official(official, now)
    radar = validated_radar(radar, now)
    blockers = []
    if 'error' in valid['alarm']:
        blockers.append('预警资料无效：' + valid['alarm']['error'])
    data = {key: value.get('data') or {} for key, value in valid.items()}
    alarm = data['alarm']
    citywide = str(alarm.get('ingnalNum', '0')) not in ('0', '') or bool(alarm.get('subAlarm'))
    names, others = warning_names(alarm)
    if citywide and not names and not others:
        blockers.append('预警状态无法识别，请人工核对信号名称和覆盖区域')
    hourly = [item for item in data['futian'].get('hoursweather', [])
              if str(item.get('forecasttime', ''))[:10] == f'{day:%Y-%m-%d}']
    daily = any(str(item.get('reportTime', ''))[:10] == f'{day:%Y-%m-%d}' for item in data['tenday'].get('daynew', []))
    daily = daily or str((data['futian'].get('dayreport') or {}).get('forecasttime', ''))[:10] == f'{day:%Y-%m-%d}'
    if not hourly and not daily:
        blockers.append('缺少覆盖活动日的有效市气象台日预报和逐时预报')
    kind = edition(day, issue)
    if kind != 20 and 'error' in valid['observation']:
        blockers.append('当天实况资料无效：' + valid['observation']['error'])
    if kind == 16 and (not radar or 'error' in radar):
        blockers.append('雷达资料缺测或过期')
    return blockers


def save_briefing(output_dir, day, issue, external, brief, markdown, metadata):
    """Create an immutable draft version; never replace a previous publication."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f'灯光秀气象研判_{issue:%Y%m%d_%H%M}_活动{day:%m%d}'
    files = [(stem.with_suffix('.md'), markdown, 'utf-8'),
             (stem.with_suffix('.txt'), external + '\n', 'utf-8-sig'),
             (stem.with_name(stem.name + '_简版.txt'), brief + '\n', 'utf-8-sig'),
             (output_dir / 'briefing_result.json', json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', 'utf-8')]
    if any(path.exists() for path, _, _ in files):
        raise FileExistsError('稿件版本已存在；请指定新的唯一 --output-dir，禁止覆盖历史稿件')
    created = []
    try:
        for path, content, encoding in files:
            with path.open('x', encoding=encoding) as handle:
                created.append(path)
                handle.write(content)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return stem


def covered_nowcasts(official, now=None):
    tips = validate_official(official, now)['tips'].get('data') or {}
    items = [tips.get('important')] + list(tips.get('more') or [])
    return list(dict.fromkeys(item['d'] for item in items if item and item.get('d') and '短临' in str(item.get('prompt', ''))[:20]))


def wait_until(moment):
    remaining = (moment - datetime.now(BJT)).total_seconds()
    if remaining > 0:
        print(f'EC 已就绪，等待至 {moment:%H:%M} 读取最新资料并发送……', flush=True)
        time.sleep(remaining)


def notice(day, issue, external, checks, test, trigger=None):
    lines = [f'【{"测试｜" if test else ""}灯光秀研判草稿】{issue.day}日{issue:%H:%M}（活动日{day.month}月{day.day}日）'
             if issue.minute else f'【{"测试｜" if test else ""}灯光秀研判草稿】{issue.day}日{issue:%H}时（活动日{day.month}月{day.day}日）']
    match = RISK_LINE.search(external)
    if match:
        lines.append(f'结论：{match.group(5)}（降雨 {match.group(1)}｜雷电 {match.group(2)}｜大风 {match.group(3)}｜闷热 {match.group(4)}）')
    if trigger:
        lines.append(f'触发：{trigger}')
    lines.append('后面两条依次为详细版和简版正文，审核后可长按转发。')
    if checks:
        lines.append('发出前请核对：' + '；'.join(checks))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Event date in Beijing time, YYYY-MM-DD (default: derived from the issue time)')
    parser.add_argument('--issue', help="Briefing time in Beijing time, 'YYYY-MM-DD HH:MM' (default: nearest 08/16/20 slot)")
    parser.add_argument('--ec-run', help='EC run in UTC, YYYYMMDDHH (default: latest covering the event date)')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--history-dir', type=Path, help='Directory containing only fully sent historical briefings (default: output-dir)')
    parser.add_argument('--mode', choices=('scheduled', 'nowcast', 'manual'), default='manual')
    parser.add_argument('--require-publishable', action='store_true', help='Exit 2 after saving an unsafe draft, so runners cannot publish it')
    parser.add_argument('--no-lightning', dest='lightning_enabled', action='store_false', help='Explicitly mark lightning unavailable (cloud default)')
    parser.add_argument('--enable-lightning', dest='lightning_enabled', action='store_true', help='Opt in to a configured reachable intranet source')
    parser.set_defaults(lightning_enabled=os.environ.get('LIGHTSHOW_ENABLE_LIGHTNING', '').lower() in ('1', 'true', 'yes'))
    parser.add_argument('--push', action='store_true', help='Send the draft to WeChat through OpenClaw')
    parser.add_argument('--send-before', type=int, help='Wait until this many minutes before the issue time, then fetch live data')
    parser.add_argument('--test', action='store_true', help='Mark pushed messages as a test')
    parser.add_argument('--detail-only', action='store_true', help='Push only the detailed text as one message')
    parser.add_argument('--trigger', help='Reason shown in the WeChat notice, e.g. a new nowcast')
    args = parser.parse_args()
    current = datetime.now(BJT)
    issue = at(args.issue) if args.issue else nominal_issue(current) if args.mode == 'scheduled' else current.replace(second=0, microsecond=0)
    day = activity_day(issue, args.mode, args.date)
    print(f'{datetime.now(BJT):%Y-%m-%d %H:%M:%S} 开始：{issue:%Y-%m-%d %H:%M} 时次，活动日 {day:%Y-%m-%d}', flush=True)
    if args.send_before is not None and issue - timedelta(minutes=args.send_before) - datetime.now(BJT) > timedelta(minutes=45):
        print(f'距 {issue:%H:%M} 时次发送还早于45分钟，本次跳过（可能是开机后补跑）', flush=True)
        return
    try:
        ec, ec_note = load_ec(day, issue, args.ec_run)
        if args.send_before is not None:
            wait_until(issue - timedelta(minutes=args.send_before))
        official = fetch(DATA_DIR / 'sz_official' / datetime.now(BJT).strftime('%Y%m%d%H%M'))
        previous = previous_briefing(args.history_dir or args.output_dir, day, issue)
        radar = None
        lightning = None
        if edition(day, issue) == 16:
            try:
                radar = read_radar(now=datetime.now(BJT))
            except Exception as error:
                radar = {'error': f'{type(error).__name__}: {error}'}
            lightning = sz_lightning.read_lightning(enabled=args.lightning_enabled)
        actual_now = datetime.now(BJT)
        external, brief, markdown, checks = build(day, issue, ec, official, previous, radar, lightning, now=actual_now)
        if ec_note:
            checks.append(ec_note)
        blockers = publication_checks(official, day, issue, radar, actual_now)
        metadata = {'external': external, 'brief': brief, 'checks': checks, 'blocking_checks': blockers,
                    'event_date': f'{day:%Y-%m-%d}', 'issue': f'{issue:%Y-%m-%d %H:%M}',
                    'generated_at': actual_now.isoformat(), 'publication_allowed': not blockers,
                    'mode': args.mode, 'lightning_disabled': not args.lightning_enabled,
                    'covered_nowcasts': covered_nowcasts(official, actual_now)}
        stem = save_briefing(args.output_dir, day, issue, external, brief, markdown, metadata)
        print(markdown)
        print(f'已保存：{stem}.txt（详细版）、{stem.name}_简版.txt（简版）和 .md（含内部依据）', flush=True)
    except Exception as error:
        raise
    if blockers and (args.require_publishable or args.push):
        print('禁止自动发布：' + '；'.join(blockers), flush=True)
        raise SystemExit(2)
    if args.push:
        if args.detail_only:
            send(external)
        else:
            send(notice(day, issue, external, checks, args.test, args.trigger))
            send(external)
            send(brief)
        print(f'{datetime.now(BJT):%Y-%m-%d %H:%M:%S} 已推送到微信', flush=True)


if __name__ == '__main__':
    main()
