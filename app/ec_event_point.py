"""Extract ECMWF IFS open-data guidance near the Futian Civic Center."""
import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from eccodes import codes_get, codes_get_values, codes_grib_new_from_file, codes_release
from ecmwf.opendata import Client

BJT = timezone(timedelta(hours=8))
SITE_LAT, SITE_LON = 22.543, 114.059
AREA = (22.25, 22.75, 113.75, 114.5)
CROP = (22.0, 23.0, 113.5, 114.75)
SURFACE = ['tp', 'tcc', 'mucape', '10fg', '2t', '2d', '10u', '10v']
SOURCES = ('google', 'aws', 'azure', 'ecmwf')
INDEX_ROOTS = ('https://storage.googleapis.com/ecmwf-open-data',
               'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com',
               'https://ai4edataeuwest.blob.core.windows.net/ecmwf',
               'https://data.ecmwf.int/forecasts')
DIRECTIONS = ('北', '东北', '东', '东南', '南', '西南', '西', '西北')
BEAUFORT = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7)


def max_step(run):
    return 144 if run.hour in (0, 12) else 90


def event_steps(run, day, begin=None):
    origin = run.replace(tzinfo=timezone.utc)
    first = ((begin or day) - origin).total_seconds() / 3600
    last = (day - origin).total_seconds() / 3600 + 27
    first = max(3, int(first // 3) * 3)
    return [step for step in range(first, max_step(run) + 1, 3) if step <= last]


def available(run, step):
    name = f'{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/{run:%Y%m%d%H}0000-{step}h-oper-fc.index'
    for root in INDEX_ROOTS:
        try:
            url = f'{root}/{name}'
            response = requests.head(url, timeout=(5, 10), allow_redirects=True)
            if response.status_code == 403 and 'blob.core.windows.net' in root:
                # Azure now requires the public read SAS used by ecmwf-opendata.
                token_response = requests.get('https://planetarycomputer.microsoft.com/api/sas/v1/token/ai4edataeuwest/ecmwf', timeout=(5, 10))
                token_response.raise_for_status()
                token = token_response.json()['token']
                if not isinstance(token, str) or not token:
                    raise ValueError('Azure read token is unavailable')
                url += '?' + token
                response = requests.head(url, timeout=(5, 10), allow_redirects=True)
            if response.status_code == 200:
                return True
            # Some mirrors reject HEAD but allow the small text index via GET.
            if response.status_code in (403, 405):
                with requests.get(url, timeout=(5, 10), stream=True) as fallback:
                    if fallback.status_code == 200:
                        return True
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
    return False


def latest_run(day, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    run -= timedelta(hours=run.hour % 6)
    for _ in range(12):
        steps = event_steps(run, day)
        if steps and run.replace(tzinfo=timezone.utc) + timedelta(hours=steps[-1]) >= day + timedelta(hours=23) \
                and available(run, steps[-1]):
            return run
        run -= timedelta(hours=6)
    raise RuntimeError('No ECMWF IFS run covering the event day is available')


def retrieve(run, steps, target, validator=None, **request):
    temporary = target.with_name(target.name + '.part')
    errors = []
    for source in SOURCES:
        try:
            client = Client(source=source, model='ifs', resol='0p25', maximum_retries=2, retry_after=10)
            client.retrieve(date=int(run.strftime('%Y%m%d')), time=run.hour, stream='oper', type='fc',
                            step=steps, target=str(temporary), **request)
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise ValueError('EC mirror returned an empty download')
            if validator is not None:
                validator(temporary)
            os.replace(temporary, target)
            return
        except Exception as error:
            errors.append(f'{source}: {error}')
            temporary.unlink(missing_ok=True)
    raise RuntimeError('; '.join(errors))


def decode(path, run, fields):
    grid = None
    with open(path, 'rb') as handle:
        while (message := codes_grib_new_from_file(handle)) is not None:
            try:
                issued = datetime.strptime(f"{codes_get(message, 'dataDate')}{codes_get(message, 'dataTime'):04d}", '%Y%m%d%H%M')
                if issued != run:
                    raise ValueError(f'GRIB run {issued} does not match {run}')
                name = codes_get(message, 'shortName')
                if codes_get(message, 'typeOfLevel') == 'isobaricInhPa':
                    name = f"{name}{codes_get(message, 'level')}"
                ni, nj = codes_get(message, 'Ni'), codes_get(message, 'Nj')
                dx = codes_get(message, 'iDirectionIncrementInDegrees')
                dy = codes_get(message, 'jDirectionIncrementInDegrees')
                lons = codes_get(message, 'longitudeOfFirstGridPointInDegrees') + np.arange(ni) * dx * (-1 if codes_get(message, 'iScansNegatively') else 1)
                lons = (lons + 180) % 360 - 180
                lats = codes_get(message, 'latitudeOfFirstGridPointInDegrees') + np.arange(nj) * dy * (1 if codes_get(message, 'jScansPositively') else -1)
                iy = np.flatnonzero((lats >= CROP[0] - 1e-6) & (lats <= CROP[1] + 1e-6))
                ix = np.flatnonzero((lons >= CROP[2] - 1e-6) & (lons <= CROP[3] + 1e-6))
                iy, ix = iy[np.argsort(lats[iy])], ix[np.argsort(lons[ix])]
                values = codes_get_values(message).reshape(nj, ni)[np.ix_(iy, ix)].astype(float)
                fields.setdefault(int(codes_get(message, 'endStep')), {})[name] = values
                grid = (lats[iy], lons[ix])
            finally:
                codes_release(message)
    return grid


def bilinear(values, lats, lons):
    if len(lats) < 2 or len(lons) < 2 or not lats[0] <= SITE_LAT <= lats[-1] or not lons[0] <= SITE_LON <= lons[-1]:
        raise ValueError('EC grid does not enclose the Civic Center interpolation point')
    i = max(0, min(len(lats) - 2, int(np.searchsorted(lats, SITE_LAT)) - 1))
    j = max(0, min(len(lons) - 2, int(np.searchsorted(lons, SITE_LON)) - 1))
    ty = (SITE_LAT - lats[i]) / (lats[i + 1] - lats[i])
    tx = (SITE_LON - lons[j]) / (lons[j + 1] - lons[j])
    return float((1 - ty) * (1 - tx) * values[i, j] + (1 - ty) * tx * values[i, j + 1]
                 + ty * (1 - tx) * values[i + 1, j] + ty * tx * values[i + 1, j + 1])


def relative_humidity(t, td):
    vapour = lambda c: 6.112 * math.exp(17.62 * c / (243.12 + c))
    return 100 * vapour(td) / vapour(t)


def beaufort(speed):
    return sum(speed >= edge for edge in BEAUFORT)


def wind(u, v):
    speed = math.hypot(u, v)
    direction = (math.degrees(math.atan2(-u, -v)) + 360) % 360
    return DIRECTIONS[int((direction + 22.5) // 45) % 8], round(speed, 1), beaufort(speed)


def extract(run, day, output, begin=None):
    steps = event_steps(run, day, begin)
    if not steps:
        raise ValueError('EC run does not cover the event day')
    output.mkdir(parents=True, exist_ok=True)
    fields, grid = {}, None
    for path in sorted(output.glob(f'ec_{run:%Y%m%d%H}_*_sfc.grib2')):
        cached = {}
        try:
            candidate_grid = decode(path, run, cached)
        except Exception:
            # Keep the original cache for diagnosis, but redownload corrupt data.
            continue
        fields.update({step: dict(fields.get(step, {}), **values) for step, values in cached.items()})
        grid = candidate_grid or grid
    missing = [step for step in steps if not set(SURFACE) <= set(fields.get(step, {}))]
    if missing:
        target = output / f'ec_{run:%Y%m%d%H}_f{missing[0]:03d}-f{missing[-1]:03d}_sfc.grib2'
        def validate_download(path):
            candidate = {}
            candidate_grid = decode(path, run, candidate)
            if candidate_grid is None or any(not set(SURFACE) <= set(candidate.get(step, {})) for step in missing):
                raise ValueError('EC mirror returned incomplete fields')
            if any(not np.isfinite(candidate[step][name]).all() for step in missing for name in SURFACE):
                raise ValueError('EC mirror returned missing or non-finite values')
        retrieve(run, missing, target, validator=validate_download, param=SURFACE)
        grid = decode(target, run, fields)
    if any(not set(SURFACE) <= set(fields.get(step, {})) for step in steps):
        raise RuntimeError('EC fields are incomplete after download')
    if grid is None:
        raise RuntimeError('EC grid is missing after download')
    if any(not np.isfinite(fields[step][name]).all() for step in steps for name in SURFACE):
        raise RuntimeError('EC fields contain missing or non-finite values')
    lats, lons = grid
    box = np.ix_((lats >= AREA[0]) & (lats <= AREA[1]), (lons >= AREA[2]) & (lons <= AREA[3]))
    rows = []
    for step in steps:
        now = fields[step]
        if step == 3:
            before = np.zeros_like(now['tp'])
        elif step - 3 in fields:
            before = fields[step - 3]['tp']
        else:
            continue
        rain = np.maximum((now['tp'] - before) * 1000, 0)
        t, td = now['2t'] - 273.15, now['2d'] - 273.15
        direction, speed, level = wind(bilinear(now['10u'], lats, lons), bilinear(now['10v'], lats, lons))
        valid = run.replace(tzinfo=timezone.utc) + timedelta(hours=step)
        point_t, point_td = bilinear(t, lats, lons), bilinear(td, lats, lons)
        gust = bilinear(now['10fg'], lats, lons)
        rows.append({
            'step': step,
            'valid_bjt': f'{valid.astimezone(BJT):%Y-%m-%d %H:%M}',
            'period_bjt': f"{(valid - timedelta(hours=3)).astimezone(BJT):%H}-{valid.astimezone(BJT):%H}时",
            'rain_point_mm': round(bilinear(rain, lats, lons), 2),
            'rain_area_max_mm': round(float(rain[box].max()), 2),
            'rain_area_ge1_fraction': round(float((rain[box] >= 1).mean()), 2),
            'tcc_percent': round(100 * bilinear(now['tcc'], lats, lons)),
            'mucape_point': round(bilinear(now['mucape'], lats, lons)),
            'mucape_area_max': round(float(now['mucape'][box].max())),
            'gust_point_ms': round(gust, 1),
            'gust_area_max_ms': round(float(now['10fg'][box].max()), 1),
            't2m_c': round(point_t, 1),
            'td2m_c': round(point_td, 1),
            'rh2m_percent': round(relative_humidity(point_t, point_td)),
            'wind_dir': direction,
            'wind_ms': speed,
            'wind_level': level,
        })
    result = {'model': 'ECMWF IFS 0.25° open data', 'run_id': f'{run:%Y%m%d%H}',
              'run_utc': f'{run:%Y-%m-%d %H:00}',
              'run_bjt': f'{run.replace(tzinfo=timezone.utc).astimezone(BJT):%Y-%m-%d %H:00}',
              'date': f'{day:%Y-%m-%d}', 'site': {'name': '福田区市民中心', 'lat': SITE_LAT, 'lon': SITE_LON},
              'area_box': AREA, 'rows': rows}
    validate_event_coverage(result, day)
    (output / f'ec_{run:%Y%m%d%H}_{day:%Y%m%d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def validate_event_coverage(ec, day):
    """Reject incomplete guidance rather than silently assigning low risks."""
    required = {'rain_point_mm', 'rain_area_max_mm', 'rain_area_ge1_fraction', 'tcc_percent',
                'mucape_point', 'mucape_area_max', 'gust_point_ms', 'gust_area_max_ms',
                't2m_c', 'rh2m_percent', 'wind_ms', 'wind_level'}
    intervals = []
    for row in (ec or {}).get('rows') or []:
        missing = required - row.keys()
        if missing:
            raise ValueError('EC row is missing fields: ' + ', '.join(sorted(missing)))
        if not row.get('wind_dir'):
            raise ValueError('EC row is missing fields: wind_dir')
        try:
            invalid = any(not math.isfinite(float(row[name])) for name in required)
        except (ValueError, TypeError) as error:
            raise ValueError('EC row contains missing or non-numeric values') from error
        if invalid:
            raise ValueError('EC row contains missing or non-finite values')
        try:
            end = datetime.strptime(row['valid_bjt'], '%Y-%m-%d %H:%M').replace(tzinfo=BJT)
        except (KeyError, ValueError) as error:
            raise ValueError('EC row has an invalid valid_bjt time') from error
        intervals.append((end - timedelta(hours=3), end))
    cursor, finish = day + timedelta(hours=19.5), day + timedelta(hours=21.5)
    for start, end in sorted(intervals):
        if start <= cursor < end:
            cursor = max(cursor, end)
        if cursor >= finish:
            return
    raise ValueError('EC activity-window coverage is insufficient (requires 19:30–21:30 Beijing time)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, help='Event date in Beijing time, YYYY-MM-DD')
    parser.add_argument('--run', help='EC run in UTC, YYYYMMDDHH (default: latest covering the date)')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('LIGHTSHOW_DATA_DIR', str(Path(__file__).resolve().parent))) / 'ec_event')
    args = parser.parse_args()
    day = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=BJT)
    run = datetime.strptime(args.run, '%Y%m%d%H') if args.run else latest_run(day)
    print(json.dumps(extract(run, day, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
