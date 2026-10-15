"""Radar nowcasts and rain-approach alerts with durable delivery state."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

BJT = timezone(timedelta(hours=8))
APP = Path(__file__).resolve().parent
HOME = Path(os.environ.get('LIGHTSHOW_HOME', APP.parent))
ENV_FILE = Path(os.environ.get('LIGHTSHOW_ENV_FILE', HOME / 'config.env'))
MESSAGE_FORMAT = 'notice-detail-brief-v1'
RAIN_MESSAGE_FORMAT = 'rain-alert-v1'


def config():
    values = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in values.items():
        os.environ.setdefault(key, value)
    os.environ['LIGHTSHOW_ENV_FILE'] = str(ENV_FILE)
    os.environ['LIGHTSHOW_DATA_DIR'] = str(HOME / 'cache')
    os.environ['RADAR_CACHE_DIR'] = str(HOME / 'cache' / 'radar')
    os.environ['LIGHTSHOW_OUTPUT_DIR'] = str(HOME / 'drafts')
    return values


def auto_enabled():
    # The on-disk switch must be enabled too; an inherited environment cannot bypass it.
    values = config()
    return values.get('AUTO_PUSH_ENABLED', '0') == '1' and os.environ.get('AUTO_PUSH_ENABLED', '0') == '1'


def alerts_enabled():
    values = config()
    return values.get('RAIN_ALERTS_ENABLED', '0') == '1' and os.environ.get('RAIN_ALERTS_ENABLED', '0') == '1'


def mode_enabled(mode):
    key = {'rolling': 'ROLLING_ENABLED', 'nowcast': 'OFFICIAL_TRIGGER_ENABLED', 'rain_alert': 'RAIN_ALERTS_ENABLED'}.get(mode)
    if key is None:
        return True
    values = config()
    return values.get(key, '0') == '1' and os.environ.get(key, '0') == '1'


def target_settings(settings=None):
    """Use one validated target snapshot for the generator and delivery gate."""
    from radar_nowcast import Config
    settings = config() if settings is None else settings
    try:
        selected = Config(name=settings.get('RADAR_TARGET_NAME', '市民中心'),
                          lat=float(settings.get('RADAR_TARGET_LAT', '22.543')),
                          lon=float(settings.get('RADAR_TARGET_LON', '114.059')),
                          radius_km=float(settings.get('RADAR_TARGET_RADIUS_KM', '3')))
        selected.validate()
    except (ValueError, TypeError) as error:
        raise ValueError('地区配置无效：' + str(error)) from None
    return {'name': selected.name, 'lat': selected.lat, 'lon': selected.lon,
            'radius_km': selected.radius_km}


def target_signature(target):
    normalized = {key: target[key] for key in ('name', 'lat', 'lon', 'radius_km')}
    for key in ('lat', 'lon', 'radius_km'):
        normalized[key] = float(normalized[key])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()[:16]


def target_matches(metadata):
    try:
        return target_signature(metadata['target']) == target_signature(target_settings())
    except (ValueError, TypeError, KeyError):
        return False


def database():
    (HOME / 'state').mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(HOME / 'state' / 'delivery.sqlite3', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, issue TEXT NOT NULL, event_date TEXT NOT NULL,
          mode TEXT NOT NULL, state TEXT NOT NULL, draft TEXT, created TEXT NOT NULL,
          error TEXT, notified INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS messages (
          job_id TEXT NOT NULL, part INTEGER NOT NULL, content TEXT NOT NULL,
          digest TEXT NOT NULL, state TEXT NOT NULL, receipt TEXT, error TEXT,
          PRIMARY KEY(job_id,part));
        CREATE TABLE IF NOT EXISTS nowcasts (
          stamp TEXT PRIMARY KEY, state TEXT NOT NULL, job_id TEXT);
    ''')
    db.commit()
    return db


@contextmanager
def lock():
    (HOME / 'state').mkdir(parents=True, exist_ok=True)
    with (HOME / 'state' / 'runner.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有出稿或发送任务运行，本次跳过')
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def issue_for(slot, date=None, mode='scheduled', explicit=None, now=None):
    now = now or datetime.now(BJT)
    if explicit:
        return datetime.strptime(explicit, '%Y-%m-%d %H:%M').replace(tzinfo=BJT)
    if mode != 'scheduled':
        return now.replace(second=0, microsecond=0)
    date = date or now.strftime('%Y-%m-%d')
    return datetime.strptime(date + ' ' + str(int(slot)).zfill(2) + ':00', '%Y-%m-%d %H:%M').replace(tzinfo=BJT)


def event_date(issue, mode):
    # Evening radar products always describe now, rather than the next activity day.
    return issue.strftime('%Y-%m-%d')


def within_send_window(issue, mode, now=None):
    now = now or datetime.now(BJT)
    due = issue - timedelta(minutes=30) if mode == 'scheduled' else issue
    return due <= now <= due + timedelta(minutes=int(os.environ.get('SEND_DEADLINE_MINUTES', '15')))


def job_key(issue, mode):
    return f'radar-{mode}:{issue:%Y%m%d%H%M}:{target_signature(target_settings())}'


def notice(metadata):
    if metadata.get('message_format') == MESSAGE_FORMAT:
        return message_texts(metadata)[0]
    checks = metadata.get('checks') or []
    text = (f'【雷达短临预报】{metadata["issue"]}\n'
            '未来0—2小时的降水回波外推，请结合最新雷达和实况核对。')
    if checks:
        text += '\n核对事项：' + '；'.join(checks)
    return text


def generate(issue, mode, dry_run=False, allow_blocked=False):
    settings = config()
    target = target_settings(settings)
    version = HOME / 'drafts' / f'{issue:%Y%m%d_%H%M}_{mode}_{uuid.uuid4().hex[:10]}'
    version.mkdir(parents=True)
    directory = version / 'product'
    # Actual wall-clock radar data determine the forecast, not the nominal slot.
    command = [sys.executable, str(APP / 'rolling_nowcast.py'), '--output-dir', str(directory),
               '--cache-dir',str(HOME/'cache'/'radar'),
               '--target-name='+target['name'], '--lat',str(target['lat']), '--lon',str(target['lon']),
               '--radius-km',str(target['radius_km']),
               '--rain-threshold',settings.get('RADAR_RAIN_DBZ','25'),
               '--strong-threshold',settings.get('RADAR_STRONG_DBZ','35'),
               '--support-fraction',settings.get('RADAR_SUPPORT_FRACTION','0.10'),
               '--support-pixels',settings.get('RADAR_SUPPORT_PIXELS','3')]
    timeout = int(os.environ.get('RADAR_GENERATION_TIMEOUT_SECONDS', '180'))
    with (version / 'generation.log').open('x') as log:
        try:
            result = subprocess.run(command, cwd=APP, stdout=log, stderr=subprocess.STDOUT,
                                    env=os.environ.copy(), timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError('出稿超过允许时长，已终止；未发送任何正文')
    metadata_file = directory / 'radar_result.json'
    if not metadata_file.is_file():
        raise RuntimeError(f'出稿失败（退出码{result.returncode}）；请查看生成日志')
    metadata = json.loads(metadata_file.read_text())
    metadata['_draft'] = str(directory)
    metadata['_generation_log']=str(version/'generation.log')
    if result.returncode != 0 or not metadata.get('publication_allowed', False):
        if allow_blocked:
            return metadata
        raise RuntimeError('资料未通过发布校验：' + '；'.join(metadata.get('blocking_checks') or ['出稿异常']))
    cleanup_cache()
    return metadata


def read_metadata(directory):
    directory = Path(directory)
    path = directory / 'radar_result.json'
    if not path.exists():
        path = directory / 'briefing_result.json'
    return json.loads(path.read_text())


def message_texts(metadata):
    if metadata.get('message_format') == MESSAGE_FORMAT:
        expected = [metadata.get('notice'), metadata.get('external'), metadata.get('brief')]
        if (metadata.get('messages') != expected or
                not all(isinstance(text, str) and text.strip() for text in expected)):
            raise ValueError('雷达推送必须包含按顺序排列的提醒、详细版和简版')
        return expected
    if metadata.get('message_format') == RAIN_MESSAGE_FORMAT:
        text = metadata.get('external')
        if not isinstance(text, str) or not text.strip() or metadata.get('messages') != [text]:
            raise ValueError('降水临近提醒必须是独立单条正文')
        return [text]
    return metadata.get('messages') or [metadata['external']]


def format_matches(metadata, mode):
    if metadata.get('product') != 'radar_nowcast':
        return True
    expected_format = RAIN_MESSAGE_FORMAT if mode == 'rain_alert' else MESSAGE_FORMAT
    if metadata.get('message_format') != expected_format:
        return False
    try:
        return len(message_texts(metadata)) == (1 if mode == 'rain_alert' else 3)
    except ValueError:
        return False


def ledger_matches(parts, metadata):
    from wechat_push import split_text
    expected = [part for text in message_texts(metadata) for part in split_text(text)]
    return (len(parts) == len(expected) and all(
        row['part'] == index and row['content'] == text and
        row['digest'] == hashlib.sha256(text.encode()).hexdigest()
        for index, (row, text) in enumerate(zip(parts, expected))))


def cleanup_cache():
    """Bound new cloud cache growth; never remove drafts or sent history."""
    cache = HOME / 'cache'
    if not cache.is_dir():
        return
    cutoff = time.time() - int(os.environ.get('CACHE_RETENTION_DAYS', '7')) * 86400
    for path in cache.rglob('*'):
        if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
            path.unlink()
    for path in sorted((p for p in cache.rglob('*') if p.is_dir() and not p.is_symlink()),
                       key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def prepare_messages(db, key, metadata):
    from wechat_push import split_text
    parts = [part for text in message_texts(metadata) for part in split_text(text)]
    if db.execute('SELECT 1 FROM messages WHERE job_id=?', (key,)).fetchone():
        return
    for index, text in enumerate(parts):
        db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?)',
                   (key, index, text, hashlib.sha256(text.encode()).hexdigest(), 'pending', None, None))
    db.execute('UPDATE jobs SET state=?,draft=? WHERE id=?', ('ready', metadata['_draft'], key))
    db.commit()


def deliver(db, key, send_part=None):
    from wechat_push import DeliveryPaused, DeliveryUnknown, PermanentFailure, RetryableFailure
    if send_part is None:
        from wechat_push import send_part
    if not auto_enabled():
        return 'disabled'
    job = db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone()
    metadata = read_metadata(job['draft'])
    key_matches_target = (job['mode'] == 'rain_alert' or
                          key == f'radar-{job["mode"]}:{datetime.fromisoformat(job["issue"]):%Y%m%d%H%M}:{target_signature(metadata["target"])}')
    if job['mode'] == 'nowcast':
        key_matches_target = key_matches_target or any(
            key == f'radar-nowcast-{stamp[-12:]}:{datetime.fromisoformat(job["issue"]):%Y%m%d%H%M}:{target_signature(metadata["target"])}'
            for stamp in metadata.get('covered_nowcasts', []) if stamp.startswith('official:'))
    can_send = lambda: (auto_enabled() and target_matches(metadata)
                       and key_matches_target
                       and format_matches(metadata, job['mode'])
                       and mode_enabled(job['mode'])
                       and (job['mode'] != 'rain_alert' or alerts_enabled()))
    if not can_send():
        return 'disabled'
    parts = db.execute('SELECT * FROM messages WHERE job_id=? ORDER BY part', (key,)).fetchall()
    if not parts:
        raise RuntimeError('没有待发送消息')
    if metadata.get('product') == 'radar_nowcast' and not ledger_matches(parts, metadata):
        db.execute("UPDATE jobs SET state='permanent_failure',error=? WHERE id=?",
                   ('投递清单与稿件不一致，需人工核对', key))
        db.commit()
        return 'permanent_failure'
    if any(row['state'] == 'permanent_failure' for row in parts):
        return 'permanent_failure'
    # A process crash after handing a message to HTTP is indistinguishable from delivery.
    if any(row['state'] in ('sending', 'unknown') for row in parts):
        db.execute("UPDATE messages SET state='unknown' WHERE job_id=? AND state='sending'", (key,))
        db.execute("UPDATE jobs SET state='unknown',error=? WHERE id=?", ('投递结果未知，自动补发已停止', key))
        db.commit()
        return 'unknown'
    issue = datetime.fromisoformat(job['issue'])
    if not within_send_window(issue, job['mode']):
        db.execute("UPDATE jobs SET state='expired',error=? WHERE id=?", ('已过发送截止时间', key))
        db.commit()
        return 'expired'
    if not metadata.get('publication_allowed', False):
        return 'permanent_failure'
    generated = datetime.fromisoformat(metadata['generated_at'])
    generated = generated.replace(tzinfo=BJT) if generated.tzinfo is None else generated
    age = datetime.now(BJT) - generated
    if age > timedelta(minutes=int(os.environ.get('DRAFT_MAX_AGE_MINUTES', '10'))) or age < timedelta(minutes=-5):
        db.execute("UPDATE jobs SET state='expired',error=? WHERE id=?", ('稿件已过期，禁止补发旧正文', key))
        db.commit()
        return 'expired'
    db.execute("UPDATE jobs SET state='sending' WHERE id=?", (key,))
    db.commit()
    for row in parts:
        if row['state'] == 'sent':
            continue
        if not can_send():
            db.execute("UPDATE jobs SET state='paused' WHERE id=?", (key,))
            db.commit()
            return 'disabled'
        db.execute("UPDATE messages SET state='sending' WHERE job_id=? AND part=?", (key, row['part']))
        db.commit()
        try:
            receipt = send_part(row['content'], before_request=can_send)
        except DeliveryPaused:
            db.execute("UPDATE messages SET state='pending',error=NULL WHERE job_id=? AND part=?", (key, row['part']))
            db.execute("UPDATE jobs SET state='paused' WHERE id=?", (key,))
            db.commit()
            return 'disabled'
        except DeliveryUnknown:
            db.execute("UPDATE messages SET state='unknown',error=? WHERE job_id=? AND part=?",
                       ('投递结果未知', key, row['part']))
            db.execute("UPDATE jobs SET state='unknown',error=? WHERE id=?", ('投递结果未知，禁止自动重发', key))
            db.commit()
            return 'unknown'
        except (PermanentFailure, RetryableFailure) as error:
            permanent = isinstance(error, PermanentFailure)
            db.execute("UPDATE messages SET state=?,error=? WHERE job_id=? AND part=?",
                       ('permanent_failure' if permanent else 'failed', type(error).__name__, key, row['part']))
            db.execute("UPDATE jobs SET state=?,error=? WHERE id=?",
                       ('permanent_failure' if permanent else 'partial', type(error).__name__, key))
            db.commit()
            return 'permanent_failure' if permanent else 'partial'
        except Exception as error:
            db.execute("UPDATE messages SET state='unknown',error=? WHERE job_id=? AND part=?",
                       (type(error).__name__, key, row['part']))
            db.execute("UPDATE jobs SET state='unknown',error=? WHERE id=?", (type(error).__name__, key))
            db.commit()
            return 'unknown'
        db.execute("UPDATE messages SET state='sent',receipt=?,error=NULL WHERE job_id=? AND part=?", (receipt, key, row['part']))
        db.commit()
    db.execute("UPDATE jobs SET state='sent',error=NULL WHERE id=?", (key,))
    db.commit()
    history = HOME / 'history' / key.replace(':', '_')
    history.mkdir(parents=True, exist_ok=True)
    for source in Path(job['draft']).iterdir():
        if source.suffix not in ('.txt', '.md'):
            continue
        target = history / source.name
        if not target.exists():
            shutil.copy2(source, target)
    for stamp in metadata.get('covered_nowcasts', []):
        existing = db.execute('SELECT state FROM nowcasts WHERE stamp=?', (stamp,)).fetchone()
        if not existing or existing['state'] not in ('unknown', 'permanent_failure'):
            db.execute('INSERT OR REPLACE INTO nowcasts VALUES (?,?,?)', (stamp, 'sent', key))
    db.commit()
    return 'sent'


def run(args, db):
    if not args.dry_run and (not auto_enabled() or not mode_enabled(args.mode)):
        print('自动群推送已关闭；没有出稿或发送。')
        return 0
    now = datetime.now(BJT)
    slot = args.slot or ('08' if now.hour < 12 else '16' if now.hour < 18 else '20')
    issue = issue_for(slot, args.date, args.mode, args.issue, now)
    if not args.dry_run:
        due = issue - timedelta(minutes=30) if args.mode == 'scheduled' else issue
        if now > due + timedelta(minutes=int(os.environ.get('SEND_DEADLINE_MINUTES', '15'))):
            print('已过发送截止时间，禁止补发旧时次。')
            return 2
        if now < due:
            print('未到雷达短临发报时间，本次跳过。')
            return 0
    key = job_key(issue, args.mode)
    if args.dry_run:
        metadata = generate(issue, args.mode, True)
        from wechat_push import split_text
        messages = [part for text in message_texts(metadata) for part in split_text(text)]
        print(json.dumps({'dry_run': True, 'draft': metadata['_draft'], 'issue': metadata['issue'],
                          'event_date': metadata['event_date'], 'product': metadata.get('product'),
                          'target': metadata.get('target'),
                          'message_format':metadata.get('message_format'),
                          'logical_messages':len(message_texts(metadata)),
                          'forecast_reliable': metadata.get('forecast_reliable'), 'publication_allowed': metadata['publication_allowed'],
                          'message_parts': len(messages), 'max_utf8_bytes': max(map(lambda s: len(s.encode()), messages))}, ensure_ascii=False))
        return 0
    job = db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone()
    if job and job['state'] in ('sent', 'unknown', 'permanent_failure', 'expired'):
        print(f'任务{key}状态为{job["state"]}，不重复发送。')
        return 0 if job['state'] == 'sent' else 2
    if not job:
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                   (key, issue.isoformat(), event_date(issue, args.mode), args.mode, 'generating', None, now.isoformat(), None, 0))
        db.commit()
    parts = db.execute('SELECT 1 FROM messages WHERE job_id=?', (key,)).fetchone()
    if not parts:
        try:
            metadata = generate(issue, args.mode)
            prepare_messages(db, key, metadata)
        except Exception as error:
            db.execute("UPDATE jobs SET state='failed',error=? WHERE id=?", (str(error), key))
            db.commit()
            notify_failure(db, key, str(error))
            raise
    outcome = deliver(db, key)
    if outcome not in ('sent', 'disabled'):
        notify_failure(db, key, outcome)
    print(json.dumps({'job': key, 'delivery': outcome}, ensure_ascii=False))
    return 0 if outcome in ('sent', 'disabled') else 2


def notify_failure(db, key, reason):
    if not auto_enabled():
        return
    job = db.execute('SELECT notified,mode FROM jobs WHERE id=?', (key,)).fetchone()
    if job and job['notified']:
        return
    # Do not retry an alert after an ambiguous result either.
    db.execute('UPDATE jobs SET notified=1 WHERE id=?', (key,))
    db.commit()
    from wechat_push import DeliveryPaused, send
    can_send=lambda: auto_enabled() and (not job or mode_enabled(job['mode'])) and (not job or job['mode']!='rain_alert' or alerts_enabled())
    try:
        send(f'【雷达短临运行告警】{key}\n本时次尚未成功发布。\n原因：{reason[:300]}\n请检查服务器日志；雷达缺测不作“无降水”处理。',
             before_request=can_send)
    except DeliveryPaused:
        db.execute('UPDATE jobs SET notified=0 WHERE id=?', (key,))
        db.commit()
    except Exception as error:
        print('告警通道异常：' + type(error).__name__)


def watchdog(args, db):
    if not auto_enabled():
        print('自动推送和值守已关闭；没有补发或告警。')
        return 0
    issue = issue_for(args.slot, args.date)
    key = job_key(issue, 'scheduled')
    job = db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone()
    if not job or job['state'] in ('failed', 'generating'):
        return run(argparse.Namespace(slot=args.slot, date=args.date, issue=None, mode='scheduled', dry_run=False), db)
    if job['state'] == 'sent':
        print('本时次全部消息已成功发送。')
        return 0
    if job['state'] in ('unknown', 'permanent_failure', 'expired'):
        notify_failure(db, key, job['error'] or job['state'])
        print('需要人工处理：' + job['state'])
        return 2
    outcome = deliver(db, key)
    if outcome != 'sent':
        notify_failure(db, key, outcome)
    print('值守结果：' + outcome)
    return 0 if outcome == 'sent' else 2


def alert_status(outcome):
    return {'sent':'accepted','disabled':'paused','partial':'failed','unknown':'unknown',
            'permanent_failure':'permanent_failure','expired':'permanent_failure'}.get(outcome,'unknown')


def send_rain_alert(db, decision, metadata):
    from rain_alert import mark_delivery
    key = decision['message_key']
    job = db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
    if job and job['state'] in ('sent','unknown','permanent_failure','expired'):
        outcome = job['state']
        mark_delivery(db,key,alert_status(outcome))
        return outcome
    if not auto_enabled() or not alerts_enabled():
        return 'disabled'
    now = datetime.now(BJT)
    if not job:
        directory=HOME/'drafts'/f'rain_alert_{uuid.uuid4().hex}'
        directory.mkdir(parents=True)
        alert_metadata=dict(metadata,external=decision['text'],brief=decision['text'],
                            messages=[decision['text']],message_format=RAIN_MESSAGE_FORMAT,_draft=str(directory))
        (directory/'radar_result.json').write_text(json.dumps(alert_metadata,ensure_ascii=False,indent=2))
        (directory/'降水临近提醒.txt').write_text(decision['text'])
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                   (key,now.isoformat(),now.strftime('%Y-%m-%d'),'rain_alert','ready',str(directory),now.isoformat(),None,0))
        db.commit()
        job=db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
    if not db.execute('SELECT 1 FROM messages WHERE job_id=?',(key,)).fetchone():
        # The process may have stopped after committing the job but before its
        # message ledger. Rebuild from that original immutable draft, never the
        # current decision or a newly generated version.
        persisted=read_metadata(job['draft'])
        persisted['_draft']=job['draft']
        try:
            prepare_messages(db,key,persisted)
        except BaseException:
            # prepare_messages commits only after every segment is inserted.
            # An error part way through preparation must leave no partial
            # ledger in this connection for a later retry to mistake as ready.
            db.rollback()
            raise
    mark_delivery(db,key,'sending')
    outcome=deliver(db,key)
    mark_delivery(db,key,alert_status(outcome))
    if outcome not in ('sent','disabled'):
        notify_failure(db,key,outcome)
    return outcome


def reconcile_alert_receipts(db):
    from rain_alert import init_schema,mark_delivery
    init_schema(db)
    rows=db.execute("SELECT message_key FROM rain_alert_messages WHERE status='sending'").fetchall()
    for row in rows:
        key=row['message_key']
        job=db.execute('SELECT state FROM jobs WHERE id=?',(key,)).fetchone()
        parts=db.execute('SELECT state FROM messages WHERE job_id=?',(key,)).fetchall()
        states=[part['state'] for part in parts]
        if states and all(state=='sent' for state in states):
            db.execute("UPDATE jobs SET state='sent',error=NULL WHERE id=?",(key,))
            db.commit()
            mark_delivery(db,key,'accepted')
        elif any(state in ('sending','unknown') for state in states):
            mark_delivery(db,key,'unknown')
        elif any(state=='permanent_failure' for state in states):
            mark_delivery(db,key,'permanent_failure')
        else:
            # Durable HTTP ledger says no request is in flight or ambiguous.
            mark_delivery(db,key,'failed' if any(state in ('sent','failed') for state in states) else 'paused')


def monitor(args, db):
    if not args.dry_run and (not auto_enabled() or not alerts_enabled()):
        print('降水临近自动提醒已关闭；没有检测任务或群推送。')
        return 0
    from rain_alert import evaluate
    now=datetime.now(BJT)
    metadata=generate(now,'radar_monitor',True,allow_blocked=True)
    if args.dry_run:
        # Preview against a copy; repeated previews cannot confirm an event or
        # consume the real notifier's counters / delivery state.
        preview=sqlite3.connect(':memory:')
        preview.row_factory=sqlite3.Row
        db.backup(preview)
        try:
            reconcile_alert_receipts(preview)
            decision=evaluate(metadata,preview,now=datetime.now(BJT))
        finally:
            preview.close()
        print(json.dumps({'dry_run':True,'draft':metadata['_draft'],
                          'forecast_reliable':metadata.get('forecast_reliable'),
                          'alert':decision,'group_message_sent':False},ensure_ascii=False))
        return 0
    reconcile_alert_receipts(db)
    decision=evaluate(metadata,db,now=datetime.now(BJT))
    if not decision.get('should_notify'):
        print(json.dumps({'alert':decision,'group_message_sent':False},ensure_ascii=False))
        return 0
    outcome=send_rain_alert(db,decision,metadata)
    print(json.dumps({'alert':decision,'delivery':outcome},ensure_ascii=False))
    return 0 if outcome in ('sent','disabled') else 2


def rolling(args, db):
    if not args.dry_run and (not auto_enabled() or not mode_enabled('rolling')):
        print('15分钟滚动推送已关闭；没有出稿或群推送。')
        return 0
    now = datetime.now(BJT)
    issue = now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)
    # The scheduled three-part briefing owns these slots. Failed scheduled jobs
    # use their own watchdog/ledger; never send another overlapping product.
    if not args.dry_run and issue.hour in (7, 15, 19) and issue.minute == 30:
        print('本时次由定时研判覆盖，滚动任务不重复发送。')
        return 0
    return run(argparse.Namespace(slot=None, date=None, issue=issue.strftime('%Y-%m-%d %H:%M'),
                                  mode='rolling', dry_run=args.dry_run), db)


def nowcast(args, db):
    if not args.dry_run and (not auto_enabled() or not mode_enabled('nowcast')):
        print('官方短临触发推送已关闭；没有取数或群推送。')
        return 0
    import rolling_official as official
    now = datetime.now(BJT)
    target = target_settings()
    profile = official.profile_for(target)
    if profile is None or 'tips' not in profile['feeds']:
        print('当前地区未配置匹配的官方短临来源。')
        return 0
    feed = official.sz_official.validate_feed('tips', official.read_feed('tips', profile['feeds']['tips']), now)
    item = official.recent_nowcast(feed, now)
    if item is None:
        print('未发现3小时内的有效官方短临预报。')
        return 0
    stamp = official.event_stamp(item, target)
    previous = db.execute('SELECT * FROM nowcasts WHERE stamp=?', (stamp,)).fetchone()
    if args.dry_run:
        print(json.dumps({'dry_run':True, 'official_nowcast':item, 'processed_state':previous['state'] if previous else None,
                          'group_message_sent':False}, ensure_ascii=False))
        return 0
    if previous and previous['state'] in ('sent', 'unknown', 'permanent_failure', 'expired'):
        print('此官方短临已处理或需人工核对，不重复推送。')
        return 0
    # Avoid overlapping the three daily editions without marking an unsent
    # forecast as covered. A later check verifies the real delivery receipt.
    for hour in (7, 15, 19):
        due = now.replace(hour=hour, minute=30, second=0, microsecond=0)
        if due - timedelta(minutes=20) <= now <= due + timedelta(minutes=5):
            print('临近定时研判，本次等待定时稿覆盖；尚未标记已发送。')
            return 0
    key = previous['job_id'] if previous else f'radar-nowcast-{stamp[-12:]}:{now:%Y%m%d%H%M}:{target_signature(target)}'
    if previous is None:
        db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                   (key, now.isoformat(), now.strftime('%Y-%m-%d'), 'nowcast', 'generating', None, now.isoformat(), None, 0))
        db.execute('INSERT INTO nowcasts VALUES (?,?,?)', (stamp, 'pending', key))
        db.commit()
    job = db.execute('SELECT * FROM jobs WHERE id=?', (key,)).fetchone()
    if job is None:
        raise RuntimeError('官方短临投递状态缺少任务记录，需要人工核对')
    if not db.execute('SELECT 1 FROM messages WHERE job_id=?', (key,)).fetchone():
        try:
            metadata = generate(datetime.fromisoformat(job['issue']), 'nowcast')
            if stamp not in metadata.get('covered_nowcasts', []):
                db.execute("UPDATE nowcasts SET state='expired' WHERE stamp=?", (stamp,))
                db.execute("UPDATE jobs SET state='expired' WHERE id=?", (key,))
                db.commit()
                print('出稿时官方短临已更新，本次旧触发取消，等待下一轮。')
                return 0
            prepare_messages(db, key, metadata)
        except Exception as error:
            db.execute("UPDATE jobs SET state='failed' WHERE id=?", (key,))
            db.commit()
            notify_failure(db, key, str(error))
            raise
    outcome = deliver(db, key)
    db.execute('UPDATE nowcasts SET state=? WHERE stamp=?', (outcome, stamp))
    db.commit()
    if outcome not in ('sent', 'disabled'):
        notify_failure(db, key, outcome)
    print(json.dumps({'official_nowcast':item['time'], 'delivery':outcome}, ensure_ascii=False))
    return 0 if outcome in ('sent', 'disabled') else 2


def main():
    config()
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--slot', choices=('08', '16', '20'))
    run_parser.add_argument('--date')
    run_parser.add_argument('--issue')
    run_parser.add_argument('--mode', choices=('scheduled', 'manual', 'nowcast', 'rolling'), default='scheduled')
    run_parser.add_argument('--dry-run', action='store_true')
    watch = commands.add_parser('watchdog')
    watch.add_argument('--slot', choices=('08', '16', '20'), required=True)
    watch.add_argument('--date')
    cast = commands.add_parser('nowcast')
    cast.add_argument('--dry-run', action='store_true')
    roll = commands.add_parser('rolling')
    roll.add_argument('--dry-run', action='store_true')
    monitor_parser = commands.add_parser('monitor')
    monitor_parser.add_argument('--dry-run', action='store_true')
    commands.add_parser('status')
    args = parser.parse_args()
    with lock():
        db = database()
        try:
            if args.command == 'status':
                print(json.dumps({'product':'radar_nowcast','auto_push_enabled': auto_enabled(),
                                  'rain_alerts_enabled':alerts_enabled(),'home': str(HOME),
                                  'rolling_enabled':mode_enabled('rolling'), 'official_trigger_enabled':mode_enabled('nowcast'),
                                  'message_format':MESSAGE_FORMAT,
                                  'target':target_settings(),
                                  'jobs': [dict(row) for row in db.execute('SELECT id,state,error FROM jobs ORDER BY created DESC LIMIT 10')]}, ensure_ascii=False))
                return 0
            return {'run': run, 'watchdog': watchdog, 'nowcast': nowcast,'monitor':monitor, 'rolling':rolling}[args.command](args, db)
        finally:
            db.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('任务失败：' + str(error), file=sys.stderr)
        sys.exit(1)
