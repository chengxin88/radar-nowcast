"""Enable or disable this project's cron block without touching other tasks."""
import argparse
import os
from pathlib import Path
import subprocess

HOME = Path(os.environ.get('LIGHTSHOW_HOME', Path(__file__).resolve().parents[1]))
BEGIN = '# BEGIN RADAR-NOWCAST'
END = '# END RADAR-NOWCAST'
LEGACY_BEGIN = '# BEGIN LIGHTSHOW-DUTY'
LEGACY_END = '# END LIGHTSHOW-DUTY'


def read_cron():
    result = subprocess.run(['crontab','-l'], capture_output=True, text=True)
    if result.returncode not in (0,1):
        raise RuntimeError('无法读取用户计划任务')
    return result.stdout if result.returncode == 0 else ''


def without_block(text):
    out, inside = [], False
    for line in text.splitlines():
        if line in (BEGIN,LEGACY_BEGIN):
            inside = True
        elif line in (END,LEGACY_END):
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise RuntimeError('计划任务块格式异常，未改动现有任务')
    return '\n'.join(out).rstrip()


def schedule_block():
    runner = str(HOME/'run.sh')
    log = str(HOME/'logs'/'scheduler.log')
    lines = [BEGIN, '# Server timezone: Asia/Shanghai; daily indefinitely.']
    for hour,slot in ((7,'08'),(15,'16'),(19,'20')):
        lines.append(f'30 {hour} * * * {runner} run --slot {slot} >> {log} 2>&1')
        lines.append(f'35,40,45 {hour} * * * {runner} watchdog --slot {slot} >> {log} 2>&1')
    settings=dict(line.split('=',1) for line in (HOME/'config.env').read_text().splitlines()
                  if '=' in line and not line.lstrip().startswith('#')) if (HOME/'config.env').is_file() else {}
    if settings.get('RAIN_ALERTS_ENABLED','0')=='1':
        lines.append(f'*/5 * * * * {runner} monitor >> {log} 2>&1')
    else:
        lines.append('# Rain-approach alerts are disabled; monitor would run every 5 minutes.')
    if settings.get('ROLLING_ENABLED','0')=='1':
        lines.append(f'*/15 * * * * {runner} rolling >> {log} 2>&1')
    if settings.get('OFFICIAL_TRIGGER_ENABLED','0')=='1':
        lines.append(f'*/5 * * * * {runner} nowcast >> {log} 2>&1')
    lines.append(END)
    return '\n'.join(lines)


def set_switch(enabled,key='AUTO_PUSH_ENABLED'):
    path=HOME/'config.env'
    if not path.is_file():
        raise RuntimeError('缺少私有配置文件')
    text=path.read_text()
    lines=[line for line in text.splitlines() if not line.startswith(key+'=')]
    lines.append(key+'='+('1' if enabled else '0'))
    temporary=path.with_suffix('.env.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.chmod(0o600)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('status','enable','disable','preview','enable-alerts','disable-alerts',
                                        'enable-rolling','disable-rolling','enable-official','disable-official'))
    args=parser.parse_args()
    existing=read_cron()
    if args.action=='status':
        print('managed_cron_installed='+str(BEGIN in existing or LEGACY_BEGIN in existing).lower())
        return
    if args.action=='preview':
        print(schedule_block())
        return
    (HOME/'logs').mkdir(parents=True,exist_ok=True)
    remaining=without_block(existing)
    if args.action=='disable':
        set_switch(False)
        set_switch(False,'RAIN_ALERTS_ENABLED')
        set_switch(False,'ROLLING_ENABLED')
        set_switch(False,'OFFICIAL_TRIGGER_ENABLED')
        if BEGIN in existing or LEGACY_BEGIN in existing:
            subprocess.run(['crontab','-'], input=remaining+'\n',text=True,check=True)
        print('所有自动群推送已关闭；本项目计划任务已移除。')
        return
    if args.action in ('enable-alerts','disable-alerts','enable-rolling','disable-rolling','enable-official','disable-official'):
        settings=dict(line.split('=',1) for line in (HOME/'config.env').read_text().splitlines()
                      if '=' in line and not line.startswith('#'))
        enabled=args.action.startswith('enable-')
        if enabled and settings.get('AUTO_PUSH_ENABLED','0')!='1':
            raise RuntimeError('正式群推送仍关闭，请先明确启用后再开启降水提醒')
        key={'alerts':'RAIN_ALERTS_ENABLED','rolling':'ROLLING_ENABLED','official':'OFFICIAL_TRIGGER_ENABLED'}[args.action.split('-')[1]]
        set_switch(enabled,key)
        if settings.get('AUTO_PUSH_ENABLED','0')=='1':
            subprocess.run(['crontab','-'],input=remaining+'\n'+schedule_block()+'\n',text=True,check=True)
        print(key+('已启用。' if enabled else '已关闭。'))
        return
    # Install the schedule while the switch is still off, then enable as the final step.
    set_switch(False)
    text=remaining+'\n'+schedule_block()+'\n'
    subprocess.run(['crontab','-'],input=text,text=True,check=True)
    set_switch(True)
    print('雷达短临已启用：07:30、15:30、19:30发布当前0—2小时预报；发布后5、10、15分钟值守。')


if __name__=='__main__':
    main()
