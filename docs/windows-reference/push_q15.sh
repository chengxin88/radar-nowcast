#!/bin/bash
# 10月4日晚每15分钟发报：按16时版出稿，只推简版（市民中心区域夜间天气研判）一条。
export HTTPS_PROXY=http://127.0.0.1:7900 HTTP_PROXY=http://127.0.0.1:7900 https_proxy=http://127.0.0.1:7900 http_proxy=http://127.0.0.1:7900 NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8 no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8
cd /mnt/c/Users/chengyang/Documents/Codex/2026-10-03/xi-w/work || exit 1
mkdir -p logs
now_hm=$(date +%H%M)
if [ -z "$FORCE" ] && { [ "$(date +%F)" != "2026-10-04" ] || [ "$now_hm" -gt 2155 ]; }; then echo "超出发报时段，跳过"; exit 0; fi
exec 9>/tmp/push_q15.lock; flock -n 9 || { echo "上一次还在运行，跳过"; exit 0; }
issue="$(date '+%F %H:%M')"
log="logs/push_$(date +%Y%m).log"
echo "===== $(date '+%F %T') 每15分钟发报启动（时次 $issue，civic_q15）" >> "$log"
out="../outputs/市民中心15分钟研判_$(date '+%Y%m%d_%H%M').txt"
/usr/bin/timeout 150 /home/chengyang/.venvs/ec-gfs-server/bin/python civic_q15.py > "$out" 2>> "$log"
code=$?
cat "$out" >> "$log"
/home/chengyang/.venvs/ec-gfs-server/bin/python - "$out" "$code" "$issue" >> "$log" 2>&1 <<'EOF2'
import sys
from pathlib import Path
from wechat_push import send
out, code, issue = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = out.read_text(encoding='utf-8').strip() if out.exists() else ''
if code != '0' or not text:
    send(f'【市民中心研判】{issue[11:]} 出稿失败（退出码 {code}），请看 work/logs 推送日志。')
else:
    send(text)
    print('已推送')
EOF2
echo "===== $(date '+%F %T') 结束（出稿退出码 $code）" >> "$log"
