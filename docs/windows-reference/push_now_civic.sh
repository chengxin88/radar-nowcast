#!/bin/bash
cd /mnt/c/Users/chengyang/Documents/Codex/2026-10-03/xi-w/work || exit 1
FORCE=1 ./push_q15.sh
f=$(ls -t ../outputs/市民中心15分钟研判_*.txt | head -1); echo; cat "$f"; echo; tail -2 logs/push_$(date +%Y%m).log
