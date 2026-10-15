# 雷达外推短临预报

本仓库整理自阿里云已启用的 **`20261015-radar-brief-v5.13`** 发布包。导出时，服务器 `radar-nowcast/current` 指向该发布包；其雷达短临、监测调度与业务简报共用同一套代码。

本次完成源码版本管理，包含应用、测试、模板、配置示例和原有说明。服务器私有配置、企业微信机器人凭据、状态数据库、雷达缓存和历史出稿留在运行环境中。

## 主要模块

| 位置 | 用途 |
| --- | --- |
| [`app/rolling_nowcast.py`](app/rolling_nowcast.py) | 未来两小时雷达滚动研判与提醒、详细版、简版出稿 |
| [`app/radar_nowcast.py`](app/radar_nowcast.py) | 雷达帧处理、运动估计与圆域短临外推 |
| [`app/rolling_radar.py`](app/rolling_radar.py) | 上游走廊、降水到达和移出判断 |
| [`app/radar_monitor.py`](app/radar_monitor.py) | 回波演变与监测特征 |
| [`app/radar_echo_filter.py`](app/radar_echo_filter.py) | 回波识别及过滤 |
| [`app/rolling_official.py`](app/rolling_official.py) | 官方预报、实况、预警和短临资料配对 |
| [`app/yjt_observation.py`](app/yjt_observation.py) | 匹配站点的实况接入 |
| [`app/cloud_runner.py`](app/cloud_runner.py) | 调度、出稿、持久化投递与去重 |
| [`app/manage_schedule.py`](app/manage_schedule.py) | 项目 cron 配置管理 |
| [`app/wechat_push.py`](app/wechat_push.py) | 消息分段、限流与投递状态 |
| [`tests/`](tests/) | 雷达、区域配置、资料时次、调度与投递测试 |
| [`templates/`](templates/) | 业务研判模板及判定标准 |

原始功能说明见 [`雷达短临预报说明.txt`](雷达短临预报说明.txt)。其中部分章节标题保留早期版本号；当前发布版本以 [`deployment-manifest.json`](deployment-manifest.json) 为准。

## 环境与安装

需要 Linux/WSL 和 Python 3.10 或以上版本。锁定依赖来自阿里云发布包，包含 NumPy、Pillow、Requests、ecCodes 与 ECMWF Open Data 客户端。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
cp config.example.env config.env
chmod 600 config.env
chmod +x run.sh schedule.sh
```

接口凭据在本机 `config.env` 配置。示例中的自动推送和各功能开关均为关闭状态。地区设置与官方资料映射见 [`official-profile.example.json`](official-profile.example.json) 和原始功能说明。

## 本地入口

仓库根目录的 `run.sh`、`schedule.sh` 适用于直接克隆后的布局，默认使用仓库 `.venv`，也可通过 `PYTHON_BIN` 指定解释器。现有阿里云 `current/app/` 布局的原启动器保留在 `deploy/`。

```bash
./run.sh --help
./run.sh status
./run.sh rolling --dry-run
```

`rolling --dry-run` 会请求实际气象资料并生成本地预览稿。资料可能受源端可用性和更新时间影响。`status` 查看本地任务状态；计划任务管理和实际群推送需在部署时另外配置。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests
```

测试使用模拟资料及 HTTP 模拟，覆盖雷达外推、走廊判断、资料时效、地区映射、消息分段、开关、去重和投递恢复等行为。

## 部署与源码来源

- [阿里云导出与部署说明](docs/github-export.md)
- [逐文件来源、原始校验值和脱敏记录](docs/aliyun-source-manifest.json)
- [原发布包校验清单](docs/aliyun-original-sha256sums.txt)

本仓库为私有项目。本次没有切换服务器的 `current` 链接、修改线上配置、调整计划任务或发送群消息。

雷达外推的基本假设、资料适用范围和预报不确定性见原始功能说明。测试通过代表程序行为验证，预报效果需要结合实际天气另行检验。本次整理未新增源码许可声明。
