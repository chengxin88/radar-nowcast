"""Send text through a WeCom group webhook without exposing its secret.

Use split_text + send_part with the durable runner. Accepted receipts are LOCAL
references, not server message IDs or evidence that a member read the message.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import random
import shlex
import sqlite3
import time
import unicodedata
from urllib.parse import parse_qs, urlsplit
import uuid

import requests
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

MAX_TEXT_BYTES = 2048
RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 60.0
HTTP_TIMEOUT = (10, 30)
HERE = Path(__file__).resolve().parent


class SendError(RuntimeError):
    """A sanitized failure without a sensitive response or exception chain."""

    status = "failed"


class PermanentFailure(SendError):
    """Invalid configuration/content or a permanent API rejection."""


class RetryableFailure(SendError):
    """A known rejection, or failure before the connection sent the request."""


class DeliveryUnknown(SendError):
    """The message may already have arrived; never automatically resend."""

    status = "unknown"


class DeliveryPaused(SendError):
    """The caller disabled delivery before HTTP; this segment remains pending."""

    status = "paused"


def _file_settings() -> dict[str, str]:
    """Read dotenv-style data without sourcing or expanding shell code."""
    path = os.environ.get("LIGHTSHOW_ENV_FILE")
    if not path:
        return {}
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
        values = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, raw = line.partition("=")
            if not separator:
                continue
            if name.strip() not in {"WECHAT_WEBHOOK_URL", "LIGHTSHOW_STATE_DIR"}:
                continue
            tokens = shlex.split(raw, comments=True, posix=True)
            if len(tokens) != 1:
                raise ValueError("invalid value")
            values[name.strip()] = tokens[0]
        return values
    except (OSError, UnicodeError, ValueError):
        pass
    raise PermanentFailure("无法读取消息推送配置文件，或配置格式错误")


def _webhook_url(settings: dict[str, str] | None = None) -> str:
    settings = _file_settings() if settings is None else settings
    url = os.environ.get("WECHAT_WEBHOOK_URL") or settings.get("WECHAT_WEBHOOK_URL", "")
    try:
        parts = urlsplit(url)
        query = parse_qs(parts.query, keep_blank_values=True)
        valid = (
            parts.scheme == "https"
            and parts.hostname == "qyapi.weixin.qq.com"
            and parts.port in (None, 443)
            and parts.path == "/cgi-bin/webhook/send"
            and not parts.username and not parts.password and not parts.fragment
            and set(query) == {"key"} and len(query["key"]) == 1
            and bool(query["key"][0].strip())
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise PermanentFailure("未配置有效的企业微信群 HTTPS webhook")
    return url


def _clusters(text: str) -> list[str]:
    """Preserve code points, combining marks, flags and common emoji sequences."""
    groups = []
    current = ""
    for char in text:
        point = ord(char)
        regional = 0x1F1E6 <= point <= 0x1F1FF
        continuation = (
            char == "\u200d" or current.endswith("\u200d")
            or unicodedata.category(char).startswith("M")
            or 0x1F3FB <= point <= 0x1F3FF
            or 0xE0020 <= point <= 0xE007F
            or (regional and len(current) == 1 and 0x1F1E6 <= ord(current) <= 0x1F1FF)
            or (char == "\n" and current == "\r")
        )
        if current and not continuation:
            groups.append(current)
            current = ""
        current += char
    if current:
        groups.append(current)
    return groups


def _split_with_count(groups: list[str], count: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(groups):
        prefix = f"【{len(chunks) + 1}/{count}】 "
        capacity = MAX_TEXT_BYTES - len(prefix.encode("utf-8"))
        end = start
        size = 0
        boundaries = {"line": [], "sentence": [], "space": []}
        while end < len(groups):
            width = len(groups[end].encode("utf-8"))
            if size + width > capacity:
                break
            size += width
            end += 1
            if size >= capacity // 2:
                last = groups[end - 1]
                if "\n" in last:
                    boundaries["line"].append(end)
                elif last[-1] in "。！？；.!?;":
                    boundaries["sentence"].append(end)
                elif last[-1].isspace() or last[-1] in "，,、":
                    boundaries["space"].append(end)
        if end == start:
            raise PermanentFailure("单个字符组合超过文本字节上限，无法安全分段")
        if end < len(groups):
            for kind in ("line", "sentence", "space"):
                if boundaries[kind]:
                    end = boundaries[kind][-1]
                    break
        chunks.append("".join(groups[start:end]))
        start = end
    return chunks


def split_text(text: str) -> list[str]:
    """Split at UTF-8 2048 bytes, including the numbered multipart prefix.

    Prefer paragraph/sentence boundaries. Concatenating the parts after removing
    their 【i/n】 prefixes reproduces the original text exactly. One part has no
    prefix. Common emoji graphemes are kept whole, rather than just their bytes.
    """
    if not isinstance(text, str) or not text.strip():
        raise PermanentFailure("推送文本不能为空")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError:
        size = None
    if size is None:
        raise PermanentFailure("推送文本包含无效 Unicode 字符")
    if size <= MAX_TEXT_BYTES:
        return [text]
    groups = _clusters(text)
    count = 2
    while True:
        chunks = _split_with_count(groups, count)
        if len(chunks) == count:
            return [f"【{index}/{count}】 {chunk}" for index, chunk in enumerate(chunks, 1)]
        count = len(chunks)


def _acquire_slot(settings: dict[str, str]) -> None:
    """Reserve an attempt in a shared, persistent rolling 60-second window."""
    folder = Path(os.environ.get("LIGHTSHOW_STATE_DIR") or settings.get("LIGHTSHOW_STATE_DIR") or HERE.parent / "state")
    failed = False
    try:
        folder.mkdir(parents=True, exist_ok=True)
        database = folder / "wechat_rate_limit.sqlite3"
        while True:
            with closing(sqlite3.connect(database, timeout=30)) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS attempts (at REAL NOT NULL)")
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                connection.execute("DELETE FROM attempts WHERE at <= ?", (now - RATE_WINDOW_SECONDS,))
                rows = connection.execute("SELECT at FROM attempts ORDER BY at").fetchall()
                if len(rows) < RATE_LIMIT:
                    connection.execute("INSERT INTO attempts(at) VALUES (?)", (now,))
                    connection.commit()
                    return
                delay = max(0.1, rows[0][0] + RATE_WINDOW_SECONDS - now + 0.1)
                connection.commit()
            time.sleep(delay)
    except (OSError, sqlite3.Error):
        failed = True
    if failed:
        raise PermanentFailure("无法记录企业微信推送限速状态，已停止发送")


def _connection_was_not_opened(error: BaseException) -> bool:
    pending = [error]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (ConnectTimeoutError, NewConnectionError)):
            return True
        pending.extend(value for value in (
            getattr(item, "__cause__", None), getattr(item, "__context__", None),
            getattr(item, "reason", None), *getattr(item, "args", ()),
        ) if isinstance(value, BaseException))
    return False


def _request_once(url: str, text: str, settings: dict[str, str], before_request=None) -> str:
    _acquire_slot(settings)
    # The switch can change during a rate-limit wait or retry backoff. Check
    # again after waiting, immediately before each actual HTTP request.
    if before_request is not None and not before_request():
        raise DeliveryPaused("群推送已暂停，本段尚未发送")
    body = json.dumps({"msgtype": "text", "text": {"content": text}}, ensure_ascii=False).encode("utf-8")
    failure = None
    try:
        response = requests.post(
            url, data=body, headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=HTTP_TIMEOUT, allow_redirects=False,
        )
    except requests.exceptions.ConnectTimeout:
        failure = RetryableFailure("企业微信连接超时，请求尚未发送")
    except requests.exceptions.SSLError:
        failure = PermanentFailure("企业微信 TLS 验证失败")
    except requests.exceptions.ConnectionError as error:
        if _connection_was_not_opened(error):
            failure = RetryableFailure("企业微信连接失败，请求尚未发送")
        else:
            failure = DeliveryUnknown("企业微信连接中断，消息发送结果未知；已停止自动重发")
    except requests.exceptions.Timeout:
        failure = DeliveryUnknown("企业微信响应超时，消息发送结果未知；已停止自动重发")
    except requests.exceptions.RequestException:
        failure = DeliveryUnknown("企业微信请求异常，消息发送结果未知；已停止自动重发")
    if failure is not None:
        # Raise outside the except suite: no original exception holding the URL
        # becomes __context__, including in explicit traceback.format_exception.
        raise failure

    status = response.status_code
    if status == 429:
        raise RetryableFailure("企业微信 HTTP 限流，请求被拒绝")
    if 300 <= status < 500:
        raise PermanentFailure(f"企业微信 HTTP 拒绝请求（状态码 {status}）")
    try:
        reply = response.json()
    except (ValueError, requests.exceptions.RequestException):
        reply = None
    if not isinstance(reply, dict) or type(reply.get("errcode")) is not int:
        raise DeliveryUnknown("企业微信响应格式异常，消息发送结果未知；已停止自动重发")
    code = reply["errcode"]
    if code in (-1, 45009):
        raise RetryableFailure(f"企业微信拒绝消息，稍后可重试（错误码 {code}）")
    if code != 0:
        raise PermanentFailure(f"企业微信拒绝消息（错误码 {code}）")
    if not 200 <= status < 300:
        raise DeliveryUnknown(f"企业微信 HTTP 响应异常（状态码 {status}），消息发送结果未知")
    return f"accepted:{uuid.uuid4()}"


def send_part(text: str, attempts: int = 3, pause: float = 20, before_request=None) -> str:
    """Send one segment; ambiguous delivery is never automatically retried.

    Optional before_request() is called after rate waiting and before EVERY
    HTTP attempt. False raises DeliveryPaused without sending or retrying it.
    """
    if not isinstance(text, str) or not text.strip():
        raise PermanentFailure("推送文本不能为空")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError:
        size = None
    if size is None:
        raise PermanentFailure("推送文本包含无效 Unicode 字符")
    if size > MAX_TEXT_BYTES:
        raise PermanentFailure("单段文本超过2048个 UTF-8 字节")
    if type(attempts) is not int or not 1 <= attempts <= 3 or not isinstance(pause, (int, float)) or pause < 0:
        raise PermanentFailure("推送重试参数无效，最多允许3次尝试")
    settings = _file_settings()
    url = _webhook_url(settings)
    for attempt in range(attempts):
        try:
            return _request_once(url, text, settings, before_request=before_request)
        except RetryableFailure:
            if attempt + 1 == attempts:
                raise
            time.sleep(pause * (2 ** attempt) + random.uniform(0, 1))
    raise AssertionError("unreachable")


def send_messages(messages, callback=None, attempts: int = 3, pause: float = 20, before_request=None) -> list[dict]:
    """Send [(message_key, text), ...]; callback receives a single event dict.

    Keys: message_key, part_index (1-based), part_total, status, receipt, error.
    Status: sending / accepted / failed / unknown / paused. Stop at the first failure.
    The durable runner owns resumption and must never resend unknown segments.
    """
    prepared = [(key, split_text(text)) for key, text in messages]
    accepted = []
    for key, parts in prepared:
        for index, part in enumerate(parts, 1):
            event = {"message_key": key, "part_index": index, "part_total": len(parts),
                     "status": "sending", "receipt": None, "error": None}
            if callback:
                callback(dict(event))
            try:
                receipt = send_part(part, attempts=attempts, pause=pause, before_request=before_request)
            except SendError as error:
                event.update(status=error.status, error=str(error))
                if callback:
                    callback(dict(event))
                raise
            event.update(status="accepted", receipt=receipt)
            accepted.append(event)
            if callback:
                callback(dict(event))
    return accepted


def send(text: str, attempts: int = 3, pause: float = 20, before_request=None) -> str:
    """Compatibility API; split and send, returning a LOCAL batch reference."""
    send_messages([("message", text)], attempts=attempts, pause=pause, before_request=before_request)
    return f"batch:{uuid.uuid4()}"
