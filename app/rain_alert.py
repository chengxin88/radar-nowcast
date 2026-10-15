"""Radar arrival decisions and episode deduplication; this module never sends.

evaluate() may recommend a notification, but that is not authorization to send.
The caller still checks AUTO_PUSH_ENABLED and RAIN_ALERTS_ENABLED. Confirmation
and delivery are separate: use mark_delivery before and after the HTTP attempt.
Unknown delivery locks the episode until explicitly reconciled by a person.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import sqlite3

BJT = timezone(timedelta(hours=8))
ALERT_WINDOW_MINUTES = 30
COOLDOWN_MINUTES = 30
CLEAR_MINUTES = 30
MAX_AGE_MINUTES = 20
MAX_FRAME_GAP_MINUTES = 15
LEVEL_NAMES = {0: None, 1: "rain", 2: "strong"}


def _settings():
    def integer(name, default, minimum, maximum):
        try:
            value = int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            value = None
        if value is None or not minimum <= value <= maximum:
            raise ValueError(name + "配置无效")
        return value

    return {
        "lead_minutes": integer("RAIN_ALERT_LEAD_MINUTES", ALERT_WINDOW_MINUTES, 1, 120),
        "confirm_frames": integer("RAIN_ALERT_CONFIRM_FRAMES", 2, 2, 20),
        "cooldown_minutes": integer("RAIN_ALERT_COOLDOWN_MINUTES", COOLDOWN_MINUTES, 0, 1440),
    }


def init_schema(db: sqlite3.Connection) -> None:
    """Create independent tables without committing the caller's transaction."""
    statements = (
        """CREATE TABLE IF NOT EXISTS rain_alert_state (
             target_key TEXT PRIMARY KEY, last_frame TEXT, rain_streak INTEGER NOT NULL DEFAULT 0,
             strong_streak INTEGER NOT NULL DEFAULT 0, clear_streak INTEGER NOT NULL DEFAULT 0,
             clear_since TEXT, last_threat_at TEXT, active_event_id TEXT,
             last_delivery_at TEXT, updated_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS rain_alert_events (
             event_id TEXT PRIMARY KEY, target_key TEXT NOT NULL, started_at TEXT NOT NULL,
             started_frame TEXT NOT NULL, ended_at TEXT, confirmed_level INTEGER NOT NULL DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS rain_alert_messages (
             message_key TEXT PRIMARY KEY, event_id TEXT NOT NULL, level INTEGER NOT NULL,
             status TEXT NOT NULL, text TEXT NOT NULL, radar_time TEXT NOT NULL,
             created_at TEXT NOT NULL, updated_at TEXT NOT NULL, attempted_at TEXT,
             accepted_at TEXT, UNIQUE(event_id, level))""",
    )
    for statement in statements:
        db.execute(statement)


@contextmanager
def _transaction(db):
    db.execute("SAVEPOINT rain_alert_change")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK TO SAVEPOINT rain_alert_change")
        db.execute("RELEASE SAVEPOINT rain_alert_change")
        raise
    else:
        db.execute("RELEASE SAVEPOINT rain_alert_change")


def _one(db, sql, parameters=()):
    cursor = db.execute(sql, parameters)
    row = cursor.fetchone()
    return dict(zip((column[0] for column in cursor.description), row)) if row is not None else None


def _time(value):
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment.astimezone(BJT) if moment.tzinfo is not None else None


def _number(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _target(metadata):
    target = metadata.get("target")
    if not isinstance(target, dict):
        return None, None
    latitude, longitude, radius = (_number(target.get(field)) for field in ("lat", "lon", "radius_km"))
    if latitude is None or longitude is None or radius is None or not (-90 <= latitude <= 90 and -180 <= longitude <= 180 and 0 < radius <= 500):
        return None, None
    normalized = {"lat": latitude, "lon": longitude, "radius_km": radius}
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(raw.encode()).hexdigest()[:16]
    name = target.get("name") or target.get("label") or metadata.get("target_name") or "目标区域"
    normalized["name"] = str(name)[:40]
    return key, normalized


def _classify(metadata, now, lead_minutes):
    """Trust supported echo flags, but independently reject unusable timings."""
    frame, generated = _time(metadata.get("radar_time")), _time(metadata.get("generated_at"))
    base = {"frame": frame, "generated": generated, "level": 0, "clear": False,
            "clear_basis": None, "arrival": None, "strong_arrival": None, "observed": False}
    if frame is None or generated is None:
        return {**base, "quality": "unknown", "reason": "missing_or_invalid_time"}
    if frame > now or generated > now + timedelta(minutes=5):
        return {**base, "quality": "unknown", "reason": "future_radar_or_generation"}
    if now - frame > timedelta(minutes=MAX_AGE_MINUTES) or now - generated > timedelta(minutes=MAX_AGE_MINUTES):
        return {**base, "quality": "unknown", "reason": "stale"}
    if metadata.get("fresh") is not True or metadata.get("publication_allowed") is not True:
        return {**base, "quality": "unknown", "reason": "not_publishable"}
    current = metadata.get("current")
    if not isinstance(current, dict) or current.get("coverage") is not True:
        return {**base, "quality": "unknown", "reason": "current_coverage_unknown"}
    rain, strong = current.get("rain"), current.get("strong")
    if type(rain) is not bool or (strong is not None and type(strong) is not bool):
        return {**base, "quality": "unknown", "reason": "current_echo_unknown"}
    if (strong is True and rain is not True) or (current.get("status") == "clear" and rain):
        return {**base, "quality": "unknown", "reason": "inconsistent_current_echo"}
    age = max(0.0, (now - generated).total_seconds() / 60)
    horizon = _number(metadata.get("horizon_minutes"))
    reliable = metadata.get("forecast_reliable") is True and horizon is not None
    # Explicit upstream observational clear requires a fresh, fully covered
    # target AND no supported echoes within 100 km. It is not a future forecast.
    observed_clear = metadata.get("observed_clear") is True and rain is False and strong is False
    if rain is False and strong is not True and not reliable and not observed_clear:
        return {**base, "quality": "unknown", "reason": "forecast_unknown_without_current_echo"}

    def arrival(field):
        minutes = _number(metadata.get(field))
        if not reliable or minutes is None or not 0 <= minutes <= horizon:
            return None
        remaining = minutes - age
        return remaining if 0 <= remaining <= lead_minutes else None

    imminent, strong_imminent = arrival("arrival_minutes"), arrival("strong_arrival_minutes")
    level = 2 if strong is True or strong_imminent is not None else 1 if rain or imminent is not None else 0
    # An explicit reliable full-horizon clear is required; missing arrival is not clear.
    forecast_clear = (metadata.get("clear") is True and reliable and horizon - age >= lead_minutes
                      and rain is False and strong is not True and imminent is None and strong_imminent is None)
    clear = forecast_clear or (observed_clear and level == 0)
    return {**base, "quality": "usable", "reason": "candidate" if level else "clear" if clear else "no_confirmed_clear",
            "level": level, "clear": clear, "arrival": imminent, "strong_arrival": strong_imminent,
            "observed": strong is True if level == 2 else rain is True,
            "clear_basis": "forecast" if forecast_clear else "observed" if clear else None}


def _clock_text(moment, reference):
    return ("次日" if moment.date() > reference.date() else "") + moment.strftime("%H:%M")


def _text(metadata, target, classification, level, now, confirm_frames):
    frame = classification["frame"]
    title = "较强降水回波临近提醒" if level == 2 else "降水回波临近提醒"
    echo = "较强降水回波" if level == 2 else "降水回波"
    observed = classification["observed"] if level == classification["level"] else metadata["current"].get("rain") is True
    if observed:
        situation = f"{frame:%m月%d日%H:%M}雷达帧显示，{target['name']}半径{target['radius_km']:g}公里范围内已有{echo}，当地可能出现降水，请核对实况。"
    else:
        remaining = classification["strong_arrival"] if level == 2 else classification["arrival"]
        estimated = now + timedelta(minutes=remaining or 0)
        situation = (f"依据{frame:%m月%d日%H:%M}雷达帧，{echo}可能在约{_clock_text(estimated, now)}"
                     f"（约{math.ceil(remaining or 0)}分钟后）接近{target['name']}半径{target['radius_km']:g}公里范围。")
    frames = "两" if confirm_frames == 2 else str(confirm_frames)
    return (f"【雷达{title}】\n{situation}\n已由{frames}张不同的新雷达帧确认回波趋势。"
            "雷达回波及外推不等同于地面雨量实测，位置和强度仍可能变化。"
            "本提醒不是正式气象预警，请关注当地气象台最新预报、预警。")


def _base_result(classification, state=None, event=None):
    state = state or {}
    return {
        "candidate": classification["level"] > 0, "confirmed": False, "should_notify": False,
        "level": LEVEL_NAMES[classification["level"]], "severity": classification["level"],
        "event_id": state.get("active_event_id"), "message_key": None, "text": None,
        "delivery_status": None, "reason": classification["reason"], "frame_advanced": False,
        "clear": classification["clear"], "clear_basis": classification["clear_basis"],
        "episode_active": bool(state.get("active_event_id")),
        "episode_confirmed_level": (event or {}).get("confirmed_level", 0),
        "rain_streak": state.get("rain_streak", 0), "strong_streak": state.get("strong_streak", 0),
        "clear_streak": state.get("clear_streak", 0), "cooldown_remaining_minutes": 0.0,
        "radar_time": classification["frame"].isoformat() if classification["frame"] else None,
    }


def evaluate(metadata, db: sqlite3.Connection, now=None) -> dict:
    """Persist only radar confirmation and deduplication; never perform HTTP.

    Two distinct increasing usable frames confirm each level. Unknown new data
    breaks unconfirmed streaks but never closes a confirmed episode. Closing
    requires two new clear frames AND 30 minutes of continuously reliable clear.
    A stronger confirmed level may bypass cooldown once; same-level messages
    retain the same key. The caller must separately record delivery outcomes.
    """
    now = _time(now) if now is not None else datetime.now(BJT)
    if now is None:
        raise ValueError("now must include a timezone")
    if not isinstance(metadata, dict):
        metadata = {}
    settings = _settings()
    classification = _classify(metadata, now, settings["lead_minutes"])
    key, target = _target(metadata)
    if key is None:
        result = _base_result(classification)
        result.update(candidate=False, severity=0, level=None, reason="invalid_target")
        return result
    init_schema(db)
    with _transaction(db):
        # This write also serializes evaluations across callers sharing SQLite.
        db.execute("INSERT OR IGNORE INTO rain_alert_state(target_key) VALUES (?)", (key,))
        state = _one(db, "SELECT * FROM rain_alert_state WHERE target_key=?", (key,))
        frame, previous = classification["frame"], _time(state["last_frame"])
        ordered = frame is not None and (previous is None or frame > previous)
        older = frame is not None and previous is not None and frame < previous
        duplicate = frame is not None and previous is not None and frame == previous
        if classification["quality"] != "usable":
            # Even the same stale/unknown frame breaks pending confirmation.
            # It is never evidence of clear and never closes an active episode.
            state.update(rain_streak=0, strong_streak=0, clear_streak=0, clear_since=None)
            # Future/stale frames cannot move the monotonic cursor forward.
            if ordered and classification["reason"] not in ("stale", "future_radar_or_generation"):
                state["last_frame"] = frame.isoformat()
            _save_state(db, state, now)
            return _base_result(classification, state)
        if older:
            result = _base_result(classification, state)
            result.update(candidate=False, severity=0, level=None, reason="out_of_order_frame")
            return result
        gap_reset = ordered and previous is not None and frame - previous > timedelta(minutes=MAX_FRAME_GAP_MINUTES)
        if ordered:
            if gap_reset:
                state.update(rain_streak=0, strong_streak=0, clear_streak=0, clear_since=None)
            state["last_frame"] = frame.isoformat()
            if classification["level"]:
                state["rain_streak"] += 1
                state["strong_streak"] = state["strong_streak"] + 1 if classification["level"] == 2 else 0
                state.update(clear_streak=0, clear_since=None, last_threat_at=now.isoformat())
            elif classification["clear"]:
                state.update(rain_streak=0, strong_streak=0, clear_streak=state["clear_streak"] + 1)
                state["clear_since"] = state["clear_since"] or now.isoformat()
                if (state["active_event_id"] and state["clear_streak"] >= settings["confirm_frames"]
                        and now - _time(state["clear_since"]) >= timedelta(minutes=CLEAR_MINUTES)):
                    db.execute("UPDATE rain_alert_events SET ended_at=? WHERE event_id=?", (now.isoformat(), state["active_event_id"]))
                    state["active_event_id"] = None
            else:
                state.update(rain_streak=0, strong_streak=0, clear_streak=0, clear_since=None)
            _save_state(db, state, now)
        current_level = (2 if classification["level"] == 2 and state["strong_streak"] >= settings["confirm_frames"]
                         else 1 if classification["level"] and state["rain_streak"] >= settings["confirm_frames"] else 0)
        if current_level and not state["active_event_id"]:
            event_id = f"{key}:{frame:%Y%m%dT%H%M%S%z}"
            db.execute("INSERT INTO rain_alert_events VALUES (?,?,?,?,?,?)", (event_id, key, now.isoformat(), frame.isoformat(), None, current_level))
            state["active_event_id"] = event_id
            _save_state(db, state, now)
        event = _one(db, "SELECT * FROM rain_alert_events WHERE event_id=?", (state["active_event_id"],))
        if event and current_level > event["confirmed_level"]:
            db.execute("UPDATE rain_alert_events SET confirmed_level=? WHERE event_id=?", (current_level, event["event_id"]))
            event["confirmed_level"] = current_level
        result = _base_result(classification, state, event)
        result.update(confirmed=bool(current_level), severity=current_level or classification["level"],
                      level=LEVEL_NAMES[current_level or classification["level"]], frame_advanced=ordered,
                      frame_gap_reset=gap_reset, **settings)
        if not event or not current_level:
            if duplicate:
                result["reason"] = "duplicate_frame"
            return result
        # A crash while sending is delivery-unknown, not an invitation to resend.
        db.execute("UPDATE rain_alert_messages SET status='unknown' WHERE event_id=? AND status='sending'", (event["event_id"],))
        blocked = _one(db, "SELECT * FROM rain_alert_messages WHERE event_id=? AND status IN ('unknown','permanent_failure') ORDER BY level DESC LIMIT 1", (event["event_id"],))
        if blocked:
            result.update(message_key=blocked["message_key"], text=blocked["text"],
                          delivery_status=blocked["status"], reason="delivery_requires_manual_review")
            return result
        accepted = db.execute("SELECT COALESCE(MAX(level),0) FROM rain_alert_messages WHERE event_id=? AND status='accepted'", (event["event_id"],)).fetchone()[0]
        if accepted >= current_level:
            message = _one(db, "SELECT * FROM rain_alert_messages WHERE event_id=? AND status='accepted' ORDER BY level DESC LIMIT 1", (event["event_id"],))
            result.update(message_key=message["message_key"], text=message["text"], delivery_status="accepted", reason="already_notified")
            return result
        message_key = f"rain:{event['event_id']}:L{current_level}"
        message = _one(db, "SELECT * FROM rain_alert_messages WHERE message_key=?", (message_key,))
        if message is None:
            text = _text(metadata, target, classification, current_level, now, settings["confirm_frames"])
            db.execute("INSERT INTO rain_alert_messages VALUES (?,?,?,?,?,?,?,?,?,?)", (message_key, event["event_id"], current_level, "pending", text, frame.isoformat(), now.isoformat(), now.isoformat(), None, None))
            message = _one(db, "SELECT * FROM rain_alert_messages WHERE message_key=?", (message_key,))
        elif message["status"] in ("pending", "paused") and message["attempted_at"] is None:
            # With automation off, retain the stable key but refresh an unqueued
            # message's reference time. Once attempted, its text is immutable.
            text = _text(metadata, target, classification, current_level, now, settings["confirm_frames"])
            db.execute("UPDATE rain_alert_messages SET text=?,radar_time=?,updated_at=? WHERE message_key=?", (text, frame.isoformat(), now.isoformat(), message_key))
            message.update(text=text, radar_time=frame.isoformat())
        last_delivery = _time(state["last_delivery_at"])
        remaining = max(0.0, settings["cooldown_minutes"] - (now - last_delivery).total_seconds() / 60) if last_delivery else 0.0
        upgrade = current_level == 2 and accepted == 1
        result.update(message_key=message_key, text=message["text"], delivery_status=message["status"],
                      should_notify=remaining == 0 or upgrade, cooldown_remaining_minutes=remaining,
                      reason="confirmed_upgrade" if upgrade else "cooldown" if remaining else "confirmed")
        return result


def _save_state(db, state, now):
    fields = ("last_frame", "rain_streak", "strong_streak", "clear_streak", "clear_since",
              "last_threat_at", "active_event_id", "last_delivery_at")
    db.execute("UPDATE rain_alert_state SET " + ",".join(field + "=?" for field in fields) + ",updated_at=? WHERE target_key=?",
               tuple(state[field] for field in fields) + (now.isoformat(), state["target_key"]))


def mark_delivery(db: sqlite3.Connection, message_key: str, status: str, now=None, *, manual=False) -> dict:
    """Record transport results independently from meteorological confirmation.

    Call 'sending' BEFORE HTTP, then accepted / failed / paused / unknown /
    permanent_failure afterwards. 'sent', 'partial', 'disabled' are aliases.
    Unknown/permanent results require manual=True before changing status.
    Accepted messages cannot be downgraded; repeated calls do not renew cooldown.
    """
    aliases = {"sent": "accepted", "partial": "failed", "disabled": "paused", "ready": "pending"}
    status = aliases.get(status, status)
    if status not in {"pending", "sending", "accepted", "failed", "paused", "unknown", "permanent_failure"}:
        raise ValueError("invalid delivery status")
    now = _time(now) if now is not None else datetime.now(BJT)
    if now is None:
        raise ValueError("now must include a timezone")
    init_schema(db)
    with _transaction(db):
        row = _one(db, "SELECT * FROM rain_alert_messages WHERE message_key=?", (message_key,))
        if row is None:
            raise KeyError("unknown rain-alert message key")
        if row["status"] == "accepted" or (row["status"] in ("unknown", "permanent_failure") and status != row["status"] and not manual):
            return {**row, "changed": False, "manual_required": row["status"] != "accepted"}
        attempted = row["attempted_at"]
        if status in ("sending", "accepted", "failed", "unknown", "permanent_failure"):
            attempted = attempted or now.isoformat()
        accepted_at = now.isoformat() if status == "accepted" else row["accepted_at"]
        db.execute("UPDATE rain_alert_messages SET status=?,updated_at=?,attempted_at=?,accepted_at=? WHERE message_key=?", (status, now.isoformat(), attempted, accepted_at, message_key))
        if status == "accepted":
            event = _one(db, "SELECT target_key FROM rain_alert_events WHERE event_id=?", (row["event_id"],))
            db.execute("UPDATE rain_alert_state SET last_delivery_at=? WHERE target_key=?", (now.isoformat(), event["target_key"]))
        return {**row, "status": status, "updated_at": now.isoformat(), "attempted_at": attempted,
                "accepted_at": accepted_at, "changed": True, "manual_required": False}
