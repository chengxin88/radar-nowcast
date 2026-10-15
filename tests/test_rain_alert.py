"""Synthetic radar metadata only; the alert module has no network capability."""
from datetime import datetime, timedelta
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import rain_alert as alert

START = datetime(2026, 10, 13, 13, 0, tzinfo=alert.BJT)


def metadata(minutes=0, *, rain=True, strong=False, reliable=True, arrival=None,
             strong_arrival=None, clear=False, **overrides):
    now = START + timedelta(minutes=minutes)
    status = "unknown" if rain is None else "strong" if strong else "rain" if rain else "clear"
    result = {
        "product": "radar_nowcast", "generated_at": now.isoformat(), "radar_time": now.isoformat(),
        "fresh": True, "publication_allowed": True, "forecast_reliable": reliable,
        "current": {"status": status, "rain": rain, "strong": strong, "coverage": True,
                    "observed_at": now.isoformat(), "rain_fraction": .2 if rain else 0,
                    "rain_pixels": 4 if rain else 0, "strong_fraction": .2 if strong else 0,
                    "strong_pixels": 4 if strong else 0},
        "arrival_minutes": arrival, "strong_arrival_minutes": strong_arrival,
        "clear": clear, "steps": [], "horizon_minutes": 120,
        "target": {"lat": 22.543, "lon": 114.059, "radius_km": 3, "name": "市民中心"},
    }
    result.update(overrides)
    return result


class RainAlertTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)

    def check(self, minutes, **kwargs):
        return alert.evaluate(metadata(minutes, **kwargs), self.db, now=START + timedelta(minutes=minutes))

    def confirmed(self, *, strong=False, reliable=True):
        self.check(0, strong=strong, reliable=reliable)
        return self.check(6, strong=strong, reliable=reliable)

    def mark(self, key, status, minutes=6, **kwargs):
        return alert.mark_delivery(self.db, key, status, now=START + timedelta(minutes=minutes), **kwargs)

    def test_two_new_frames_confirm_but_do_not_mark_delivery(self):
        first = self.check(0)
        self.assertTrue(first["candidate"])
        self.assertFalse(first["confirmed"])
        self.assertFalse(first["should_notify"])
        self.assertIsNone(first["message_key"])
        second = self.check(6)
        self.assertTrue(second["confirmed"])
        self.assertTrue(second["should_notify"])
        self.assertEqual(second["delivery_status"], "pending")
        row = self.db.execute("SELECT status,accepted_at FROM rain_alert_messages").fetchone()
        self.assertEqual(row, ("pending", None))

    def test_duplicate_frame_cannot_confirm_or_advance_streak(self):
        self.check(0)
        repeat = self.check(0)
        self.assertEqual(repeat["rain_streak"], 1)
        self.assertFalse(repeat["confirmed"])
        self.assertFalse(repeat["frame_advanced"])
        self.assertFalse(repeat["should_notify"])
        self.assertTrue(self.check(6)["confirmed"])

    def test_out_of_order_frame_is_ignored(self):
        first = self.check(6)
        older = self.check(0)
        self.assertEqual(older["reason"], "out_of_order_frame")
        self.assertEqual(older["rain_streak"], first["rain_streak"])
        self.assertFalse(older["should_notify"])
        self.assertTrue(self.check(12)["confirmed"])

    def test_same_episode_same_level_has_one_stable_message(self):
        result = self.confirmed()
        key = result["message_key"]
        self.mark(key, "sent")
        for minutes in (12, 18, 24, 30, 36, 42, 48, 54, 60):
            current = self.check(minutes)
            self.assertFalse(current["should_notify"])
            self.assertEqual(current["message_key"], key)
            self.assertEqual(current["event_id"], result["event_id"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_messages").fetchone()[0], 1)

    def test_strong_upgrade_needs_two_frames_and_can_bypass_cooldown_once(self):
        weak = self.confirmed()
        self.mark(weak["message_key"], "accepted")
        first_strong = self.check(12, strong=True)
        self.assertFalse(first_strong["should_notify"])
        second_strong = self.check(18, strong=True)
        self.assertTrue(second_strong["should_notify"])
        self.assertEqual(second_strong["reason"], "confirmed_upgrade")
        self.assertGreater(second_strong["cooldown_remaining_minutes"], 0)
        self.assertEqual(second_strong["level"], "strong")
        self.assertNotEqual(second_strong["message_key"], weak["message_key"])
        self.mark(second_strong["message_key"], "accepted", 18)
        for minutes in (24, 30, 54):
            self.assertFalse(self.check(minutes, strong=True)["should_notify"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_messages").fetchone()[0], 2)

    def test_initial_strong_process_does_not_also_send_lower_level(self):
        result = self.confirmed(strong=True)
        self.assertEqual(result["level"], "strong")
        self.assertTrue(result["should_notify"])
        self.mark(result["message_key"], "accepted")
        self.assertFalse(self.check(12, strong=False)["should_notify"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_messages").fetchone()[0], 1)

    def test_unknown_blocks_retry_and_cannot_be_bypassed_by_upgrade(self):
        weak = self.confirmed()
        self.mark(weak["message_key"], "sending")
        self.mark(weak["message_key"], "unknown")
        for minutes in (12, 18, 24):
            result = self.check(minutes, strong=True)
            self.assertFalse(result["should_notify"])
            self.assertEqual(result["message_key"], weak["message_key"])
            self.assertEqual(result["delivery_status"], "unknown")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_messages").fetchone()[0], 1)
        refused = self.mark(weak["message_key"], "failed", 24)
        self.assertFalse(refused["changed"])
        self.assertTrue(refused["manual_required"])
        self.assertFalse(self.check(30, strong=True)["should_notify"])

    def test_crash_during_sending_becomes_unknown(self):
        result = self.confirmed()
        self.mark(result["message_key"], "sending")
        restarted = self.check(12)
        self.assertFalse(restarted["should_notify"])
        self.assertEqual(restarted["delivery_status"], "unknown")

    def test_manual_reconciliation_can_unlock_a_real_upgrade(self):
        result = self.confirmed()
        self.mark(result["message_key"], "unknown")
        self.check(12, strong=True)
        self.check(18, strong=True)
        accepted = self.mark(result["message_key"], "accepted", 18, manual=True)
        self.assertTrue(accepted["changed"])
        upgraded = self.check(24, strong=True)
        self.assertTrue(upgraded["should_notify"])
        self.assertEqual(upgraded["level"], "strong")

    def test_permanent_failure_requires_manual_review(self):
        result = self.confirmed()
        self.mark(result["message_key"], "permanent_failure")
        retry = self.check(12)
        self.assertFalse(retry["should_notify"])
        self.assertEqual(retry["delivery_status"], "permanent_failure")
        self.assertFalse(self.mark(result["message_key"], "pending")["changed"])

    def test_known_rejection_reuses_same_key_even_on_same_frame(self):
        result = self.confirmed()
        self.mark(result["message_key"], "sending")
        self.mark(result["message_key"], "failed")
        repeat = self.check(6)
        self.assertTrue(repeat["should_notify"])
        self.assertEqual(repeat["message_key"], result["message_key"])
        self.assertEqual(repeat["rain_streak"], 2)
        self.assertEqual(repeat["text"], result["text"])

    def test_paused_delivery_does_not_consume_once_per_episode(self):
        result = self.confirmed()
        self.mark(result["message_key"], "paused")
        repeat = self.check(12)
        self.assertTrue(repeat["should_notify"])
        self.assertEqual(repeat["message_key"], result["message_key"])

    def test_accepted_is_terminal_and_idempotent_marks_do_not_extend_cooldown(self):
        result = self.confirmed()
        self.mark(result["message_key"], "accepted", 6)
        self.assertFalse(self.mark(result["message_key"], "failed", 12)["changed"])
        self.assertFalse(self.mark(result["message_key"], "accepted", 18)["changed"])
        when = self.db.execute("SELECT last_delivery_at FROM rain_alert_state").fetchone()[0]
        self.assertEqual(when, (START + timedelta(minutes=6)).isoformat())

    def test_clear_needs_two_frames_and_thirty_minutes_before_new_episode(self):
        event = self.confirmed()
        self.mark(event["message_key"], "accepted")
        first = self.check(12, rain=False, clear=True)
        second = self.check(18, rain=False, clear=True)
        self.assertTrue(first["episode_active"])
        self.assertTrue(second["episode_active"])
        for minutes in (24, 30, 36):
            self.check(minutes, rain=False, clear=True)
        done = self.check(42, rain=False, clear=True)
        self.assertFalse(done["episode_active"])
        self.assertFalse(done["should_notify"])
        self.assertFalse(self.check(48)["confirmed"])
        new = self.check(54)
        self.assertTrue(new["should_notify"])
        self.assertNotEqual(new["event_id"], event["event_id"])
        self.assertNotEqual(new["message_key"], event["message_key"])

    def test_unknown_never_clears_an_episode_and_breaks_clear_continuity(self):
        event = self.confirmed()
        self.mark(event["message_key"], "accepted")
        self.check(12, rain=False, clear=True)
        self.check(18, rain=False, clear=True)
        missing = self.check(24, rain=None, strong=None, reliable=False, clear=False)
        self.assertTrue(missing["episode_active"])
        self.assertFalse(missing["should_notify"])
        self.assertEqual(missing["clear_streak"], 0)
        self.check(30, rain=False, clear=True)
        self.check(36, rain=False, clear=True)
        not_yet = self.check(42, rain=False, clear=True)
        self.assertTrue(not_yet["episode_active"])
        self.check(48, rain=False, clear=True)
        self.check(54, rain=False, clear=True)
        self.assertFalse(self.check(60, rain=False, clear=True)["episode_active"])

    def test_unknown_breaks_pending_confirmation_without_claiming_clear(self):
        self.check(0)
        unknown = self.check(6, rain=None, strong=None, reliable=False)
        self.assertFalse(unknown["clear"])
        self.assertFalse(unknown["should_notify"])
        self.assertFalse(self.check(12)["confirmed"])
        self.assertTrue(self.check(18)["confirmed"])

    def test_same_clear_frame_cannot_supply_two_clear_confirmations(self):
        self.confirmed()
        first = self.check(12, rain=False, clear=True)
        repeated = self.check(12, rain=False, clear=True)
        self.assertEqual(first["clear_streak"], 1)
        self.assertEqual(repeated["clear_streak"], 1)
        self.assertTrue(repeated["episode_active"])

    def test_stale_and_unpublishable_data_do_not_trigger_or_clear(self):
        event = self.confirmed()
        for overrides in ({"radar_time": START.isoformat()}, {"fresh": False}, {"publication_allowed": False}):
            with self.subTest(overrides=overrides):
                result = alert.evaluate(metadata(48, rain=False, clear=True, **overrides), self.db,
                                        now=START + timedelta(minutes=48))
                self.assertFalse(result["should_notify"])
                self.assertTrue(result["episode_active"])
                self.assertEqual(result["event_id"], event["event_id"])

    def test_future_missing_or_naive_times_cannot_trigger(self):
        for overrides in ({"radar_time": (START + timedelta(minutes=1)).isoformat()},
                          {"radar_time": None}, {"generated_at": "bad"},
                          {"radar_time": "2026-10-13T13:00:00"}):
            with self.subTest(overrides=overrides):
                result = alert.evaluate(metadata(**overrides), self.db, now=START)
                self.assertFalse(result["should_notify"])
                self.assertFalse(result["candidate"])

    def test_current_echo_can_confirm_without_a_motion_estimate(self):
        result = self.confirmed(reliable=False)
        self.assertTrue(result["should_notify"])
        self.assertIn("13:06雷达帧", result["text"])
        self.assertIn("当地可能出现降水", result["text"])
        self.assertIn("不等同于地面雨量实测", result["text"])
        self.assertIn("不是正式气象预警", result["text"])
        for certainty in ("已经下雨", "确定雷雨", "暴雨", "发布预警"):
            self.assertNotIn(certainty, result["text"])

    def test_future_arrival_requires_motion_reliability_and_thirty_minute_window(self):
        for reliable, minutes, expected in ((True, 30, True), (True, 0, True), (True, 31, False),
                                             (False, 10, False), (True, -1, False), (True, float("nan"), False)):
            with self.subTest(reliable=reliable, arrival=minutes):
                db = sqlite3.connect(":memory:")
                try:
                    result = alert.evaluate(metadata(rain=False, reliable=reliable, arrival=minutes), db, now=START)
                    self.assertEqual(result["candidate"], expected)
                finally:
                    db.close()

    def test_generated_time_age_is_removed_from_arrival_and_past_arrival_is_not_observation(self):
        original = metadata(rain=False, arrival=10)
        result = alert.evaluate(original, self.db, now=START + timedelta(minutes=8))
        self.assertTrue(result["candidate"])
        past = alert.evaluate(original, self.db, now=START + timedelta(minutes=12))
        self.assertFalse(past["candidate"])
        self.assertFalse(past["clear"])

    def test_strong_future_arrival_also_requires_reliable_motion(self):
        result = self.check(0, rain=False, arrival=15, strong_arrival=20)
        self.assertEqual(result["level"], "strong")
        confirmed = self.check(6, rain=False, arrival=15, strong_arrival=20)
        self.assertEqual(confirmed["level"], "strong")
        self.assertTrue(confirmed["should_notify"])
        self.assertIn("13:26", confirmed["text"])
        db = sqlite3.connect(":memory:")
        try:
            unknown = alert.evaluate(metadata(rain=False, reliable=False, strong_arrival=5), db, now=START)
            self.assertFalse(unknown["candidate"])
        finally:
            db.close()

    def test_missing_arrival_or_unreliable_clear_is_not_clear(self):
        self.confirmed()
        for minutes in (12, 42, 72):
            result = self.check(minutes, rain=False, reliable=False, clear=True)
            self.assertFalse(result["clear"])
            self.assertTrue(result["episode_active"])
            self.assertFalse(result["should_notify"])

    def test_uncovered_target_cannot_confirm(self):
        data = metadata()
        data["current"]["coverage"] = False
        result = alert.evaluate(data, self.db, now=START)
        self.assertFalse(result["candidate"])
        self.assertFalse(result["should_notify"])

    def test_automation_off_does_not_mark_accepted(self):
        with patch.dict(os.environ, {"AUTO_PUSH_ENABLED": "0", "RAIN_ALERTS_ENABLED": "0"}):
            result = self.confirmed()
            later = self.check(12)
        self.assertTrue(result["should_notify"])
        self.assertTrue(later["should_notify"])
        self.assertEqual(later["message_key"], result["message_key"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_messages WHERE status='accepted'").fetchone()[0], 0)

    def test_distinct_targets_have_independent_episodes(self):
        first = self.confirmed()
        other = metadata(6)
        other["target"]["lat"] += .01
        self.assertFalse(alert.evaluate(other, self.db, now=START + timedelta(minutes=6))["confirmed"])
        other = metadata(12)
        other["target"]["lat"] += .01
        second = alert.evaluate(other, self.db, now=START + timedelta(minutes=12))
        self.assertTrue(second["confirmed"])
        self.assertNotEqual(second["event_id"], first["event_id"])

    def test_schema_does_not_overwrite_delivery_tables_or_commit_the_caller(self):
        self.db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY)")
        self.db.execute("INSERT INTO jobs VALUES ('uncommitted')")
        self.check(0)
        self.assertTrue(self.db.in_transaction)
        self.db.rollback()
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)

    def test_unknown_key_and_invalid_status_are_rejected(self):
        with self.assertRaises(KeyError):
            alert.mark_delivery(self.db, "absent", "accepted", now=START)
        with self.assertRaises(ValueError):
            alert.mark_delivery(self.db, "absent", "invented", now=START)

    def test_persistence_survives_reopening_the_shared_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "delivery.sqlite3"
            db = sqlite3.connect(path)
            alert.evaluate(metadata(0), db, now=START)
            confirmed = alert.evaluate(metadata(6), db, now=START + timedelta(minutes=6))
            alert.mark_delivery(db, confirmed["message_key"], "accepted", now=START + timedelta(minutes=6))
            db.close()
            db = sqlite3.connect(path)
            try:
                replay = alert.evaluate(metadata(12), db, now=START + timedelta(minutes=12))
                self.assertFalse(replay["should_notify"])
                self.assertEqual(replay["message_key"], confirmed["message_key"])
            finally:
                db.close()

    def test_gap_over_fifteen_minutes_breaks_unconfirmed_streak(self):
        self.check(0)
        after_gap = self.check(20)
        self.assertTrue(after_gap["frame_gap_reset"])
        self.assertFalse(after_gap["confirmed"])
        self.assertEqual(after_gap["rain_streak"], 1)
        self.assertTrue(self.check(26)["confirmed"])

    def test_same_frame_stale_or_unknown_breaks_unconfirmed_streak(self):
        self.check(0)
        same_unknown = self.check(0, rain=None, strong=None, reliable=False)
        self.assertEqual(same_unknown["rain_streak"], 0)
        self.assertFalse(self.check(6)["confirmed"])
        self.assertTrue(self.check(12)["confirmed"])
        old = metadata(12)
        stale = alert.evaluate(old, self.db, now=START + timedelta(minutes=33))
        self.assertEqual(stale["rain_streak"], 0)
        self.assertTrue(stale["episode_active"])
        self.assertFalse(self.check(36)["confirmed"])

    def test_same_frame_unknown_motion_breaks_forecast_only_confirmation(self):
        self.check(0, rain=False, arrival=15)
        unknown = self.check(0, rain=False, reliable=False)
        self.assertEqual(unknown["rain_streak"], 0)
        self.assertFalse(unknown["clear"])
        self.assertFalse(self.check(6, rain=False, arrival=15)["confirmed"])
        self.assertTrue(self.check(12, rain=False, arrival=15)["confirmed"])

    def test_clear_gap_does_not_close_episode_from_old_clear_streak(self):
        self.confirmed()
        self.check(12, rain=False, clear=True)
        self.check(18, rain=False, clear=True)
        after_gap = self.check(48, rain=False, clear=True)
        self.assertTrue(after_gap["episode_active"])
        self.assertEqual(after_gap["clear_streak"], 1)
        for minutes in (54, 60, 66, 72):
            self.check(minutes, rain=False, clear=True)
        self.assertFalse(self.check(78, rain=False, clear=True)["episode_active"])

    def test_environment_lead_time_and_confirmation_frames_are_used(self):
        with patch.dict(os.environ, {"RAIN_ALERT_LEAD_MINUTES": "45", "RAIN_ALERT_CONFIRM_FRAMES": "3"}):
            self.assertTrue(self.check(0, rain=False, arrival=40)["candidate"])
            self.assertFalse(self.check(6, rain=False, arrival=40)["confirmed"])
            result = self.check(12, rain=False, arrival=40)
            self.assertTrue(result["confirmed"])
            self.assertTrue(result["should_notify"])
            self.assertEqual(result["lead_minutes"], 45)
            self.assertEqual(result["confirm_frames"], 3)
            self.assertIn("3张不同", result["text"])

    def test_environment_cooldown_holds_a_new_episode_but_not_a_strong_upgrade(self):
        with patch.dict(os.environ, {"RAIN_ALERT_COOLDOWN_MINUTES": "60"}):
            original = self.confirmed()
            self.mark(original["message_key"], "accepted")
            for minutes in (12, 18, 24, 30, 36, 42):
                self.check(minutes, rain=False, clear=True)
            self.check(48)
            held = self.check(54)
            self.assertTrue(held["confirmed"])
            self.assertFalse(held["should_notify"])
            self.assertEqual(held["cooldown_remaining_minutes"], 12)
            self.assertEqual(held["reason"], "cooldown")
            self.check(60)
            self.assertTrue(self.check(66)["should_notify"])

    def test_invalid_configuration_fails_before_state_changes(self):
        for name, value in (("RAIN_ALERT_LEAD_MINUTES", "bad"), ("RAIN_ALERT_CONFIRM_FRAMES", "1"),
                            ("RAIN_ALERT_COOLDOWN_MINUTES", "-1")):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                with self.assertRaises(ValueError):
                    self.check(0)

    def test_explicit_observational_clear_without_motion_ends_episode_then_allows_new_rain(self):
        original = self.confirmed()
        self.mark(original["message_key"], "accepted")
        for minutes in (12, 18, 24, 30, 36, 42):
            result = self.check(minutes, rain=False, strong=False, reliable=False,
                                clear=False, observed_clear=True)
            self.assertTrue(result["clear"])
            self.assertEqual(result["clear_basis"], "observed")
            self.assertFalse(result["should_notify"])
        self.assertFalse(result["episode_active"])
        self.assertFalse(self.check(48)["confirmed"])
        new = self.check(54)
        self.assertTrue(new["confirmed"])
        self.assertTrue(new["should_notify"])
        self.assertNotEqual(new["event_id"], original["event_id"])

    def test_observational_clear_needs_distinct_new_frames_and_thirty_minutes(self):
        self.confirmed()
        first = self.check(12, rain=False, reliable=False, observed_clear=True)
        duplicate = self.check(12, rain=False, reliable=False, observed_clear=True)
        second = self.check(18, rain=False, reliable=False, observed_clear=True)
        self.assertEqual(first["clear_streak"], 1)
        self.assertEqual(duplicate["clear_streak"], 1)
        self.assertTrue(second["episode_active"])

    def test_unknown_motion_with_nearby_echo_without_observed_clear_cannot_end_episode(self):
        original = self.confirmed()
        for minutes in (12, 18, 24, 30, 36, 42, 48):
            result = self.check(minutes, rain=False, reliable=False, clear=False)
            self.assertFalse(result["clear"])
            self.assertTrue(result["episode_active"])
            self.assertEqual(result["event_id"], original["event_id"])
            self.assertFalse(result["should_notify"])

    def test_explicit_observed_clear_cannot_override_stale_unknown_or_uncovered_data(self):
        original = self.confirmed()
        cases = []
        data = metadata(48, rain=False, reliable=False, observed_clear=True, fresh=False)
        cases.append(data)
        data = metadata(48, rain=None, strong=None, reliable=False, observed_clear=True)
        cases.append(data)
        data = metadata(48, rain=False, reliable=False, observed_clear=True)
        data["current"]["coverage"] = False
        cases.append(data)
        data = metadata(48, rain=False, strong=None, reliable=False, observed_clear=True)
        cases.append(data)
        for data in cases:
            with self.subTest(status=data["current"]["status"]):
                result = alert.evaluate(data, self.db, now=START + timedelta(minutes=48))
                self.assertFalse(result["clear"])
                self.assertFalse(result["should_notify"])
                self.assertTrue(result["episode_active"])
                self.assertEqual(result["event_id"], original["event_id"])

    def test_region_switch_isolated_in_persistent_state_for_lat_lon_or_radius(self):
        target_a = {"name": "地区A", "lat": 22.543, "lon": 114.059, "radius_km": 3}
        for field, value in (("lat", 23.13), ("lon", 113.26), ("radius_km", 5)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "delivery.sqlite3"
                db = sqlite3.connect(path)
                try:
                    def evaluate_target(minutes, target):
                        return alert.evaluate(metadata(minutes, target=dict(target)), db,
                                              now=START + timedelta(minutes=minutes))
                    evaluate_target(0, target_a)
                    first = evaluate_target(6, target_a)
                    alert.mark_delivery(db, first["message_key"], "accepted", now=START + timedelta(minutes=6))
                    # Restart the connection as a later monitor using region B.
                    db.close()
                    db = sqlite3.connect(path)
                    target_b = dict(target_a, name="地区B")
                    target_b[field] = value
                    same_frame = evaluate_target(6, target_b)
                    self.assertTrue(same_frame["candidate"])
                    self.assertFalse(same_frame["confirmed"])
                    self.assertEqual(same_frame["rain_streak"], 1)
                    self.assertFalse(same_frame["should_notify"])
                    second = evaluate_target(12, target_b)
                    self.assertTrue(second["should_notify"])
                    self.assertNotEqual(second["event_id"], first["event_id"])
                    self.assertNotEqual(second["message_key"], first["message_key"])
                    self.assertIn("地区B", second["text"])
                    self.assertNotIn("地区A", second["text"])
                    alert.mark_delivery(db, second["message_key"], "accepted", now=START + timedelta(minutes=12))
                    back_to_a = evaluate_target(12, target_a)
                    self.assertFalse(back_to_a["should_notify"])
                    self.assertEqual(back_to_a["event_id"], first["event_id"])
                    self.assertEqual(back_to_a["message_key"], first["message_key"])
                    target_a_delivery = db.execute(
                        "SELECT s.last_delivery_at FROM rain_alert_state s JOIN rain_alert_events e ON e.target_key=s.target_key WHERE e.event_id=?",
                        (first["event_id"],)).fetchone()[0]
                    self.assertEqual(target_a_delivery, (START + timedelta(minutes=6)).isoformat())
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM rain_alert_messages WHERE status='accepted'").fetchone()[0], 2)
                finally:
                    db.close()

    def test_name_only_change_keeps_accepted_process_and_names_future_upgrade(self):
        original_target = {"name": "原地区名称", "lat": 23.13, "lon": 113.26, "radius_km": 4}
        renamed_target = dict(original_target, name="广州天河测试地区")
        self.check(0, target=original_target)
        original = self.check(6, target=original_target)
        self.mark(original["message_key"], "accepted")
        renamed = self.check(12, target=renamed_target)
        self.assertFalse(renamed["should_notify"])
        self.assertEqual(renamed["message_key"], original["message_key"])
        self.assertEqual(renamed["event_id"], original["event_id"])
        self.check(18, strong=True, target=renamed_target)
        upgrade = self.check(24, strong=True, target=renamed_target)
        self.assertTrue(upgrade["should_notify"])
        self.assertEqual(upgrade["event_id"], original["event_id"])
        self.assertIn("广州天河测试地区半径4公里", upgrade["text"])
        self.assertIn("当地气象台", upgrade["text"])
        for old_name in ("原地区名称", "市民中心", "深圳市气象台"):
            self.assertNotIn(old_name, upgrade["text"])
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM rain_alert_state").fetchone()[0], 1)

    def test_name_only_change_refreshes_unattempted_text_without_changing_key(self):
        original = self.confirmed()
        target = {"name": "广州越秀测试点", "lat": 22.543, "lon": 114.059, "radius_km": 3}
        renamed = self.check(12, target=target)
        self.assertTrue(renamed["should_notify"])
        self.assertEqual(renamed["event_id"], original["event_id"])
        self.assertEqual(renamed["message_key"], original["message_key"])
        self.assertIn("广州越秀测试点半径3公里", renamed["text"])
        self.assertNotIn("市民中心", renamed["text"])

    def test_future_arrival_text_uses_metadata_region_and_radius(self):
        target = {"name": "广州覆盖区测试点", "lat": 23.13, "lon": 113.26, "radius_km": 7}
        self.check(0, rain=False, arrival=10, strong_arrival=20, target=target)
        forecast = self.check(6, rain=False, arrival=10, strong_arrival=20, target=target)
        self.assertTrue(forecast["should_notify"])
        self.assertEqual(forecast["level"], "strong")
        self.assertIn("接近广州覆盖区测试点半径7公里", forecast["text"])
        self.assertIn("当地气象台", forecast["text"])
        for fixed_region in ("市民中心", "深圳市气象台"):
            self.assertNotIn(fixed_region, forecast["text"])


if __name__ == "__main__":
    unittest.main()
