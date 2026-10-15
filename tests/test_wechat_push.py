"""All HTTP is mocked: this suite must never send a group message."""
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

import requests
from urllib3.exceptions import MaxRetryError, NewConnectionError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import wechat_push as push

FAKE_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=test-not-a-real-secret"


def response(code=0, status=200, **fields):
    result = Mock(status_code=status)
    result.json.return_value = {"errcode": code, "errmsg": "ok", **fields}
    return result


def bodies(parts):
    return [re.sub(r"^【\d+/\d+】 ", "", part) for part in parts]


class SplitTests(unittest.TestCase):
    def test_exact_utf8_limit_is_one_part(self):
        text = "雨" * 682 + "ab"
        self.assertEqual(len(text.encode()), 2048)
        self.assertEqual(push.split_text(text), [text])

    def test_byte_limit_and_prefix_at_ten_part_boundary(self):
        for text in ("雨" * 683, "气象研判🌧️" * 1400, "a" * 21000):
            with self.subTest(length=len(text)):
                parts = push.split_text(text)
                self.assertGreater(len(parts), 1)
                self.assertEqual("".join(bodies(parts)), text)
                for index, part in enumerate(parts, 1):
                    self.assertTrue(part.startswith(f"【{index}/{len(parts)}】 "))
                    self.assertLessEqual(len(part.encode("utf-8")), 2048)

    def test_emoji_flags_modifiers_and_combining_marks_stay_whole(self):
        atoms = ("👨‍👩‍👧‍👦", "🇨🇳", "👍🏽", "e\u0301", "1️⃣")
        text = "".join(atoms) * 300
        parts = push.split_text(text)
        self.assertEqual("".join(bodies(parts)), text)
        for body in bodies(parts):
            for atom in atoms:
                body = body.replace(atom, "")
            self.assertEqual(body, "")

    def test_prefer_a_paragraph_boundary_and_preserve_newline(self):
        first = "a" * 1200 + "\n"
        text = first + "b" * 1100
        parts = push.split_text(text)
        self.assertEqual(bodies(parts)[0], first)
        self.assertEqual("".join(bodies(parts)), text)

    def test_unsafe_or_empty_input_is_rejected(self):
        for text in ("", " \n", None, "\ud800", "a" + "\u0301" * 2100):
            with self.subTest(text_type=type(text).__name__):
                with self.assertRaises(push.PermanentFailure):
                    push.split_text(text)


class HttpTests(unittest.TestCase):
    def setUp(self):
        # A suite-wide post mock prevents real HTTP even if an assertion fails.
        self.environment = patch.dict(os.environ, {"WECHAT_WEBHOOK_URL": FAKE_URL}, clear=True)
        self.post_patch = patch.object(push.requests, "post", return_value=response())
        self.slot_patch = patch.object(push, "_acquire_slot")
        self.sleep_patch = patch.object(push.time, "sleep")
        self.jitter_patch = patch.object(push.random, "uniform", return_value=0)
        self.environment.start()
        self.post = self.post_patch.start()
        self.slot = self.slot_patch.start()
        self.sleep = self.sleep_patch.start()
        self.jitter_patch.start()
        self.addCleanup(patch.stopall)

    def assert_secret_absent(self, exception):
        rendered = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        self.assertNotIn(FAKE_URL, rendered)
        self.assertNotIn("test-not-a-real-secret", rendered)
        self.assertIsNone(exception.__context__)
        self.assertIsNone(exception.__cause__)

    def test_success_requires_http_and_errcode_and_returns_local_receipt(self):
        receipt = push.send_part("测试🌧️")
        self.assertRegex(receipt, r"^accepted:[0-9a-f-]{36}$")
        args, kwargs = self.post.call_args
        self.assertEqual(args, (FAKE_URL,))
        self.assertEqual(json.loads(kwargs["data"]), {"msgtype": "text", "text": {"content": "测试🌧️"}})
        self.assertEqual(kwargs["timeout"], (10, 30))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertIn("utf-8", kwargs["headers"]["Content-Type"])
        self.slot.assert_called_once()

    def test_permanent_api_errors_do_not_retry_or_echo_errmsg(self):
        for code in (45002, 93000, 93001, 93004, 40058, 999999):
            with self.subTest(code=code):
                self.post.reset_mock()
                self.post.return_value = response(code, errmsg=FAKE_URL)
                with self.assertRaises(push.PermanentFailure) as caught:
                    push.send_part("正文")
                self.assertEqual(self.post.call_count, 1)
                self.assert_secret_absent(caught.exception)

    def test_system_failure_and_limit_have_bounded_retries(self):
        for code in (-1, 45009):
            with self.subTest(code=code):
                self.post.reset_mock()
                self.sleep.reset_mock()
                self.post.return_value = response(code)
                with self.assertRaises(push.RetryableFailure):
                    push.send_part("正文", attempts=3, pause=1)
                self.assertEqual(self.post.call_count, 3)
                self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2])

    def test_connect_timeout_can_retry(self):
        self.post.side_effect = [requests.exceptions.ConnectTimeout(FAKE_URL), response()]
        self.assertTrue(push.send_part("正文").startswith("accepted:"))
        self.assertEqual(self.post.call_count, 2)

    def test_dns_or_refused_connection_can_retry(self):
        underlying = NewConnectionError(None, FAKE_URL)
        nested = requests.exceptions.ConnectionError(MaxRetryError(None, FAKE_URL, underlying))
        self.post.side_effect = [nested, response()]
        push.send_part("正文")
        self.assertEqual(self.post.call_count, 2)

    def test_read_timeout_and_ambiguous_connection_stop_without_retry(self):
        for error in (requests.exceptions.ReadTimeout(FAKE_URL),
                      requests.exceptions.ConnectionError(FAKE_URL),
                      requests.exceptions.RequestException(FAKE_URL)):
            with self.subTest(error=type(error).__name__):
                self.post.reset_mock()
                self.post.side_effect = error
                with self.assertRaises(push.DeliveryUnknown) as caught:
                    push.send_part("正文")
                self.assertEqual(self.post.call_count, 1)
                self.assert_secret_absent(caught.exception)

    def test_tls_failure_never_disables_verification_or_retries(self):
        self.post.side_effect = requests.exceptions.SSLError(FAKE_URL)
        with self.assertRaises(push.PermanentFailure) as caught:
            push.send_part("正文")
        self.assertEqual(self.post.call_count, 1)
        self.assert_secret_absent(caught.exception)
        self.assertNotIn("verify", self.post.call_args.kwargs)

    def test_unparseable_or_invalid_response_is_unknown(self):
        for reply in (None, [], {}, {"errcode": "0"}, {"errcode": False}):
            with self.subTest(reply=reply):
                self.post.reset_mock()
                self.post.return_value = response()
                self.post.return_value.json.return_value = reply
                with self.assertRaises(push.DeliveryUnknown):
                    push.send_part("正文")
                self.assertEqual(self.post.call_count, 1)
        self.post.return_value.json.side_effect = ValueError(FAKE_URL)
        with self.assertRaises(push.DeliveryUnknown) as caught:
            push.send_part("正文")
        self.assert_secret_absent(caught.exception)

    def test_http_error_and_redirect_are_not_accepted(self):
        for status, error_type in ((302, push.PermanentFailure), (403, push.PermanentFailure),
                                   (500, push.DeliveryUnknown)):
            with self.subTest(status=status):
                self.post.reset_mock()
                self.post.return_value = response(0, status)
                with self.assertRaises(error_type):
                    push.send_part("正文")
                self.assertEqual(self.post.call_count, 1)
        self.post.side_effect = [response(0, 429), response()]
        self.assertTrue(push.send_part("正文").startswith("accepted:"))

    def test_validation_happens_before_any_http(self):
        for kwargs in ({"text": "雨" * 683}, {"text": ""}, {"text": "正文", "attempts": 4},
                       {"text": "正文", "pause": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(push.PermanentFailure):
                    push.send_part(**kwargs)
        self.post.assert_not_called()

    def test_only_expected_https_host_and_endpoint_are_allowed(self):
        for url in ("", FAKE_URL.replace("https:", "http:"),
                    FAKE_URL.replace("qyapi.weixin.qq.com", "example.com"),
                    FAKE_URL + "&key=another", FAKE_URL + "#fragment",
                    FAKE_URL.replace("/webhook/send", "/message/send")):
            with self.subTest(url_kind=url.split(":")[0]):
                os.environ["WECHAT_WEBHOOK_URL"] = url
                with self.assertRaises(push.PermanentFailure) as caught:
                    push.send_part("正文")
                self.assert_secret_absent(caught.exception)
        self.post.assert_not_called()

    def test_dotenv_is_data_and_environment_takes_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            envfile = Path(folder) / "config.env"
            envfile.write_text(f'export WECHAT_WEBHOOK_URL="{FAKE_URL}" # comment\nUNRELATED=$(bad)\n', encoding="utf-8")
            os.environ.pop("WECHAT_WEBHOOK_URL")
            os.environ["LIGHTSHOW_ENV_FILE"] = str(envfile)
            push.send_part("正文")
            self.assertEqual(self.post.call_args.args[0], FAKE_URL)
            os.environ["WECHAT_WEBHOOK_URL"] = FAKE_URL.replace("test-not-a-real-secret", "test-override")
            push.send_part("正文")
            self.assertTrue(self.post.call_args.args[0].endswith("test-override"))

    def test_send_compatibility_splits_and_preserves_order(self):
        text = "雨" * 1000
        self.assertTrue(push.send(text, attempts=1).startswith("batch:"))
        submitted = [json.loads(call.kwargs["data"])["text"]["content"] for call in self.post.call_args_list]
        self.assertEqual(submitted, push.split_text(text))

    def test_callback_records_each_part_and_stops_after_unknown(self):
        self.post.side_effect = [response(), requests.exceptions.ReadTimeout(FAKE_URL), response()]
        events = []
        with self.assertRaises(push.DeliveryUnknown):
            push.send_messages([("detail", "雨" * 1000), ("brief", "简版")], callback=events.append)
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual([event["status"] for event in events], ["sending", "accepted", "sending", "unknown"])
        self.assertEqual(events[-1]["message_key"], "detail")
        self.assertEqual(events[-1]["part_index"], 2)
        self.assertEqual(events[-1]["part_total"], 2)
        self.assertIsNone(events[-1]["receipt"])

    def test_disabled_gate_never_calls_http_and_is_not_retried(self):
        gate = Mock(return_value=False)
        with self.assertRaises(push.DeliveryPaused):
            push.send_part("正文", before_request=gate)
        self.post.assert_not_called()
        self.slot.assert_called_once()
        gate.assert_called_once()
        self.sleep.assert_not_called()

    def test_switch_closes_during_retry_backoff(self):
        enabled = [True]
        gate = Mock(side_effect=lambda: enabled[0])
        self.post.side_effect = [response(-1), response()]
        self.sleep.side_effect = lambda seconds: enabled.__setitem__(0, False)
        with self.assertRaises(push.DeliveryPaused):
            push.send_part("正文", before_request=gate)
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_switch_closes_during_persistent_rate_wait(self):
        self.slot_patch.stop()
        with tempfile.TemporaryDirectory() as folder:
            enabled = [True]
            clock = [1000.0]

            def waiting(seconds):
                clock[0] += seconds
                enabled[0] = False

            self.sleep.side_effect = waiting
            with patch.dict(os.environ, {"LIGHTSHOW_STATE_DIR": folder}), \
                 patch.object(push.time, "time", side_effect=lambda: clock[0]):
                for _ in range(20):
                    push._acquire_slot({})
                with self.assertRaises(push.DeliveryPaused):
                    push.send_part("正文", before_request=lambda: enabled[0])
        self.post.assert_not_called()
        self.assertEqual(self.sleep.call_count, 1)
        self.assertGreaterEqual(self.sleep.call_args.args[0], 60)

    def test_compatibility_entrypoints_forward_gate_and_report_paused(self):
        events = []
        with self.assertRaises(push.DeliveryPaused):
            push.send_messages([("brief", "简稿")], callback=events.append, before_request=lambda: False)
        self.assertEqual([event["status"] for event in events], ["sending", "paused"])
        with self.assertRaises(push.DeliveryPaused):
            push.send("正文", before_request=lambda: False)
        self.post.assert_not_called()


class RateTests(unittest.TestCase):
    def test_shared_rolling_window_persists_twenty_attempts_and_waits(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = [1000.0]
            slept = []

            def advance(seconds):
                slept.append(seconds)
                clock[0] += seconds

            with patch.dict(os.environ, {"LIGHTSHOW_STATE_DIR": folder}, clear=True), \
                 patch.object(push.time, "time", side_effect=lambda: clock[0]), \
                 patch.object(push.time, "sleep", side_effect=advance):
                for _ in range(20):
                    push._acquire_slot({})
                self.assertEqual(slept, [])
                # A new invocation sees persisted state, rather than 20 fresh tokens.
                push._acquire_slot({})
                self.assertEqual(len(slept), 1)
                self.assertGreaterEqual(slept[0], 60.0)
                with sqlite3.connect(Path(folder) / "wechat_rate_limit.sqlite3") as connection:
                    count = connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
                self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
