import argparse
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import cloud_runner as runner
import wechat_push
import rain_alert
import radar_nowcast
import rolling_nowcast
import numpy as np


class DeliveryStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.patch_home = patch.object(runner, 'HOME', self.home)
        self.patch_env = patch.object(runner, 'ENV_FILE', self.home / 'config.env')
        self.patch_home.start()
        self.patch_env.start()
        (self.home / 'config.env').write_text('AUTO_PUSH_ENABLED=1\nOFFICIAL_TRIGGER_ENABLED=1\n')
        self.db = runner.database()
        self.issue = datetime.now(runner.BJT).replace(second=0, microsecond=0)
        self.key = runner.job_key(self.issue, 'nowcast')
        self.draft = self.home / 'drafts' / 'immutable'
        self.draft.mkdir(parents=True)
        metadata = {'issue': self.issue.strftime('%Y-%m-%d %H:%M'), 'event_date': self.issue.strftime('%Y-%m-%d'),
                    'generated_at': self.issue.isoformat(), 'external': '详细稿', 'brief': '简稿',
                    'target': {'name':'市民中心','lat':22.543,'lon':114.059,'radius_km':3},
                    'publication_allowed': True, '_draft': str(self.draft), 'checks': []}
        (self.draft / 'briefing_result.json').write_text(json.dumps(metadata))
        self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)',
                        (self.key, self.issue.isoformat(), metadata['event_date'], 'nowcast', 'ready', str(self.draft), self.issue.isoformat(), None, 0))
        for index, content in enumerate(('提醒', '详细稿', '简稿')):
            self.db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?)',
                            (self.key,index,content,'hash','pending',None,None))
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.patch_home.stop()
        self.patch_env.stop()
        self.environment.stop()
        self.temp.cleanup()

    def states(self):
        return [row[0] for row in self.db.execute('SELECT state FROM messages ORDER BY part')]

    def test_disabled_never_calls_transport(self):
        (self.home / 'config.env').write_text('AUTO_PUSH_ENABLED=0\n')
        sender = Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
        sender.assert_not_called()
        self.assertEqual(self.states(),['pending']*3)

    def test_partial_recovery_sends_only_failed_and_remaining(self):
        sender = Mock(side_effect=['accepted:1',wechat_push.RetryableFailure('rejected')])
        self.assertEqual(runner.deliver(self.db,self.key,sender),'partial')
        self.assertEqual(self.states(),['sent','failed','pending'])
        retry = Mock(return_value='accepted:next')
        self.assertEqual(runner.deliver(self.db,self.key,retry),'sent')
        self.assertEqual([call.args[0] for call in retry.call_args_list],['详细稿','简稿'])
        self.assertEqual(self.states(),['sent']*3)

    def test_unknown_never_automatically_resends(self):
        sender = Mock(side_effect=wechat_push.DeliveryUnknown('timeout'))
        self.assertEqual(runner.deliver(self.db,self.key,sender),'unknown')
        retry = Mock()
        self.assertEqual(runner.deliver(self.db,self.key,retry),'unknown')
        retry.assert_not_called()

    def test_crash_during_http_becomes_unknown(self):
        self.db.execute("UPDATE messages SET state='sending' WHERE part=0")
        self.db.commit()
        sender = Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'unknown')
        sender.assert_not_called()
        self.assertEqual(self.states()[0],'unknown')

    def test_permanent_rejection_does_not_repeat(self):
        sender=Mock(side_effect=wechat_push.PermanentFailure('disabled'))
        self.assertEqual(runner.deliver(self.db,self.key,sender),'permanent_failure')
        retry=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,retry),'permanent_failure')
        retry.assert_not_called()

    def test_expired_draft_is_not_sent(self):
        path=self.draft/'briefing_result.json'
        metadata=json.loads(path.read_text())
        metadata['generated_at']=(self.issue-timedelta(hours=2)).isoformat()
        path.write_text(json.dumps(metadata))
        sender=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'expired')
        sender.assert_not_called()

    def test_closed_run_does_not_generate(self):
        (self.home/'config.env').write_text('AUTO_PUSH_ENABLED=0\n')
        args=argparse.Namespace(slot='08',date=None,issue=None,mode='scheduled',dry_run=False)
        with patch.object(runner,'generate') as generate:
            self.assertEqual(runner.run(args,self.db),0)
            generate.assert_not_called()

    def test_closed_watchdog_does_not_retry_or_alert(self):
        (self.home/'config.env').write_text('AUTO_PUSH_ENABLED=0\n')
        with patch.object(runner,'deliver') as deliver,patch.object(runner,'notify_failure') as notify:
            self.assertEqual(runner.watchdog(argparse.Namespace(slot='08',date=None),self.db),0)
            deliver.assert_not_called()
            notify.assert_not_called()

    def test_scheduled_and_evening_nowcast_dates(self):
        issue=runner.issue_for('20','2026-10-13')
        self.assertEqual(runner.event_date(issue,'scheduled'),'2026-10-13')
        self.assertEqual(runner.event_date(issue,'nowcast'),'2026-10-13')
        self.assertEqual(runner.event_date(issue,'manual'),'2026-10-13')

    def test_late_scheduled_task_is_expired(self):
        issue=runner.issue_for('08','2026-10-13')
        self.assertFalse(runner.within_send_window(issue,'scheduled',issue+timedelta(hours=2)))

    def test_pause_during_transport_returns_part_to_pending(self):
        sender=Mock(side_effect=wechat_push.DeliveryPaused('switch closed'))
        self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
        self.assertEqual(self.states(),['pending']*3)

    def radar_metadata(self,when):
        return {'product':'radar_nowcast','generated_at':when.isoformat(),'radar_time':when.isoformat(),
                'issue':when.isoformat(),'event_date':when.strftime('%Y-%m-%d'),'fresh':True,
                'publication_allowed':True,'forecast_reliable':True,'clear':False,
                'target':{'name':'市民中心','lat':22.543,'lon':114.059,'radius_km':3},
                'current':{'status':'clear','coverage':True,'rain':False,'strong':False},
                'horizon_minutes':120,'arrival_minutes':10,'strong_arrival_minutes':None,
                'external':'【雷达短临预报】测试草稿','brief':'测试草稿','checks':[],
                '_draft':str(self.draft)}

    def confirmed_alert(self):
        now=datetime.now(runner.BJT)
        rain_alert.evaluate(self.radar_metadata(now-timedelta(minutes=5)),self.db,now=now-timedelta(minutes=5))
        metadata=self.radar_metadata(now)
        decision=rain_alert.evaluate(metadata,self.db,now=now)
        self.assertTrue(decision['should_notify'])
        return decision,metadata

    def enable_alerts(self):
        (self.home/'config.env').write_text('AUTO_PUSH_ENABLED=1\nRAIN_ALERTS_ENABLED=1\n')

    def test_alert_monitor_off_does_not_generate(self):
        with patch.object(runner,'generate') as generate:
            self.assertEqual(runner.monitor(argparse.Namespace(dry_run=False),self.db),0)
            generate.assert_not_called()

    def test_preview_does_not_advance_real_alert_state_or_send(self):
        metadata=self.radar_metadata(datetime.now(runner.BJT))
        with patch.object(runner,'generate',return_value=metadata),patch.object(runner,'send_rain_alert') as send:
            self.assertEqual(runner.monitor(argparse.Namespace(dry_run=True),self.db),0)
            send.assert_not_called()
        self.assertIsNone(self.db.execute("SELECT name FROM sqlite_master WHERE name='rain_alert_state'").fetchone())

    def test_two_frame_alert_sent_once_via_durable_ledger(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        with patch.object(wechat_push,'send_part',return_value='accepted:test') as send:
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'sent')
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'sent')
            self.assertEqual(send.call_count,1)
        state=self.db.execute('SELECT status FROM rain_alert_messages WHERE message_key=?',(decision['message_key'],)).fetchone()[0]
        self.assertEqual(state,'accepted')

    def test_alert_switch_closed_blocks_even_confirmed_event(self):
        decision,metadata=self.confirmed_alert()
        with patch.object(wechat_push,'send_part') as send:
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'disabled')
            send.assert_not_called()
        state=self.db.execute('SELECT status FROM rain_alert_messages WHERE message_key=?',(decision['message_key'],)).fetchone()[0]
        self.assertEqual(state,'pending')

    def test_receipt_reconciles_crash_after_success_without_reposting(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        real_mark=rain_alert.mark_delivery
        def crash_after_http(db,key,status,**kwargs):
            if status=='accepted':
                raise RuntimeError('simulated crash before bookkeeping')
            return real_mark(db,key,status,**kwargs)
        with patch.object(wechat_push,'send_part',return_value='accepted:test') as send,\
             patch.object(rain_alert,'mark_delivery',side_effect=crash_after_http):
            with self.assertRaises(RuntimeError):
                runner.send_rain_alert(self.db,decision,metadata)
            self.assertEqual(send.call_count,1)
        # Last HTTP receipt can be committed before the job's final state.
        self.db.execute("UPDATE jobs SET state='sending' WHERE id=?",(decision['message_key'],))
        self.db.commit()
        runner.reconcile_alert_receipts(self.db)
        status=self.db.execute('SELECT status FROM rain_alert_messages WHERE message_key=?',(decision['message_key'],)).fetchone()[0]
        self.assertEqual(status,'accepted')
        self.assertEqual(self.db.execute('SELECT state FROM jobs WHERE id=?',(decision['message_key'],)).fetchone()[0],'sent')

    def test_unknown_alert_delivery_stays_unknown_after_reconcile(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        with patch.object(wechat_push,'send_part',side_effect=wechat_push.DeliveryUnknown('read timeout')),\
             patch.object(runner,'notify_failure'):
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'unknown')
        runner.reconcile_alert_receipts(self.db)
        status=self.db.execute('SELECT status FROM rain_alert_messages WHERE message_key=?',(decision['message_key'],)).fetchone()[0]
        self.assertEqual(status,'unknown')

    def test_alert_recovers_crash_after_job_commit_from_original_draft_once(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        key=decision['message_key']
        with patch.object(runner,'prepare_messages',side_effect=RuntimeError('crash before preparation')),\
             patch.object(wechat_push,'send_part') as send:
            with self.assertRaises(RuntimeError):
                runner.send_rain_alert(self.db,decision,metadata)
            send.assert_not_called()
        job=self.db.execute('SELECT * FROM jobs WHERE id=?',(key,)).fetchone()
        original_directory=Path(job['draft'])
        original_file=original_directory/'radar_result.json'
        original_bytes=original_file.read_bytes()
        directories=set((self.home/'drafts').iterdir())
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM messages WHERE job_id=?',(key,)).fetchone()[0],0)
        # New incoming decision text is not allowed to replace the old draft.
        changed=dict(decision,text='替换稿件不应发送')
        with patch.object(wechat_push,'send_part',return_value='accepted:restored') as send:
            self.assertEqual(runner.send_rain_alert(self.db,changed,metadata),'sent')
            self.assertEqual(runner.send_rain_alert(self.db,changed,metadata),'sent')
            self.assertEqual(send.call_count,1)
            self.assertEqual(send.call_args.args[0],decision['text'])
        self.assertEqual(original_file.read_bytes(),original_bytes)
        self.assertEqual(set((self.home/'drafts').iterdir()),directories)
        restored=self.db.execute('SELECT draft,state FROM jobs WHERE id=?',(key,)).fetchone()
        self.assertEqual(restored['draft'],str(original_directory))
        self.assertEqual(restored['state'],'sent')
        self.assertEqual(self.db.execute('SELECT status FROM rain_alert_messages WHERE message_key=?',(key,)).fetchone()[0],'accepted')

    def test_partial_alert_preparation_error_rolls_back_before_retry(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        key=decision['message_key']
        def broken_preparation(db,job_key,persisted):
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?)',
                       (job_key,0,'未完成准备的错误文本','digest','pending',None,None))
            raise RuntimeError('interrupted after the first ledger insert')
        with patch.object(runner,'prepare_messages',side_effect=broken_preparation),\
             patch.object(wechat_push,'send_part') as send:
            with self.assertRaises(RuntimeError):
                runner.send_rain_alert(self.db,decision,metadata)
            send.assert_not_called()
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM messages WHERE job_id=?',(key,)).fetchone()[0],0)
        self.assertIsNotNone(self.db.execute('SELECT id FROM jobs WHERE id=?',(key,)).fetchone())
        with patch.object(wechat_push,'send_part',return_value='accepted:complete') as send:
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'sent')
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'sent')
            self.assertEqual(send.call_count,1)
            self.assertEqual(send.call_args.args[0],decision['text'])

    def test_terminal_alert_jobs_are_not_reprepared_when_the_ledger_is_missing(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        key=decision['message_key']
        with patch.object(runner,'prepare_messages',side_effect=RuntimeError('crash before preparation')):
            with self.assertRaises(RuntimeError):
                runner.send_rain_alert(self.db,decision,metadata)
        for state in ('unknown','permanent_failure','expired','sent'):
            with self.subTest(state=state):
                self.db.execute('UPDATE jobs SET state=? WHERE id=?',(state,key))
                self.db.commit()
                with patch.object(runner,'prepare_messages') as prepare,\
                     patch.object(wechat_push,'send_part') as send:
                    self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),state)
                    prepare.assert_not_called()
                    send.assert_not_called()

    def select_region(self, name='宝安测试点', lat=22.64, lon=113.81, radius=5):
        (self.home/'config.env').write_text(
            f'AUTO_PUSH_ENABLED=1\nRAIN_ALERTS_ENABLED=1\nOFFICIAL_TRIGGER_ENABLED=1\nRADAR_TARGET_NAME={name}\n'
            f'RADAR_TARGET_LAT={lat}\nRADAR_TARGET_LON={lon}\nRADAR_TARGET_RADIUS_KM={radius}\n')

    def test_region_config_reaches_actual_cli_and_generated_metadata(self):
        self.select_region()
        now=datetime.now(runner.BJT)
        axis=np.arange(-200,201,dtype=float)
        east,north=np.meshgrid(axis,axis[::-1])
        blank=np.zeros_like(east)
        frames=[(now-timedelta(minutes=6),blank,east,north),(now,blank,east,north)]
        def execute_cli(command, **kwargs):
            with redirect_stdout(kwargs['stdout']):
                self.assertTrue(command[1].endswith('rolling_nowcast.py'))
                code=rolling_nowcast.main(command[2:])
            return SimpleNamespace(returncode=code)
        with patch.object(radar_nowcast,'load_frames',return_value=frames) as fetch,\
             patch.object(runner.subprocess,'run',side_effect=execute_cli):
            metadata=runner.generate(self.issue,'manual',dry_run=True)
        selected=fetch.call_args.kwargs['config']
        self.assertEqual((selected.name,selected.lat,selected.lon,selected.radius_km),('宝安测试点',22.64,113.81,5))
        self.assertEqual(metadata['target'],{'name':'宝安测试点','lat':22.64,'lon':113.81,'radius_km':5})
        self.assertIn('宝安测试点雷达短临预报',metadata['external'])
        self.assertNotIn('市民中心',metadata['external'])
        self.assertTrue(metadata['publication_allowed'])
        self.assertEqual(metadata['algorithm_version'], 'radar-rolling-v5')

    def test_region_switch_blocks_old_draft_and_creates_distinct_slot_key(self):
        original=self.key
        for settings in ({'lat':22.65},{'lon':113.8},{'radius':5},{'name':'新名称'}):
            with self.subTest(settings=settings):
                self.select_region(name='市民中心',lat=22.543,lon=114.059,radius=3)
                values={'name':'市民中心','lat':22.543,'lon':114.059,'radius':3}
                values.update(settings)
                self.select_region(**values)
                self.assertNotEqual(runner.job_key(self.issue,'nowcast'),original)
                sender=Mock()
                self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
                sender.assert_not_called()
                self.assertEqual(self.states(),['pending']*3)

    def test_region_switch_while_waiting_stops_http_callback(self):
        def sender(text,before_request):
            self.select_region()
            self.assertFalse(before_request())
            raise wechat_push.DeliveryPaused('region changed')
        self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
        self.assertEqual(self.states(),['pending']*3)

    def test_invalid_region_config_rejects_generation_before_subprocess(self):
        for value in ('nan','inf','91','not-a-latitude'):
            with self.subTest(value=value):
                self.select_region(lat=value)
                with patch.object(runner.subprocess,'run') as execute:
                    with self.assertRaisesRegex(ValueError,'地区配置无效'):
                        runner.generate(self.issue,'manual',dry_run=True)
                    execute.assert_not_called()

    def test_slot_key_and_metadata_mismatch_cannot_send_after_config_race(self):
        self.select_region()
        path=self.draft/'briefing_result.json'
        metadata=json.loads(path.read_text())
        metadata['target']=runner.target_settings()
        path.write_text(json.dumps(metadata))
        sender=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
        sender.assert_not_called()

    def install_three_part_forecast(self, detail=None):
        metadata=json.loads((self.draft/'briefing_result.json').read_text())
        metadata.update(product='radar_nowcast',message_format=runner.MESSAGE_FORMAT,
                        notice='【雷达短临草稿】提醒正文',external='[详细版]当前雷达与两小时研判',
                        brief='【简版】自然文字正文',internal='内部技术依据：25dBZ，至少3像元')
        if detail is not None:
            metadata['external']=detail
        metadata['messages']=[metadata['notice'],metadata['external'],metadata['brief']]
        (self.draft/'radar_result.json').write_text(json.dumps(metadata))
        self.db.execute('DELETE FROM messages WHERE job_id=?',(self.key,))
        self.db.commit()
        runner.prepare_messages(self.db,self.key,metadata)
        return metadata

    def test_forecast_delivers_notice_detail_brief_in_exact_order(self):
        metadata=self.install_three_part_forecast()
        sender=Mock(return_value='accepted:three-part')
        self.assertEqual(runner.deliver(self.db,self.key,sender),'sent')
        self.assertEqual([call.args[0] for call in sender.call_args_list],metadata['messages'])
        self.assertEqual(sender.call_count,3)
        self.assertNotIn(metadata['internal'],[call.args[0] for call in sender.call_args_list])

    def test_three_part_forecast_recovery_only_sends_remaining_bodies(self):
        metadata=self.install_three_part_forecast()
        sender=Mock(side_effect=['accepted:notice',wechat_push.RetryableFailure('known rejection')])
        self.assertEqual(runner.deliver(self.db,self.key,sender),'partial')
        self.assertEqual(self.states(),['sent','failed','pending'])
        retry=Mock(return_value='accepted:remaining')
        self.assertEqual(runner.deliver(self.db,self.key,retry),'sent')
        self.assertEqual([call.args[0] for call in retry.call_args_list],metadata['messages'][1:])

    def test_split_detail_second_part_failure_resumes_before_brief(self):
        metadata=self.install_three_part_forecast(detail='细'*1000)
        expected=[part for text in metadata['messages'] for part in wechat_push.split_text(text)]
        self.assertEqual(len(expected),4)
        sender=Mock(side_effect=['accepted:notice','accepted:detail-1',wechat_push.RetryableFailure('known rejection')])
        self.assertEqual(runner.deliver(self.db,self.key,sender),'partial')
        self.assertEqual([call.args[0] for call in sender.call_args_list],expected[:3])
        retry=Mock(return_value='accepted:rest')
        self.assertEqual(runner.deliver(self.db,self.key,retry),'sent')
        self.assertEqual([call.args[0] for call in retry.call_args_list],expected[2:])

    def test_unknown_second_detail_part_stops_before_brief_and_never_retries(self):
        metadata=self.install_three_part_forecast(detail='细'*1000)
        expected=[part for text in metadata['messages'] for part in wechat_push.split_text(text)]
        sender=Mock(side_effect=['accepted:notice','accepted:detail-1',wechat_push.DeliveryUnknown('read timeout')])
        self.assertEqual(runner.deliver(self.db,self.key,sender),'unknown')
        self.assertEqual([call.args[0] for call in sender.call_args_list],expected[:3])
        retry=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,retry),'unknown')
        retry.assert_not_called()
        self.assertEqual(self.states(),['sent','sent','unknown','pending'])

    def test_old_single_message_forecast_cannot_be_sent_as_current_format(self):
        path=self.draft/'briefing_result.json'
        metadata=json.loads(path.read_text())
        metadata['product']='radar_nowcast'
        path.write_text(json.dumps(metadata))
        sender=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
        sender.assert_not_called()

    def test_wrong_or_missing_three_part_order_is_rejected(self):
        metadata=self.install_three_part_forecast()
        for messages in ([metadata['external']],metadata['messages'][::-1]):
            with self.subTest(messages=messages):
                changed=dict(metadata,messages=messages)
                with self.assertRaises(ValueError):
                    runner.message_texts(changed)
                (self.draft/'radar_result.json').write_text(json.dumps(changed))
                sender=Mock()
                self.assertEqual(runner.deliver(self.db,self.key,sender),'disabled')
                sender.assert_not_called()

    def test_corrupted_old_or_reordered_ledger_cannot_send_forecast(self):
        metadata=self.install_three_part_forecast()
        self.db.execute("UPDATE messages SET content='旧的单条正文' WHERE job_id=? AND part=0",(self.key,))
        self.db.commit()
        sender=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'permanent_failure')
        sender.assert_not_called()

    def test_corrupted_digest_cannot_send_forecast(self):
        self.install_three_part_forecast()
        self.db.execute("UPDATE messages SET digest='incorrect' WHERE job_id=? AND part=1",(self.key,))
        self.db.commit()
        sender=Mock()
        self.assertEqual(runner.deliver(self.db,self.key,sender),'permanent_failure')
        sender.assert_not_called()

    def test_rain_alert_stays_single_when_source_forecast_has_three_messages(self):
        self.enable_alerts()
        decision,metadata=self.confirmed_alert()
        metadata.update(message_format=runner.MESSAGE_FORMAT,notice='常规提醒',
                        messages=['常规提醒','常规详细','常规简版'])
        with patch.object(wechat_push,'send_part',return_value='accepted:rain') as send:
            self.assertEqual(runner.send_rain_alert(self.db,decision,metadata),'sent')
            self.assertEqual(send.call_count,1)
            self.assertEqual(send.call_args.args[0],decision['text'])
        job=self.db.execute('SELECT draft FROM jobs WHERE id=?',(decision['message_key'],)).fetchone()
        saved=runner.read_metadata(job['draft'])
        self.assertEqual(saved['message_format'],runner.RAIN_MESSAGE_FORMAT)
        self.assertEqual(runner.message_texts(saved),[decision['text']])


if __name__=='__main__':
    unittest.main()
