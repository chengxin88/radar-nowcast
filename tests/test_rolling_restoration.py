"""Restored 15-minute algorithm and official context; no live HTTP or WeChat."""
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import numpy as np
import requests
import radar_nowcast as radar
import radar_track
import rolling_radar as corridor
import rolling_official as official
import rolling_nowcast as product
import cloud_runner as controller
import manage_schedule as schedule
import wechat_push

NOW = datetime(2026, 10, 13, 18, 37, tzinfo=radar.BJT)


def frames(now=NOW, field_fn=None, extent=150):
    e, n = np.meshgrid(np.arange(-extent, extent + 1, dtype=float), np.arange(extent, -extent - 1, -1, dtype=float))
    z = np.zeros_like(e) if field_fn is None else field_fn(e, n)
    return [(now - timedelta(minutes=30), z.copy(), e, n), (now, z, e, n)]


def moved():
    return {'status': 'ok', 'reason': None, 've': 60., 'vn': 0., 'correlation': .9, 'peak_margin': .3}


def feeds(now=NOW):
    def feed(data):
        return {'cache_time': now.isoformat(), 'data': data}
    start = now.replace(minute=0, second=0, microsecond=0)
    rows = [{'areaName': '福田区', 'forecasttime': (start + timedelta(hours=i)).isoformat(),
             'pubTime': (now - timedelta(hours=1)).isoformat(), 'state': '少云', 'minT': 28-i,
             'maxT': 28-i, 'wd': '南', 'ws': '2'} for i in range(4)]
    return {'hourly': feed({'hoursweather':rows}), 'observation':feed({'date':now.isoformat(), 't':27.9, 'u':79, 'v':12.2, 'h01r':0}),
            'alarm':feed({'subAlarm':[], 'ingnalNum':0}), 'tips':feed({'important':None,'more':[]})}


def build(fs=None, raw=None, config=None, now=NOW, moving=True):
    config = config or radar.Config()
    target = {'lat':config.lat, 'lon':config.lon, 'radius_km':config.radius_km}
    profile = official.profile_for(target, path='')
    with patch.object(radar_track, 'motion', return_value=moved()) if moving else patch.object(radar_track, 'motion', wraps=radar_track.motion):
        return product.build(frames(now) if fs is None else fs, feeds(now) if raw is None else raw, profile, config, now)


class RollingRestorationTests(unittest.TestCase):
    def setUp(self):
        for method in ('get','post','head'):
            p = patch.object(requests, method, side_effect=AssertionError('No live network in tests'))
            p.start()
            self.addCleanup(p.stop)

    def test_original_corridor_has_three_independent_clocks_and_no_rain_rate(self):
        fs = frames(field_fn=lambda e,n: np.where((e <= -20) & (e >= -60) & (abs(n)<=8),
                          np.where(e >= -40, 45., 25.), 0.))
        r = build(fs)['rolling_radar']
        self.assertTrue(r['timing_reliable'])
        self.assertLess(r['arrival_minutes'], r['strong_end_minutes'])
        self.assertLess(r['strong_end_minutes'], r['end_minutes'])
        self.assertEqual((r['width_km'],r['reach_km'],r['step_km']), (5,90,2))
        report = build(fs)
        for msg in report['messages']:
            self.assertNotIn('毫米', msg)
            self.assertNotIn('7-8级', msg)
        self.assertIn('强降水可能随之减弱', report['brief'])
        self.assertIn('雨停时间以现场实况为准', report['external'])

    def test_gap_between_two_bands_is_not_final_rain_end(self):
        self.assertEqual(corridor.trailing([40,40,0,0,0,40,40,20,0,0,0],35),14)
        self.assertEqual(corridor.trailing([40,40,0,0,0,40,40,20,0,0,0],20),16)
        self.assertIsNone(corridor.trailing([40,None,0,0,0],35))
        self.assertIsNone(corridor.trailing([40,0,0],35))

    def test_latency_changes_remaining_time_not_observation_clock(self):
        fs = frames(field_fn=lambda e,n: np.where((e<-35)&(e>-55)&(abs(n)<7),40.,0.))
        a, b = build(fs), build(fs, now=NOW+timedelta(minutes=10))
        self.assertAlmostEqual(a['rolling_radar']['arrival_minutes'] - b['rolling_radar']['arrival_minutes'],10)
        self.assertEqual(a['radar_time'],b['radar_time'])

    def test_midnight_windows_have_next_day_and_do_not_wrap_back(self):
        now = NOW.replace(hour=23,minute=55)
        self.assertIn('次日', corridor.clock_window(now,20))
        self.assertIn('次日', corridor.clock_window(now,40,pad=15,unit=30))

    def test_partial_corridor_cannot_be_padding_clear(self):
        out = build(frames(extent=70))
        r = out['rolling_radar']
        self.assertFalse(r['complete'])
        self.assertFalse(r['timing_reliable'])
        self.assertIsNone(r['end_minutes'])
        self.assertFalse(out['radar_monitor']['complete'])
        self.assertIn('150公里雷达覆盖不足', out['external'])
        self.assertNotIn('周边3公里', out['external'])
        self.assertNotIn('周边150公里暂无明显降水回波', out['external'])

    def test_radius_changes_profile_width_and_arrival(self):
        fs = frames(field_fn=lambda e,n: np.where((e<-35)&(e>-55)&(abs(n)<10),40.,0.))
        a, b = build(fs), build(fs,config=radar.Config(radius_km=8))
        self.assertEqual(b['rolling_radar']['width_km'],8)
        self.assertLess(b['rolling_radar']['arrival_minutes'], a['rolling_radar']['arrival_minutes'])
        self.assertGreater(b['rolling_radar']['end_minutes'],a['rolling_radar']['end_minutes'])

    def test_changed_geometry_never_uses_default_futian_context(self):
        for config in (radar.Config(lat=22.6,lon=114.1,name='新地点'),radar.Config(radius_km=8)):
            out = build(config=config)
            self.assertFalse(out['official']['matched'])
            self.assertEqual(out['official']['hourly'],[])
            self.assertNotIn('深圳国家基本气象站',out['external'])
            self.assertNotIn('目前福田区',out['brief'])

    def test_renaming_preserves_region_binding(self):
        out = build(config=radar.Config(name='测试地区'))
        self.assertTrue(out['official']['matched'])
        self.assertTrue(all('测试地区' in msg for msg in out['messages']))

    def test_profile_allows_explicit_new_geometry_mapping(self):
        profile = copy.deepcopy(official.DEFAULT_PROFILE)
        profile['target']['radius_km']=4
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'profile.json'; p.write_text(json.dumps(profile))
            self.assertIsNotNone(official.profile_for(profile['target'], str(p)))
            self.assertIsNone(official.profile_for({'lat':22.543,'lon':114.059,'radius_km':3},str(p)))

    def test_stale_radar_never_gets_valid_timing_from_good_official_data(self):
        out = build(frames(now=NOW-timedelta(minutes=21)))
        self.assertFalse(out['publication_allowed'])
        self.assertFalse(out['rolling_radar']['timing_reliable'])
        self.assertIn('暂不能确定',out['external'])
        self.assertIn('【降水过程研判】',out['external'])
        self.assertIn('最新收到的雷达资料为18:16',out['external'])
        self.assertNotIn('周边150公里暂无明显降水回波',out['external'])

    def test_no_echo_is_not_called_an_unclear_moving_rainband(self):
        out=build(moving=False)
        self.assertIn('暂无明显降水回波',out['brief'])
        for message in out['messages']:
            self.assertIn('市民中心周边150公里暂无明显降水回波',message)
            self.assertNotIn('【降水过程研判】',message)
            self.assertNotIn('降水起止',message)
            self.assertNotIn('回波新生',message)
            self.assertNotIn('截至18:37，雷达',message)
        self.assertNotIn('回波移动趋势尚不明确',out['brief'])
        self.assertNotIn('未来两小时无雨',out['brief'])

    def test_remote_sparse_echo_does_not_add_empty_local_rain_assessment(self):
        fs=frames(field_fn=lambda e,n:np.where((e>=79)&(e<=81)&(abs(n)<=1),30.,0.))
        out=build(fs,moving=False)
        self.assertFalse(out['observed_clear'])
        self.assertEqual(out['current']['status'],'clear')
        self.assertFalse(out['rolling_radar']['timing_reliable'])
        for message in out['messages']:
            self.assertNotIn('周边3公里',message)
            self.assertIn('东方向约79公里处有降水回波',message)
            self.assertIn('回波移动趋势尚不明确',message)
            self.assertIn('未来2小时是否影响市民中心仍需继续监测',message)
            self.assertNotIn('周边150公里暂无明显降水回波',message)
            self.assertNotIn('【降水过程研判】',message)
            self.assertNotIn('降水起止',message)
            self.assertNotIn('回波新生',message)

    def test_known_incoming_rain_still_has_assessment_when_target_currently_clear(self):
        fs=frames(field_fn=lambda e,n:np.where((e<-35)&(e>-55)&(abs(n)<10),40.,0.))
        out=build(fs)
        self.assertEqual(out['current']['status'],'clear')
        self.assertIn('【降水过程研判】',out['external'])
        self.assertIn('开始影响',out['brief'])
        for message in out['messages']:
            self.assertNotIn('周边3公里',message)
            self.assertIn('可追踪回波整体向东移动',message)
            self.assertIn('未来2小时可能影响市民中心',message)

    def test_reliable_clear_path_has_a_qualified_no_impact_outlook(self):
        fs=frames(field_fn=lambda e,n:np.where((e>20)&(e<40)&(abs(n)<8),30.,0.))
        out=build(fs)
        self.assertEqual(out['radar_outlook']['impact']['status'],'not_expected')
        for message in out['messages']:
            self.assertIn('可追踪回波整体向东移动',message)
            self.assertIn('按现有回波移动外推，预计未来2小时对市民中心暂无明显影响',message)
            self.assertNotIn('周边3公里',message)

    def test_echo_beyond_existing_corridor_is_not_a_150_km_all_clear(self):
        fs=frames(field_fn=lambda e,n:np.where((e>=129)&(e<=131)&(abs(n)<=1),30.,0.))
        out=build(fs,moving=False)
        self.assertEqual(out['radar_monitor']['status'],'echo')
        self.assertEqual(out['target']['radius_km'],3)
        self.assertEqual(out['rolling_radar']['reach_km'],90)
        for message in out['messages']:
            self.assertNotIn('周边3公里',message)
            self.assertIn('东方向约129公里处有降水回波',message)
            self.assertNotIn('周边150公里暂无明显降水回波',message)
            self.assertNotIn('【降水过程研判】',message)

    def test_slow_rain_does_not_fabricate_ending(self):
        fs=frames(field_fn=lambda e,n:np.where((abs(e)<5)&(abs(n)<5),40.,0.))
        with patch.object(radar_track,'motion',return_value={**moved(),'ve':3.}):
            r=product.build(fs,feeds(),official.DEFAULT_PROFILE,radar.Config(),NOW)
        self.assertEqual(r['rolling_radar']['status'],'slow')
        self.assertIn('移动缓慢',r['brief'])
        self.assertIsNone(r['rolling_radar']['strong_end_minutes'])

    def test_weak_local_echo_is_omitted_without_altering_raw_analysis(self):
        out=build(frames(field_fn=lambda e,n:np.where((abs(e)<5)&(abs(n)<5),22.,0.)))
        self.assertTrue(out['rolling_radar']['current_weak'])
        self.assertEqual(out['radar_public_summary']['status'],'filtered_only')
        for message in out['messages']:
            self.assertIn('暂未见明显成片降水回波',message)
            self.assertNotIn('较弱降水回波',message)
            self.assertNotIn('【降水过程研判】',message)
            self.assertNotIn('暂无明显影响',message)
            self.assertNotIn('备好雨具',message)

    def test_tiny_remote_echo_does_not_reappear_in_public_timing_or_outlook(self):
        fs=frames(field_fn=lambda e,n:np.where((e==60)&(abs(n)<=1),25.,0.))
        out=build(fs)
        self.assertEqual(out['radar_monitor']['status'],'echo')
        self.assertEqual(out['radar_public_summary']['status'],'filtered_only')
        for message in out['messages']:
            self.assertIn('暂未见明显成片降水回波',message)
            self.assertNotIn('60公里处',message)
            self.assertNotIn('移动趋势',message)
            self.assertNotIn('未来2小时可能影响',message)
            self.assertNotIn('【降水过程研判】',message)

    def test_measured_site_rain_is_still_reported_when_background_echoes_are_filtered(self):
        raw=feeds()
        station={'name':'市民中心','subdistrict':'莲花街道','areaid':'5949302',
                 'updateTime':NOW.isoformat()}
        raw['yjt_observation']={
            'rain':{'data':dict(station,r05m=1.,r10m=1.,r30m=1.,r01h=1.),
                    'source':'https://yjt.121.com.cn/Data/ForecstReal/JsonRain.json'},
            'wind':{'data':dict(station,w01h=7.2),
                    'source':'https://yjt.121.com.cn/Data/ForecstReal/JsonWind.json'}}
        fs=frames(field_fn=lambda e,n:np.where((abs(e)<5)&(abs(n)<5),20.,0.))
        out=build(fs,raw=raw)
        self.assertEqual(out['radar_public_summary']['status'],'filtered_only')
        self.assertIn('市民中心站过去5分钟1毫米',out['external'])
        for message in out['messages']:
            self.assertIn('市民中心站近期有实测降水',message)
            self.assertNotIn('暂无明显影响',message)

    def test_filtered_nearby_noise_does_not_hide_significant_incoming_echo_outside_150km(self):
        fs=frames(extent=180,field_fn=lambda e,n:np.where(
            (abs(e+120)<=4)&(abs(n+120)<=4),40.,np.where((e==30)&(n==0),20.,0.)))
        with patch.object(radar_track,'motion',return_value={**moved(),'ve':60.,'vn':60.}):
            out=product.build(fs,feeds(),official.DEFAULT_PROFILE,radar.Config(),NOW)
        self.assertEqual(out['radar_public_summary']['status'],'filtered_only')
        self.assertEqual(out['radar_public']['radar_outlook']['impact']['status'],'possible')
        for message in out['messages']:
            self.assertIn('未来2小时可能影响市民中心',message)
            self.assertNotIn('暂无明显影响',message)
            self.assertNotIn('30公里处',message)

    def test_missing_official_data_has_no_fixed_weather_fillers(self):
        out=build(raw={})
        for msg in out['messages']:
            for fabricated in ('26-29','东北风3-4','10公里以上','7-8级','无生效预警'):
                self.assertNotIn(fabricated,msg)
        self.assertIn('暂缺',out['external'])
        self.assertIn('需核实',out['brief'])

    def test_current_visibility_is_not_future_visibility_forecast(self):
        out=build()
        self.assertIn('能见度12.2公里',out['external'])
        self.assertNotIn('能见度',out['brief'])
        self.assertIn('（国家基本站）',out['external'])

    def test_future_or_stale_observation_and_forecast_rejected(self):
        for delta in (timedelta(hours=-2),timedelta(minutes=10)):
            raw=feeds();raw['observation']['data']['date']=(NOW+delta).isoformat()
            self.assertIsNone(build(raw=raw)['official']['observation'])
        raw=feeds()
        for row in raw['hourly']['data']['hoursweather']:row['pubTime']=(NOW-timedelta(hours=19)).isoformat()
        self.assertFalse(build(raw=raw)['official']['forecast_complete'])

    def test_hourly_gap_not_hidden_by_two_available_endpoints(self):
        raw=feeds();raw['hourly']['data']['hoursweather'].pop(1)
        out=build(raw=raw)
        self.assertFalse(out['official']['forecast_complete'])
        self.assertIn('完整天气趋势暂不能确定',out['brief'])

    def test_warning_filter_excludes_other_district_and_unknown_area(self):
        raw=feeds();raw['alarm']['data']['subAlarm']=[
            {'alarmArea':'南山区','alarmType':'暴雨','alarmColor':'橙色'},
            {'alarmArea':'全市陆地','alarmType':'森林火险','alarmColor':'黄色'},
            {'alarmArea':'','alarmType':'雷电','alarmColor':'黄色'}]
        out=build(raw=raw)
        self.assertNotIn('橙色暴雨',out['external'])
        self.assertNotIn('黄色雷电预警信号生效',out['external'])
        self.assertIn('黄色森林火险',out['brief'])
        self.assertTrue(out['official']['warning_unknown'])
        self.assertEqual(official.warning_scope('除福田区外全市',official.DEFAULT_PROFILE),'other')
        self.assertEqual(official.warning_scope('福田区部分街道',official.DEFAULT_PROFILE),'partial')

    def test_old_warning_can_remain_active_in_fresh_current_list(self):
        raw=feeds();raw['alarm']['data']['subAlarm']=[{'date':'2026-10-07 10:00','alarmArea':'全市陆地','alarmColor':'黄色','alarmType':'森林火险'}]
        self.assertIn('森林火险',build(raw=raw)['brief'])

    def test_empty_warning_payload_is_not_no_warning(self):
        raw=feeds();raw['alarm']['data']={'ingnalNum':42}
        self.assertTrue(build(raw=raw)['official']['warning_unknown'])

    def test_invalid_wind_is_missing_not_fixed_or_absurd_force(self):
        raw=feeds();raw['hourly']['data']['hoursweather'][1]['ws']='99'
        out=build(raw=raw)
        self.assertNotIn('99级',out['brief'])
        self.assertIn('风力资料不完整',out['brief'])

    def test_observation_only_has_no_nonexistent_timing_claim(self):
        out=build(moving=False)
        self.assertNotIn('上述到达和结束时间',out['external'])
        self.assertNotIn('回波新生',out['external'])

    def test_city_land_and_sea_warning_also_covers_land_target(self):
        self.assertEqual(official.warning_scope('全市陆地和海区',official.DEFAULT_PROFILE),'local')

    def test_far_nearby_strong_echo_not_claimed_over_target(self):
        fs=frames(field_fn=lambda e,n:np.where((e>4)&(e<10)&(abs(n)<7),40.,0.))
        out=build(fs)
        self.assertFalse(out['rolling_radar']['current_strong'])
        self.assertNotIn('较强回波已影响目标区域',out['brief'])

    def test_short_term_requires_actual_recent_publication_not_fresh_cache(self):
        raw=feeds();raw['tips']['data']['important']={'t':'短临预报','prompt':'明天沿海阵风7-8级','d':(NOW-timedelta(days=1)).isoformat()}
        self.assertIsNone(build(raw=raw)['official']['nowcast'])
        raw['tips']['data']['important']['d']=(NOW+timedelta(minutes=1)).isoformat()
        self.assertIsNone(build(raw=raw)['official']['nowcast'])
        raw['tips']['data']['important']['d']=(NOW-timedelta(minutes=30)).isoformat()
        out=build(raw=raw)
        self.assertIn('全市参考',out['external'])
        self.assertNotIn('7-8级',out['brief'])
        self.assertEqual(len(out['covered_nowcasts']),1)

    def test_three_texts_use_same_restored_data_and_save_all_files(self):
        out=build()
        self.assertEqual(out['messages'],[out['notice'],out['external'],out['brief']])
        self.assertEqual(out['algorithm_version'],'radar-rolling-v5')
        self.assertIn('南风2级',out['brief'])
        self.assertIn('【当前实况】',out['external'])
        self.assertNotIn('【降水过程研判】',out['external'])
        from wechat_push import split_text
        self.assertEqual(sum(len(split_text(t)) for t in out['messages']),3)
        with tempfile.TemporaryDirectory() as d:
            radar.save_result(out,Path(d)/'new')
            self.assertEqual(len(list((Path(d)/'new').iterdir())),5)

    def test_default_off_prevents_all_automatic_generators(self):
        with patch.object(controller,'auto_enabled',return_value=False),patch.object(controller,'generate',side_effect=AssertionError('must not generate')):
            for function in (controller.rolling,controller.nowcast,controller.monitor):
                self.assertEqual(function(Namespace(dry_run=False),None),0)

    def test_mode_switch_off_is_independent_of_master(self):
        with patch.object(controller,'auto_enabled',return_value=True),patch.object(controller,'mode_enabled',return_value=False),patch.object(controller,'generate',side_effect=AssertionError('must not generate')):
            self.assertEqual(controller.rolling(Namespace(dry_run=False),None),0)
            self.assertEqual(controller.nowcast(Namespace(dry_run=False),None),0)

    def test_cron_has_independent_rolling_official_and_alert_switches(self):
        with tempfile.TemporaryDirectory() as d,patch.object(schedule,'HOME',Path(d)):
            p=Path(d)/'config.env';p.write_text('ROLLING_ENABLED=1\nOFFICIAL_TRIGGER_ENABLED=1\nRAIN_ALERTS_ENABLED=0\n')
            text=schedule.schedule_block()
            self.assertIn('*/15 * * * *',text)
            self.assertIn('run.sh rolling',text)
            self.assertIn('run.sh nowcast',text)
            self.assertNotIn('run.sh monitor',text)
            p.write_text('ROLLING_ENABLED=0\nOFFICIAL_TRIGGER_ENABLED=0\n')
            self.assertNotIn('*/15',schedule.schedule_block())


class TriggerDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.home=Path(self.temp.name)
        for p in (patch.dict(os.environ,{},clear=True),patch.object(controller,'HOME',self.home),
                  patch.object(controller,'ENV_FILE',self.home/'config.env'),
                  patch.object(controller,'notify_failure'),
                  patch.object(requests,'get',side_effect=AssertionError('No live HTTP')),
                  patch.object(requests,'post',side_effect=AssertionError('No live POST'))):
            p.start();self.addCleanup(p.stop)
        (self.home/'config.env').write_text('AUTO_PUSH_ENABLED=1\nOFFICIAL_TRIGGER_ENABLED=1\nROLLING_ENABLED=1\n')
        class Frozen(datetime):
            @classmethod
            def now(cls,tz=None):return NOW
        p=patch.object(controller,'datetime',Frozen);p.start();self.addCleanup(p.stop)
        self.db=controller.database();self.addCleanup(self.db.close)
        self.raw=feeds()
        self.raw['tips']['data']['important']={'t':'短临预报','prompt':'【短临预报】未来两小时请关注降雨变化。',
                                              'd':(NOW-timedelta(minutes=10)).isoformat(),'taskid':123}
        p=patch.object(official,'read_feed',return_value=self.raw['tips']);p.start();self.addCleanup(p.stop)
        self.out=build(raw=self.raw)
        def generate(*args,**kwargs):
            folder=self.home/'drafts'/'one'
            radar.save_result(self.out,folder)
            return {**self.out,'_draft':str(folder)}
        p=patch.object(controller,'generate',side_effect=generate)
        self.generate=p.start();self.addCleanup(p.stop)
        self.args=Namespace(dry_run=False)

    def test_new_official_short_term_sends_three_once_and_marks_only_after_success(self):
        with patch.object(wechat_push,'send_part',return_value='accepted:mock') as send:
            self.assertEqual(controller.nowcast(self.args,self.db),0)
            self.assertEqual(send.call_count,3)
            self.assertEqual(controller.nowcast(self.args,self.db),0)
            self.assertEqual(send.call_count,3)
        self.assertEqual(self.db.execute('SELECT state FROM nowcasts').fetchone()[0],'sent')
        self.assertEqual(self.generate.call_count,1)

    def test_known_failure_reuses_immutable_draft_and_does_not_resend_notice(self):
        with patch.object(wechat_push,'send_part',side_effect=['accepted:mock',wechat_push.RetryableFailure('rejected')]):
            self.assertEqual(controller.nowcast(self.args,self.db),2)
        with patch.object(wechat_push,'send_part',return_value='accepted:mock') as send:
            self.assertEqual(controller.nowcast(self.args,self.db),0)
            self.assertEqual([c.args[0] for c in send.call_args_list],self.out['messages'][1:])
        self.assertEqual(self.generate.call_count,1)

    def test_unknown_result_does_not_resend_or_send_brief(self):
        with patch.object(wechat_push,'send_part',side_effect=['accepted:mock',wechat_push.DeliveryUnknown('timeout')]) as send:
            self.assertEqual(controller.nowcast(self.args,self.db),2)
            self.assertEqual(send.call_count,2)
        with patch.object(wechat_push,'send_part',side_effect=AssertionError('must not send')):
            self.assertEqual(controller.nowcast(self.args,self.db),0)

    def test_official_switch_change_during_delivery_stops_remaining_parts(self):
        def sender(text,before_request):
            self.assertTrue(before_request())
            (self.home/'config.env').write_text('AUTO_PUSH_ENABLED=1\nOFFICIAL_TRIGGER_ENABLED=0\n')
            return 'accepted:mock'
        with patch.object(wechat_push,'send_part',side_effect=sender) as send:
            self.assertEqual(controller.nowcast(self.args,self.db),0)
            self.assertEqual(send.call_count,1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM messages WHERE state='sent'").fetchone()[0],1)

    def test_preview_does_not_create_source_or_delivery_state(self):
        controller.nowcast(Namespace(dry_run=True),self.db)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM nowcasts').fetchone()[0],0)
        self.generate.assert_not_called()

    def test_scheduled_slot_suppresses_rolling_duplicate(self):
        class Frozen(datetime):
            @classmethod
            def now(cls,tz=None):return NOW.replace(hour=19,minute=30)
        with patch.object(controller,'datetime',Frozen):
            self.assertEqual(controller.rolling(self.args,self.db),0)
        self.generate.assert_not_called()

    def test_near_scheduled_slot_waits_without_marking_source_sent(self):
        class Frozen(datetime):
            @classmethod
            def now(cls,tz=None):return NOW.replace(hour=19,minute=15)
        with patch.object(controller,'datetime',Frozen):
            self.assertEqual(controller.nowcast(self.args,self.db),0)
        self.generate.assert_not_called()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM nowcasts').fetchone()[0],0)


if __name__=='__main__':
    unittest.main()
