from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import manage_schedule as schedule

class RadarScheduleTests(unittest.TestCase):
    def test_three_actual_half_hour_publications_and_alerts_off(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(schedule,'HOME',Path(folder)):
            (Path(folder)/'config.env').write_text('AUTO_PUSH_ENABLED=0\nRAIN_ALERTS_ENABLED=0\n')
            text=schedule.schedule_block()
            for hour in (7,15,19):
                self.assertIn(f'30 {hour} * * *',text)
            self.assertNotIn('*/5 * * * *',text)

    def test_five_minute_alert_check_is_separately_configured(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(schedule,'HOME',Path(folder)):
            (Path(folder)/'config.env').write_text('AUTO_PUSH_ENABLED=1\nRAIN_ALERTS_ENABLED=1\n')
            self.assertIn('*/5 * * * *',schedule.schedule_block())

    def test_upgrade_preserves_other_projects_and_removes_legacy_block(self):
        original='0 1 * * * other-project\n'+schedule.LEGACY_BEGIN+'\nold-task\n'+schedule.LEGACY_END+'\n'
        self.assertEqual(schedule.without_block(original),'0 1 * * * other-project')

if __name__=='__main__':
    unittest.main()
