import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

spec = importlib.util.spec_from_file_location('notify', Path(__file__).resolve().parents[1]/'tools/notify.py')
notify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notify)


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self.fixture.cleanup)
        root = Path(self.fixture.name)
        self.project = root/'project'
        self.project.mkdir()
        self.sessions = root/'sessions'
        self.thread = '12345678-1234-1234-1234-123456789abc'
        self.path = self.sessions/'2026/09/05'/f'rollout-{self.thread}.jsonl'
        self.path.parent.mkdir(parents=True)
        self.meta = {'id':self.thread, 'source':'vscode', 'cwd':str(self.project)}
        self.write_session()
        self.event = {'type':'agent-turn-complete', 'cwd':str(self.project),
                      'thread-id':self.thread, 'turn-id':'turn-1',
                      'last-assistant-message':'private research result',
                      'input-messages':['private user prompt']}
        self.sent = []
        self.options = dict(project=self.project, sessions=self.sessions,
                            topic_file=root/'topic', state_file=root/'state.sqlite3',
                            sender=lambda topic: self.sent.append('sent'))

    def write_session(self):
        records = [{'type':'session_meta', 'payload':self.meta},
                   {'type':'turn_context', 'payload':{'turn_id':'turn-1'}}]
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in records))

    def test_only_main_agent_in_this_project_is_notified(self):
        self.meta['source'] = {'subagent':{'thread_spawn':{'depth':1}}}
        self.meta['agent_path'] = '/root/builder'
        self.write_session()
        self.assertEqual(notify.deliver(self.event, **self.options)['state'], 'skipped')
        self.assertEqual(self.sent, [])
        self.meta.update(source='vscode', agent_path=None)
        self.write_session()
        for change in ({'cwd':str(self.project)+'-other'}, {'cwd':''}, {'cwd':None},
                       {'type':'approval-requested'}, {'turn-id':''},
                       {'thread-id':'../../private'}):
            self.assertEqual(notify.deliver(self.event | change, **self.options)['state'], 'skipped')
        self.assertEqual(self.sent, [])
        self.assertEqual(notify.deliver(self.event, **self.options)['state'], 'sent')

    def test_explicit_and_native_completion_share_dedup_but_new_turn_sends(self):
        direct = notify.completed_event(self.thread, self.sessions, self.project)
        self.assertEqual(notify.deliver(direct, **self.options)['state'], 'sent')
        self.assertEqual(notify.deliver(self.event, **self.options)['state'], 'duplicate')
        self.assertEqual(notify.deliver(self.event | {'turn-id':'turn-2'}, **self.options)['state'], 'sent')
        self.assertEqual(len(self.sent), 2)

    def test_failure_is_redacted_and_can_retry(self):
        def failing_sender(topic):
            raise URLError('https://ntfy.sh/private-topic-secret')
        result = notify.deliver(self.event, **(self.options | {'sender':failing_sender}))
        self.assertEqual(result, {'state':'failed', 'error':'URLError'})
        self.assertEqual(notify.deliver(self.event, **self.options)['state'], 'sent')

    def test_concurrent_callbacks_send_once(self):
        start = threading.Barrier(2)
        def callback():
            start.wait(timeout=5)
            return notify.deliver(self.event, **self.options)['state']
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(callback) for _ in range(2)]
            self.assertCountEqual([r.result(timeout=10) for r in results], ['sent', 'duplicate'])
        self.assertEqual(self.sent, ['sent'])

    def test_direct_dispatch_uses_latest_turn_context(self):
        with self.path.open('a') as handle:
            handle.write(json.dumps({'type':'turn_context', 'payload':{'turn_id':'turn-2'}})+'\n')
        event = notify.completed_event(self.thread, self.sessions, self.project)
        self.assertEqual(event['turn-id'], 'turn-2')

    def test_publish_sends_only_fixed_notice_and_has_timeout(self):
        topic = self.options['topic_file']
        topic.write_text('a-random-test-topic-1234567890')
        with patch.object(notify.urllib.request, 'urlopen') as send:
            send.return_value.__enter__.return_value.status = 200
            notify.publish(topic)
        request = send.call_args.args[0]
        self.assertEqual(request.data.decode(), notify.BODY)
        self.assertNotIn('private', request.data.decode())
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(send.call_args.kwargs['timeout'], 8)


if __name__ == '__main__':
    unittest.main()
