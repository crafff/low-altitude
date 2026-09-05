#!/usr/bin/env python3
"""Notify the existing ntfy subscriber after a verified main-agent turn ends."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
SESSIONS = Path.home() / '.codex' / 'sessions'
TOPIC_FILE = PROJECT.parent / '.codex-tools' / 'ntfy_topic'
STATE_FILE = PROJECT / '.cache' / 'notifications.sqlite3'
UUID = re.compile(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
BODY = '低空交通项目的主 agent 已准备好本轮结果，请打开 Codex 查看回复。'


def belongs_to_project(cwd, project):
    return isinstance(cwd, str) and bool(cwd) and Path(cwd).resolve().is_relative_to(project.resolve())


def main_session(thread_id, sessions, project):
    """Read only the matching session's metadata, never a supplied transcript path."""
    if not isinstance(thread_id, str) or not UUID.fullmatch(thread_id):
        return None
    for path in sessions.glob(f'*/*/*/*-{thread_id}.jsonl'):
        with path.open() as handle:
            meta = json.loads(handle.readline(65536))
        if meta.get('type') != 'session_meta':
            continue
        meta = meta['payload']
        source = meta.get('source')
        # Native subagents have a structured source and/or a non-root agent path.
        if (meta.get('id') == thread_id and isinstance(source, str)
                and source in {'cli', 'vscode', 'exec', 'appServer', 'app-server', 'desktop'}
                and meta.get('agent_path') in {None, '/root'}
                and belongs_to_project(meta.get('cwd'), project)):
            return path
    return None


def completed_event(thread_id, sessions, project):
    """Explicit controller dispatch for clients that omit local notify events."""
    path = main_session(thread_id, sessions, project)
    if path is None:
        return None
    turn_id = None
    # Select metadata records only; do not export or log prompts/tool output.
    with path.open() as handle:
        for line in handle:
            if re.search(r'"type"\s*:\s*"turn_context"', line):
                record = json.loads(line)
                if record.get('type') == 'turn_context':
                    turn_id = record['payload'].get('turn_id')
    return {'type': 'agent-turn-complete', 'thread-id': thread_id,
            'turn-id': turn_id, 'cwd': str(project)}


def publish(topic_file):
    topic = topic_file.read_text().strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{24,128}', topic):
        raise ValueError('invalid notification topic')
    request = urllib.request.Request(
        'https://ntfy.sh/' + topic, data=BODY.encode('utf-8'), method='POST',
        headers={'Title': 'Low-altitude: response ready', 'Priority': 'default',
                 'Tags': 'robot', 'Content-Type': 'text/plain; charset=utf-8'})
    with urllib.request.urlopen(request, timeout=8) as response:
        if not 200 <= response.status < 300:
            raise OSError('notification service did not accept the message')


def deliver(event, *, project=PROJECT, sessions=SESSIONS, topic_file=TOPIC_FILE,
            state_file=STATE_FILE, sender=publish):
    if (not isinstance(event, dict) or event.get('type') != 'agent-turn-complete'
            or not belongs_to_project(event.get('cwd'), project)):
        return {'state': 'skipped', 'reason': 'unrelated_event'}
    thread_id, turn_id = event.get('thread-id'), event.get('turn-id')
    if main_session(thread_id, sessions, project) is None:
        return {'state': 'skipped', 'reason': 'not_verified_main_agent'}
    if not isinstance(turn_id, str) or not turn_id or len(turn_id) > 200:
        return {'state': 'skipped', 'reason': 'missing_turn_id'}
    key = hashlib.sha256(f'{thread_id}\0{turn_id}'.encode()).hexdigest()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # Serialize simultaneous explicit/native completion events for the same turn.
    with sqlite3.connect(state_file, timeout=10) as db:
        db.execute('CREATE TABLE IF NOT EXISTS deliveries '
                   '(id TEXT PRIMARY KEY, sent_at TEXT NOT NULL)')
        db.execute('BEGIN IMMEDIATE')
        if db.execute('SELECT 1 FROM deliveries WHERE id = ?', (key,)).fetchone():
            return {'state': 'duplicate'}
        try:
            sender(topic_file)
        except Exception as exc:
            # Exception strings may contain the private topic URL. Keep only type.
            return {'state': 'failed', 'error': type(exc).__name__}
        db.execute('INSERT INTO deliveries VALUES (?, ?)',
                   (key, datetime.now(timezone.utc).isoformat()))
        db.execute('DELETE FROM deliveries WHERE id NOT IN '
                   '(SELECT id FROM deliveries ORDER BY sent_at DESC LIMIT 4096)')
    return {'state': 'sent'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true', help='Main agent: call immediately before final handoff')
    parser.add_argument('event', nargs='?', help='Native notify JSON argument')
    args = parser.parse_args()
    try:
        if args.complete:
            if args.event:
                parser.error('use --complete or the native event, not both')
            event = completed_event(os.environ.get('CODEX_THREAD_ID'), SESSIONS, PROJECT)
        else:
            event = json.loads(args.event) if args.event else None
        result = deliver(event)
    except Exception as exc:
        result = {'state': 'failed', 'error': type(exc).__name__}
    print(json.dumps(result))
    # Native notification failures must never fail or continue an agent turn.
    return int(args.complete and result['state'] not in {'sent', 'duplicate'})


if __name__ == '__main__':
    raise SystemExit(main())
