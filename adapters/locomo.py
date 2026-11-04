"""LoCoMo text-only adapter. Never reads answers into indexing records.

Source: Maharana et al., https://github.com/snap-research/locomo,
CC BY-NC 4.0. Images, generated observations and summaries are excluded.
Original QA has no asked_at: benchmark policy assigns last-session + 1 day,
explicitly a synthetic full-history cutoff, not measured question chronology.
"""
from datetime import datetime, timezone, timedelta
import re
from memlib.adapters import Record

class LoCoMoAdapter:
    def __init__(self, sample):
        self.ns = 'locomo-' + str(sample['sample_id'])
        self.records = []
        conversation = sample['conversation']
        for key, turns in conversation.items():
            if not re.fullmatch(r'session_\d+', key):
                continue
            stamp = datetime.strptime(conversation[key+'_date_time'], '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).isoformat()
            for turn in turns:
                self.records.append(Record(self.ns, turn['dia_id'], stamp, turn['speaker'], turn['text']))
        self.records.sort(key=lambda r: (r.time, r.id))
        self.asked_at = (max(datetime.fromisoformat(r.time) for r in self.records)+timedelta(days=1)).isoformat()
    def iter_since(self, cursor=None):
        start = 0 if cursor is None else [r.id for r in self.records].index(cursor)+1
        yield from self.records[start:]
    def get_by_ids(self, ids):
        wanted = set(ids)
        return {r.id:r for r in self.records if r.id in wanted}
    def questions(self, qa):
        known = {r.id for r in self.records}
        for i, q in enumerate(qa):
            gold = q.get('evidence', [])
            # Categories without usable evidence are explicitly excluded.
            if q['category'] == 5 or not gold or any(x not in known for x in gold):
                continue
            yield dict(qid=f'{self.ns}-q{i}', query=q['question'], asked_at=self.asked_at,
                       gold=[self.ns+':'+x for x in dict.fromkeys(gold)], type=str(q['category']))
