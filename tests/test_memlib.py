import json
from pathlib import Path
import subprocess
import sqlite3
import pytest
from memlib import Store, Recall, Options
from memlib.adapters import FixtureAdapter, Record
from memlib.index import rrf, split_spans, FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path/'test.db')
    yield s
    s.close()

class Adapter:
    def __init__(self, records):
        self.records = {r.id:r for r in records}
    def get_by_ids(self, ids):
        return {id:self.records[id] for id in ids if id in self.records}

def message(store, id='m1', text='栀子花盆土检查', ns='test'):
    r = Record(ns,id,'2026-10-01','占位甲',text)
    assert store.run_batch(id,ns,[r],id,id)
    return r


def test_bad_batch_is_atomic_later_batch_and_retry(store):
    records = [Record('test',str(i),'2026-10-01','占位甲','合成栀子花') for i in range(1,4)]
    assert store.run_batch('1','test',[records[0]],'1','1')
    assert not store.run_batch('2','test',[records[1]],'2','2',fail=True)
    assert store.run_batch('3','test',[records[2]],'3','3')
    assert [r[0] for r in store.db.execute('SELECT status FROM batches ORDER BY batch_id')] == ['success','failed','success']
    assert store.db.execute("SELECT error FROM batches WHERE batch_id='2'").fetchone()[0]=='synthetic_failure_after_first_record'
    assert store.db.execute("SELECT count(*) FROM items").fetchone()[0]==2
    assert store.run_batch('2','test',[records[1]],'2','2')
    assert store.run_batch('2','test',[records[1]],'2','2')
    assert store.db.execute('SELECT count(*) FROM items').fetchone()[0]==3

@pytest.mark.parametrize('kind',['message','impression','fragment'])
@pytest.mark.parametrize('during',[False,True])
def test_tombstones_both_gates(store,kind,during):
    r = message(store)
    store.put_impression('n','2026-10-01','合成','栀子花盆土检查')
    fid = store.db.execute('SELECT id FROM fragments').fetchone()[0]
    target,ns = {'message':('m1','test'),'impression':('n','impressions'),'fragment':(fid,'impressions')}[kind]
    block = lambda: store.tombstone(kind,target,ns)
    if not during:
        block()
    out = Recall(store,{'test':Adapter([r])}).query('栀子花',after_fetch=block if during else None)
    forbidden = 'm1' if kind=='message' else fid
    assert forbidden not in [x['id'] for x in out['items']]
    counts = out['debug']['counts']
    if during:
        assert counts['1_eligible']==2
        assert counts['6_post_tombstone']==counts['6_fetch']-1
    else:
        assert counts['1_eligible']==1


def test_revision_history_and_contentless_delete(store):
    store.put_impression('n','2026-10-01','合成','旧词蓝鲸')
    oldid = store.db.execute('SELECT id FROM fragments').fetchone()[0]
    assert store.put_impression('n','2026-10-01','合成','新词火山')==2
    assert store.db.execute('SELECT body FROM impression_history').fetchone()[0]=='旧词蓝鲸'
    assert store.db.execute('SELECT tokens FROM keyword').fetchone()[0] is None
    opts = Options(vector=False)
    recall = Recall(store,{})
    assert not recall.query('蓝鲸',opts)['items']
    out = recall.query('火山',opts)['items']
    assert out and out[0]['id']!=oldid and out[0]['text']=='新词火山'
    assert store.put_impression('n','2026-10-01','合成','新词火山')==2
    store.delete_impression('n')
    assert not recall.query('火山')['items']


def test_rrf_double_hit_and_small_k():
    # Shared third-place beats single-lane first at k=60; small k favors first.
    lanes = [['solo','x','both'],['a','b','both']]
    assert rrf(lanes,60)['both']>rrf(lanes,60)['solo']
    assert rrf(lanes,0.1)['solo']>rrf(lanes,0.1)['both']


def test_parent_cap_budget_and_source_label(store):
    store.put_impression('n','2026-10-01','合成','\n\n'.join('栀子花盆土检查'+str(i) for i in range(6)))
    recall = Recall(store,{})
    out = recall.query('栀子花',Options(max_items=20,max_chars=1000))
    assert len(out['items'])==2
    assert all(x['source_note']=='AI 当时写的叙述' for x in out['items'])
    out = recall.query('栀子花',Options(parent_cap=False,max_items=3,max_chars=14))
    assert len(out['items'])<=3
    assert sum(len(x['text']) for x in out['items'])==14
    assert not recall.query('栀子花',Options(max_items=0))['items']
    assert len(recall.query('栀子花',Options(parent_cap=False,budget=False))['items'])==6


def test_namespace_and_top_n_filter(store):
    r1 = message(store,'same','栀子花','a')
    r2 = message(store,'same','栀子花','b')
    store.tombstone('message','same','a')
    recall = Recall(store,{'a':Adapter([r1]),'b':Adapter([r2])})
    out = recall.query('栀子花',Options(n=1))
    assert [x['ns'] for x in out['items']]==['b']
    assert not recall.query('栀子花',namespaces=['a'])['items']


def test_missing_source_and_timeout(store):
    r = message(store)
    recall = Recall(store,{'test':Adapter([])})
    assert not recall.query('栀子花')['items']
    assert not Recall(store,{}).query('栀子花')['items']
    assert Recall(store,{}).query('栀子花')['debug']['source_errors']==1
    out = Recall(store,{'test':Adapter([r])}).query('栀子花',Options(timeout_seconds=0))
    assert out['items']==[] and out['debug']['timed_out']
    out = Recall(store,{'test':Adapter([r])}).query('栀子花',after_fetch=lambda: __import__('time').sleep(.02),options=Options(timeout_seconds=.01))
    assert out['items']==[] and out['debug']['timed_out']


def test_edit_during_recall_rejects_old_spans(store):
    store.put_impression('n','2026-10-01','合成','栀子花')
    out = Recall(store,{}).query('栀子花',after_fetch=lambda:store.put_impression('n','2026-10-01','合成','新火山'))
    assert not out['items']


def test_splitting_offsets_punctuation_and_hard_limit():
    body = '短段。\n\n'+'甲'*220+'。'+'乙'*220+'！'+ '\n\n尾段'
    spans = list(split_spans(body))
    assert [body[a:b] for a,b in spans]==['短段。','甲'*220+'。','乙'*220+'！','尾段']
    assert max(b-a for a,b in split_spans('甲'*901))<=400


def test_fixture_cursor_and_synthetic_examples():
    adapter = FixtureAdapter(ROOT/'examples/messages.jsonl')
    records = list(adapter.iter_since(None))
    assert len(records)==315
    assert len(list(adapter.iter_since(records[0].id)))==314
    assert adapter.get_by_ids(['missing',records[0].id])=={records[0].id:records[0]}
    with pytest.raises(ValueError):
        list(adapter.iter_since('missing'))
    assert len(json.loads((ROOT/'examples/queries.json').read_text()))==5


def test_no_message_prose_stored(store,tmp_path):
    secret='独一无二的合成正文标记QZ139'
    message(store,text=secret)
    store.db.commit()
    dump='\n'.join(store.db.iterdump())
    assert secret not in dump
    assert secret.encode() not in (tmp_path/'test.db').read_bytes()
    assert not {'text','body','tokens'} & {r[1] for r in store.db.execute('PRAGMA table_info(items)')}


def test_fake_vectors_deterministic_and_versions(store):
    e=FakeEmbedder()
    assert e.encode(['甲','乙'])==e.encode(['甲','乙'])
    message(store)
    store.db.execute("UPDATE items SET embedder_version='other'")
    with pytest.raises(ValueError,match='version mismatch'):
        Recall(store,{}).query('栀子花')


def test_time_weight_monotonic_and_lane_switches(store):
    from datetime import datetime,timezone
    a=message(store,'old')
    b=message(store,'new')
    store.db.execute("UPDATE items SET date='2026-10-29' WHERE id='new'")
    recall=Recall(store,{'test':Adapter([a,b])})
    out=recall.query('栀子花',at=datetime(2026,10,30,tzinfo=timezone.utc))
    scores=out['debug']['scores'].values()
    ratios={r['id']:r['weighted']/r['rrf'] for r in scores}
    assert ratios['new']>ratios['old']
    assert not recall.query('栀子花',Options(keyword=False,vector=False))['items']
    assert not recall.query('栀子花',Options(fetch=False))['items']


def test_git_ignores_runtime_and_no_tracked_indexes():
    ignored = subprocess.check_output(['git','check-ignore','sample.db','sample.db-wal','sample.sqlite','sample.index','sample.vec'],cwd=ROOT,text=True).splitlines()
    assert len(ignored)==5
    tracked = subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    assert not any(p.endswith(('.db','.sqlite','.index','.vec')) or '.db-' in p for p in tracked)


def test_keyword_only_checks_tokenizer_version(store):
    message(store)
    store.db.execute("UPDATE items SET tokenizer_version='other'")
    with pytest.raises(ValueError,match='version mismatch'):
        Recall(store,{}).query('栀子花',Options(vector=False))


def test_failed_reindex_preserves_previous_published_batch(store):
    original=message(store,text='原先合成栀子花')
    replacement=Record('test','m1','2026-10-02','占位甲','替换合成火山')
    assert not store.run_batch('m1','test',[replacement],'m1','m1',fail=True)
    recall=Recall(store,{'test':Adapter([original])})
    assert recall.query('栀子花',Options(vector=False))['items']
    assert not recall.query('火山',Options(vector=False))['items']
