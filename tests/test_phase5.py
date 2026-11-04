import json
import math
from pathlib import Path

import pytest

from memlib import Gate, Options, Recall, Store
from memlib.index import BigramTokenizer, EnglishTokenizer
from scripts.calibrate_gate import calibrate, collect, metrics

ROOT = Path(__file__).resolve().parents[1]


def source_adapter():
    from scripts.calibrate_gate import Adapter
    from memlib.adapters import Record
    return Adapter([Record('test', str(i), '2026-01-01', '虚构', text) for i, text in enumerate(['rare rare flower', 'blue ocean', 'green forest', 'red apple'])])


@pytest.fixture
def store():
    from scripts.calibrate_gate import NoEncoder
    s = Store(':memory:', tokenizer=EnglishTokenizer(), embedder=NoEncoder())
    # No embedding calls, source reads, or persistent source corpus needed.
    for i, text in enumerate(['rare rare flower', 'blue ocean', 'green forest', 'red apple']):
        cur = s.db.execute("INSERT INTO items(ns,kind,id,date,vector,tokenizer_version,embedder_version) VALUES(?,?,?,?,?,?,?)", ('test', 'message', str(i), '2026-01-01', '[]', s.tokenizer.version, s.embedder.version))
        s.db.execute('INSERT INTO keyword(rowid,tokens) VALUES(?,?)', (cur.lastrowid, text))
    yield s
    s.close()


def test_gate_features_oov_dedup_equality_and_no_fetch(store):
    out = Recall(store, {}).query('rare rare missing', Options(vector=False, gate=Gate(abs_min=1e-300)))
    d = out['debug']
    denominator = math.log(3.5/1.5) + math.log(4.5/.5)
    assert d['F'] == pytest.approx(d['S1']/denominator)
    assert Gate(d['S1'], d['F']).reason(d['S1'], d['F'], True) is None
    class Forbidden:
        def get_by_ids(self, ids):
            raise AssertionError('rejection must precede source access')
    recall = Recall(store, {'test': Forbidden()})
    for query, gate, reason in [('rare', Gate(abs_min=1e9), 'abs'), ('rare', Gate(norm_min=1e9), 'norm'), ('missing', Gate(norm_min=.1), 'no_hit'), ('', Gate(abs_min=.1), 'no_hit')]:
        out = recall.query(query, Options(vector=False, gate=gate))
        assert out['items'] == [] and out['debug']['gate_reason'] == reason
        assert '6_fetch' not in out['debug']['counts']


def test_gate_scope_and_global_idf(store):
    store.db.execute("UPDATE items SET ns='blocked' WHERE id='0'")
    out = Recall(store, {}).query('rare', Options(vector=False, gate=Gate(norm_min=.01)), namespaces=['test'])
    assert out['debug']['gate_reason'] == 'no_hit'
    # SQLite scores use global corpus statistics even when scope hides rows.
    out = Recall(store, {}).query('blue', Options(vector=False, gate=Gate(norm_min=.01)), namespaces=['test'])
    assert out['debug']['F'] == pytest.approx(out['debug']['S1']/math.log(3.5/1.5))
    store.tombstone('message', '1', 'test')
    assert Recall(store, {}).query('blue', Options(vector=False, gate=Gate(norm_min=.01)))['debug']['gate_reason'] == 'no_hit'


def test_gate_allows_real_pack_and_preserves_ranking(store):
    recall = Recall(store, {'test': source_adapter()})
    baseline = recall.query('rare blue', Options(vector=False, time_weight=False))
    accepted = recall.query('rare blue', Options(vector=False, time_weight=False, gate=Gate(abs_min=1e-300)))
    assert accepted['items'] == baseline['items']
    assert accepted['items'] and accepted['debug']['gate_reason'] is None
    assert accepted['debug']['lanes'] == baseline['debug']['lanes']
    assert accepted['debug']['scores'] == baseline['debug']['scores']


@pytest.mark.parametrize('value', [-1, float('inf'), float('nan')])
def test_gate_invalid(value):
    with pytest.raises(ValueError):
        Gate(norm_min=value)


def test_gate_requires_keyword():
    with pytest.raises(ValueError):
        Options(keyword=False, gate=Gate(abs_min=1))


def test_fts_chinese_and_bigram(store):
    store.db.execute('CREATE VIRTUAL TABLE temp.raw USING fts5(body)')
    store.db.execute("INSERT INTO temp.raw VALUES('星砂花店向日葵')")
    assert not store.db.execute("SELECT rowid FROM temp.raw WHERE raw MATCH '向日葵'").fetchall()
    assert store.db.execute("SELECT rowid FROM temp.raw WHERE raw MATCH '星砂花店向日葵'").fetchall()
    t = BigramTokenizer()
    assert t.tokens('星砂花店 ABC 123') == ['星砂', '砂花', '花店', 'abc', '123']
    assert t.tokens('花') == ['花']
    assert EnglishTokenizer().tokens('Hello, WORLD 123') == ['hello', 'world', '123']
    store.db.execute("DELETE FROM keyword")
    store.db.execute('INSERT INTO keyword(rowid,tokens) VALUES(1,?)', (' '.join(t.tokens('星砂花店向日葵')),))
    assert store.db.execute("SELECT rowid FROM keyword WHERE keyword MATCH '向日 OR 日葵'").fetchall()


def test_calibration_fixture_and_holdout():
    read = lambda name: json.loads((ROOT/'examples/phase5'/f'{name}.json').read_text())
    assert len(read('corpus')) == 40
    assert len(read('positives')) == len(read('negatives')) == 20
    rows = collect(read('corpus'), read('positives'), read('negatives'), BigramTokenizer())
    assert all(r['pack'] for r in rows if r['positive'])
    report = calibrate(rows)
    assert report['full']['selected']['false_reject_rate'] <= .07
    assert len(report['holdout']) == 2
    for fold in report['holdout']:
        chosen = fold['training']['selected']
        test = [r for r in rows if r['group'] % 2 != fold['train_parity']]
        assert fold['test'] == metrics(test, chosen['abs_min'], chosen['norm_min'])
    assert metrics(rows, 1e9, 0)['false_reject_rate'] == 1
    assert metrics(rows, 0, 1e9)['false_emit_rate'] == 0


def test_calibration_infeasible_and_split_validation():
    rows = [dict(group=g, positive=p, S1=0, F=0, has_hit=False, pack=False) for g in (1, 2) for p in (True, False)]
    report = calibrate(rows)
    assert report['full']['selected'] is None
    assert all(f['test'] is None for f in report['holdout'])
    with pytest.raises(ValueError):
        calibrate(rows[:2])


def test_zero_gate_bytes_match_original(store):
    # Freeze the pre-phase5 implementation from the branch's base commit.
    original = json.loads((ROOT/'tests/fixtures/phase5-baseline.json').read_text())
    recall = Recall(store, {'test': source_adapter()})
    for query, expected in original.items():
        actual = recall.query(query, Options(vector=False, time_weight=False))
        assert json.dumps(actual, ensure_ascii=False).encode() == expected.encode()
        assert recall.query(query, Options(vector=False, time_weight=False, gate=Gate(0, 0))) == actual


def test_actual_fts_term_normalization(store):
    from memlib.gate import features
    s1, f = features(store.db, ['RARE', 'rare', 'café', 'cafe'], [(1, 2.0)])
    assert s1 == 2
    assert f == pytest.approx(2/(math.log(3.5/1.5)+math.log(4.5/.5)))


def test_holdout_transfers_without_retuning():
    rows = [dict(group=g, positive=p, S1=s, F=s, has_hit=True, pack=True)
            for g, s in [(0, 10), (1, 1)] for p in (True, False)]
    # Add easy negatives so thresholds selected on even groups differ from odd.
    rows += [dict(group=g, positive=False, S1=s, F=s, has_hit=True, pack=True)
             for g, s in [(0, 5), (1, .5)]]
    report = calibrate(rows, 0)
    even = report['holdout'][0]
    assert even['training']['selected']['norm_min'] == 10
    assert even['test']['false_reject_rate'] == 1
    assert not even['meets_target']
