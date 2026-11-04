"""Synthetic gates: OOV anchors, substring df, empty queries and final Pack."""
import math
import pytest
from memlib import KeytermGate, Options, Recall, Store
from memlib.adapters import Record
from memlib.index import EnglishTokenizer, JiebaTokenizer
from scripts.calibrate_gate import Adapter, NoEncoder


def test_oov_anchor_and_top_two_have_distinct_behavior():
    gate = KeytermGate(['common anchor', 'common context'], EnglishTokenizer())
    assert gate.idf('anchor') == math.log(3/2)+1
    assert gate.idf('missing') == math.log(3)+1
    features = gate.features('common missing anchor missing', ['common anchor'])
    assert features['keyword_count'] == 3
    assert not features['k1'] and features['k2']
    assert gate.reason(features) == 'keyterm'
    gate.mode = 'K2'
    assert gate.reason(features) is None
    gate.mode = 'K3'
    gate.coverage_min = features['coverage']
    assert gate.reason(features) is None
    gate.coverage_min = math.nextafter(features['coverage'], 1)
    assert gate.reason(features) == 'keyterm'


def test_substring_df_stopwords_and_fragment_boundaries():
    gate = KeytermGate(['foobar foobar', 'bar'], EnglishTokenizer(), ['please'])
    assert gate.idf('foo') == math.log(3/2)+1
    assert gate.idf('bar') == 1
    assert gate.keywords('Please a foo foo') == ['foo']
    assert not gate.features('foobar', ['foo', 'bar'])['k1']
    for mode in ('K1', 'K2', 'K3'):
        gate.mode = mode
        assert gate.reason(gate.features('please', ['please'])) == 'no_keywords'


def test_tied_oov_idf_uses_query_order():
    gate = KeytermGate(['common'], EnglishTokenizer())
    assert not gate.features('missing absent', ['absent'])['k1']
    assert gate.features('absent missing', ['absent'])['k1']


def test_jieba_dictionary_isolated_and_versioned(tmp_path):
    path = tmp_path/'dictionary.txt'
    path.write_text('银狐星港 100000\n')
    before = JiebaTokenizer()
    custom = JiebaTokenizer(path, hmm=True)
    assert custom.tokens('银狐星港') == ['银狐星港']
    assert before.tokens('银狐星港') != ['银狐星港']
    assert custom.version != before.version


def test_gate_runs_on_final_pack_but_checks_original_text():
    record = Record('calibration', 'one', '2026-01-01', 'calibration', 'common anchor')
    store = Store(':memory:', tokenizer=EnglishTokenizer(), embedder=NoEncoder())
    try:
        store.db.execute("INSERT INTO items(ns,kind,id,date,vector,tokenizer_version,embedder_version) VALUES(?,?,?,?,?,?,?)",
                         (record.ns, 'message', record.id, record.time, '[]', store.tokenizer.version, store.embedder.version))
        store.db.execute('INSERT INTO keyword(rowid,tokens) VALUES(1,?)', (record.text,))
        recall = Recall(store, {'calibration': Adapter([record])})
        gate = KeytermGate([record.text], EnglishTokenizer())
        options = Options(vector=False, time_weight=False, max_chars=6, keyterm_gate=gate)
        # anchor is beyond the delivered text budget but in this Pack fragment.
        out = recall.query('anchor common', options)
        assert out['items'][0]['text'] == 'common'
        assert out['debug']['keyterm_reason'] is None
        out = recall.query('missing common', options)
        assert not out['items']
        assert out['debug']['keyterm_reason'] == 'keyterm'
        options.max_items = 0
        out = recall.query('anchor common', options)
        assert not out['items'] and out['debug']['keyterm_reason'] == 'keyterm'
    finally:
        store.close()


@pytest.mark.parametrize('mode,coverage', [('bad', 0), ('K3', -1), ('K3', float('nan'))])
def test_invalid_config(mode, coverage):
    with pytest.raises(ValueError):
        KeytermGate([], EnglishTokenizer(), mode=mode, coverage_min=coverage)
