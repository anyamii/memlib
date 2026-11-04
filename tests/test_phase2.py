import time
from datetime import datetime, timezone
import pytest
from memlib.deadline import Deadline
from memlib.index import JiebaTokenizer, EnglishTokenizer, rrf, FakeEmbedder
from memlib.store import Store
from memlib.adapters import Record
from memlib.recall import Recall, Options
from eval.run import metrics
from adapters.locomo import LoCoMoAdapter

def test_deadline_bounded_and_no_worker_flood():
    boundary=Deadline(); start=time.monotonic()
    with pytest.raises(TimeoutError): boundary.call(lambda: time.sleep(.3),start+.02)
    with pytest.raises(TimeoutError): boundary.call(lambda: 1,time.monotonic()+1)
    assert time.monotonic()-start < .15

@pytest.mark.parametrize('kind',['source','encoder'])
def test_real_deadlines_empty_pack(kind):
    class Adapter:
        def get_by_ids(self,ids): time.sleep(.3); return {}
    class Slow(FakeEmbedder):
        def encode(self,texts): time.sleep(.3); return super().encode(texts)
    store=Store(':memory:'); store.run_batch('b','x',[Record('x','1','2020-01-01T00:00:00+00:00','a','hello')],'1','1')
    if kind=='encoder': store.embedder=Slow()
    start=time.monotonic(); result=Recall(store,{'x':Adapter()}).query('hello',Options(timeout_seconds=.025))
    assert time.monotonic()-start < .15
    assert result['items']==[] and result['debug']['timed_out']
    store.close()

def test_duplicate_and_time_filter():
    class Adapter:
        def get_by_ids(self,ids): return {r.id:r for r in records if r.id in ids}
    records=[Record('x',str(i),date,'a',text) for i,date,text in [(1,'2020-01-01T00:00:00+00:00','hello there!'),(2,'2020-01-01T00:00:00+00:00','Hello there.'),(3,'2030-01-01T00:00:00+00:00','hello future')]]
    store=Store(':memory:'); assert store.run_batch('b','x',records,'1','3')
    result=Recall(store,{'x':Adapter()}).query('hello',at=datetime(2025,1,1,tzinfo=timezone.utc))
    assert len(result['items'])==1
    assert '3' not in [x['id'] for x in result['debug']['scores'].values()]
    store.close()

def test_tokenizers_and_zero_k():
    assert EnglishTokenizer().tokens('Hello, HELLO!')==['hello','hello']
    tokens=JiebaTokenizer().tokens('我在北京喝咖啡 Hello')
    assert '北京' in tokens and 'hello' in tokens
    assert rrf([[1],[1]],0)=={1:2}

def test_gold_metrics_multihop():
    assert metrics(['a','b'],['x','a'])==(.5,.5,.5,0)

def test_locomo_mapping_without_answers():
    sample={'sample_id':'s','conversation':{'session_1_date_time':'1:56 pm on 8 May, 2023','session_1':[{'dia_id':'D1:1','speaker':'A','text':'public fixture'}]}}
    adapter=LoCoMoAdapter(sample)
    qa=[{'question':'what','answer':'DO NOT INDEX','category':1,'evidence':['D1:1']}]
    assert list(adapter.questions(qa))[0]['gold']==['locomo-s:D1:1']
    assert all('DO NOT INDEX' not in r.text for r in adapter.records)

@pytest.mark.parametrize('kind',['source','encoder'])
def test_batch_deadline_isolated_failure(kind):
    class Slow(FakeEmbedder):
        def encode(self,texts): time.sleep(.3); return super().encode(texts)
    def records():
        if kind=='source': time.sleep(.3)
        yield Record('x','1','2020-01-01T00:00:00+00:00','a','hello')
    store=Store(':memory:',embedder=Slow() if kind=='encoder' else None,operation_timeout=.02)
    start=time.monotonic()
    assert not store.run_batch('b','x',records(),'1','1')
    assert time.monotonic()-start < .15
    assert store.db.execute('SELECT status FROM batches').fetchone()[0]=='failed'
    assert store.db.execute('SELECT count(*) FROM items').fetchone()[0]==0
    store.close()
