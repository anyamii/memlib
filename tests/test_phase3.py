from datetime import datetime, timezone
from memlib.index import FakeEmbedder, EnglishTokenizer
from memlib.store import Store
from memlib.recall import Recall, Options

def test_vector_gate_filters_before_fusion_and_one_is_bm25():
    s=Store(':memory:',EnglishTokenizer(),FakeEmbedder())
    with s.db:
        s.index('public','message','one','2020-01-01T00:00:00+00:00','apple tea')
        s.index('public','message','two','2020-01-01T00:00:00+00:00','pear juice')
    recall=Recall(s,{})
    at=datetime(2021,1,1,tzinfo=timezone.utc)
    gated=recall.query('apple',Options(vector_min_score=.95,time_weight=False,fetch=False),at=at)
    ungated=recall.query('apple',Options(time_weight=False,fetch=False),at=at)
    assert len(ungated['debug']['lanes']['vector'])==2
    assert gated['debug']['lanes']['vector']==[]
    assert '2' not in gated['debug']['scores']
    assert all(h['score']>=.95 for h in gated['debug']['lanes']['vector'])
    pure=recall.query('apple',Options(vector=False,time_weight=False,fetch=False),at=at)
    one=recall.query('apple',Options(vector_min_score=1.0,time_weight=False,fetch=False),at=at)
    assert one['debug']['scores']==pure['debug']['scores']
    assert one['debug']['lanes']['keyword']==pure['debug']['lanes']['keyword']
    s.close()
