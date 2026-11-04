"""Synthetic checks for shared-index query holdout and fragment packing."""
import pytest
from memlib.index import EnglishTokenizer
from scripts.calibrate_gate import collect


def test_shared_index_fold_labels_do_not_change_scores():
    corpus = [dict(group=0, id='a', text='silver fox'), dict(group=0, id='b', text='blue ocean')]
    positives = [dict(group=0, query='silver'), dict(group=1, query='silver')]
    rows = collect(corpus, positives, [dict(group=1, query='absent')], EnglishTokenizer(), shared_corpus=True)
    assert [r['group'] for r in rows] == [0, 1, 1]
    assert rows[0]['S1'] == rows[1]['S1'] > 0
    assert rows[0]['F'] == rows[1]['F'] > 0
    assert rows[0]['delivered_ids'] == rows[1]['delivered_ids'] == ['a']
    assert not rows[2]['pack']


def test_fragment_offsets_and_parent_cap():
    texts = ['silver fox', 'silver mountain', 'silver river']
    body = '\n\n'.join(texts)
    corpus = []
    start = 0
    for seq, text in enumerate(texts):
        corpus.append(dict(group=0, id=str(seq), text=text, parent_id='parent', parent_text=body, seq=seq, start=start, end=start+len(text)))
        start += len(text)+2
    rows = collect(corpus, [dict(group=0, query='silver')], [dict(group=1, query='absent')], EnglishTokenizer(), shared_corpus=True)
    assert len(rows[0]['delivered_ids']) == 2
    corpus[0]['end'] -= 1
    with pytest.raises(ValueError, match='offsets'):
        collect(corpus, [dict(group=0, query='silver')], [dict(group=1, query='absent')], EnglishTokenizer(), shared_corpus=True)
