#!/usr/bin/env python3
"""Offline BM25 calibration from explicitly supplied JSON, never a live store."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memlib import Store, Recall, Options, Gate
from memlib.adapters import Record
from memlib.index import BigramTokenizer, EnglishTokenizer, JiebaTokenizer


class NoEncoder:
    version = 'calibration-no-vectors-v1'

    def encode(self, texts):
        raise AssertionError('calibration must never encode vectors')


class Adapter:
    def __init__(self, records):
        self.records = {r.id: r for r in records}

    def get_by_ids(self, ids):
        return {i: self.records[i] for i in ids if i in self.records}


def collect(corpus, positives, negatives, tokenizer, shared_corpus=False):
    """Groups normally isolate corpora; shared mode splits queries only.

    Optional parent_id/parent_text/seq/start/end fields index real fragments,
    retaining Recall's parent cap and source checks without vector encoding.
    """
    if not corpus or not positives or not negatives:
        raise ValueError('corpus, positives and negatives must be nonempty')
    groups = {r['group'] for r in corpus}
    if any(type(g) is not int or g < 0 for g in groups):
        raise ValueError('group must be a nonnegative integer')
    queries = [(True, q) for q in positives] + [(False, q) for q in negatives]
    if any(type(q['group']) is not int or q['group'] < 0 or (not shared_corpus and q['group'] not in groups) or not isinstance(q['query'], str) or not q['query'].strip() for _, q in queries):
        raise ValueError('each query needs text and an existing integer group')
    result = []
    for group in ([None] if shared_corpus else sorted(groups)):
        selected_corpus = [r for r in corpus if shared_corpus or r['group'] == group]
        records = [Record('calibration', str(r['id']), '2026-01-01', 'calibration', r['text']) for r in selected_corpus]
        if len({r.id for r in records}) != len(records):
            raise ValueError('duplicate corpus id within group')
        store = Store(':memory:', tokenizer=tokenizer, embedder=NoEncoder())
        try:
            # Index only explicit input prose, with empty vector placeholders.
            for r, source in zip(records, selected_corpus):
                parent = source.get('parent_id')
                if parent is not None:
                    body = source['parent_text']
                    start, end = source['start'], source['end']
                    if not 0 <= start < end <= len(body) or body[start:end] != r.text:
                        raise ValueError('invalid fragment offsets')
                    old = store.db.execute('SELECT body FROM impressions WHERE id=?', (parent,)).fetchone()
                    if old and old['body'] != body:
                        raise ValueError('inconsistent fragment parent')
                    store.db.execute("INSERT OR IGNORE INTO impressions(id,body,revision,status) VALUES(?,?,1,'active')", (parent, body))
                    store.db.execute('INSERT INTO fragments VALUES(?,?,1,?,?,?)', (r.id, parent, source['seq'], start, end))
                cur = store.db.execute("INSERT INTO items(ns,kind,id,date,parent_id,parent_revision,vector,tokenizer_version,embedder_version) VALUES(?,?,?,?,?,?,?,?,?)", (r.ns, 'impression_fragment' if parent is not None else 'message', r.id, r.time, parent, 1 if parent is not None else None, '[]', tokenizer.version, store.embedder.version))
                store.db.execute('INSERT INTO keyword(rowid,tokens) VALUES(?,?)', (cur.lastrowid, ' '.join(tokenizer.tokens(r.text))))
            recall = Recall(store, {'calibration': Adapter(records)})
            for query_index, (positive, q) in enumerate(queries):
                if not shared_corpus and q['group'] != group:
                    continue
                # Tiny positive threshold enables diagnostics; any actual hit
                # has a strictly positive BM25 score, so this is the baseline.
                out = recall.query(q['query'], Options(vector=False, time_weight=False, gate=Gate(abs_min=1e-300)))
                d = out['debug']
                if d['timed_out']:
                    raise RuntimeError('calibration query timed out')
                result.append(dict(group=q['group'], query_index=query_index, positive=positive, S1=d['S1'], F=d['F'], has_hit=d['gate_reason'] != 'no_hit', pack=bool(out['items']), delivered_ids=[item['id'] for item in out['items']], lane_ids=[item['id'] for item in d['lanes']['keyword']]))
        finally:
            store.close()
    return result


def metrics(rows, a, b):
    pos = [r for r in rows if r['positive']]
    neg = [r for r in rows if not r['positive']]
    def accepted(r):
        return r['pack'] and Gate(a, b).reason(r['S1'], r['F'], r['has_hit']) is None
    rejected = sum(not accepted(r) for r in pos)
    emitted = sum(accepted(r) for r in neg)
    return dict(abs_min=a, norm_min=b, positives=len(pos), negatives=len(neg), false_rejects=rejected, false_emits=emitted, false_reject_rate=rejected/len(pos) if pos else None, false_emit_rate=emitted/len(neg) if neg else None)


def thresholds(rows, feature):
    values = sorted(r[feature] for r in rows if r['positive'])
    if not values:
        return [0]
    return sorted({0, *(values[min(len(values)-1, int(len(values)*p))] for p in (0, .01, .025, .05, .07, .075, .1, .15, .2, .3, .5, .7, .9, 1))})


def scan(rows, target):
    aa, bb = thresholds(rows, 'S1'), thresholds(rows, 'F')
    grid = [metrics(rows, a, b) for a in aa for b in bb]
    candidates = [r for r in grid if r['false_reject_rate'] is not None and r['false_emit_rate'] is not None and r['false_reject_rate'] <= target]
    candidates.sort(key=lambda r: (r['false_emit_rate'], r['false_reject_rate'], r['abs_min'], r['norm_min']))
    return dict(abs_sweep=[metrics(rows, a, 0) for a in aa], norm_sweep=[metrics(rows, 0, b) for b in bb], combined_grid=grid, candidates=candidates, selected=candidates[0] if candidates else None)


def calibrate(rows, target=.07):
    if not 0 <= target <= 1:
        raise ValueError('target must be between 0 and 1')
    result = dict(target=target, full=scan(rows, target), holdout=[])
    for parity in (0, 1):
        train = [r for r in rows if r['group'] % 2 == parity]
        test = [r for r in rows if r['group'] % 2 != parity]
        if any(not any(r['positive'] == label for r in part) for part in (train, test) for label in (True, False)):
            raise ValueError('each parity needs positive and negative queries in separate corpus groups')
        training = scan(train, target)
        selected = training['selected']
        held = metrics(test, selected['abs_min'], selected['norm_min']) if selected else None
        result['holdout'].append(dict(train_parity=parity, training=training, test=held, meets_target=held['false_reject_rate'] <= target if held else False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--positives', type=Path, required=True)
    p.add_argument('--negatives', type=Path, required=True)
    p.add_argument('--tokenizer', choices=['bigram', 'english', 'jieba'], default='bigram')
    p.add_argument('--target', type=float, default=.07)
    p.add_argument('--shared-corpus', action='store_true', help='Retrieve against one full index; group labels split queries only for holdout')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tokenizer = dict(bigram=BigramTokenizer, english=EnglishTokenizer, jieba=JiebaTokenizer)[args.tokenizer]()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    rows = collect(read(args.corpus), read(args.positives), read(args.negatives), tokenizer, args.shared_corpus)
    output = calibrate(rows, args.target)
    output.update(tokenizer=tokenizer.version, shared_corpus=args.shared_corpus, options=asdict(Options(vector=False, time_weight=False)), rows=rows)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
