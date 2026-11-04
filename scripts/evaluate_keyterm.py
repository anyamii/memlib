#!/usr/bin/env python3
"""Generic keyterm experiment from explicit frozen JSON; outputs may expose input text."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memlib.index import BigramTokenizer, JiebaTokenizer
from memlib.keyterm import KeytermGate
from scripts.calibrate_gate import collect, thresholds


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    path.chmod(0o600)


def mine(corpus, output, seeds, top_n=100):
    tokenizer = JiebaTokenizer(hmm=True)
    tokenizer.engine.initialize()
    counts, dfs = Counter(), Counter()
    for row in corpus:
        words = [w for w in tokenizer.tokens(row['text'])
                 if re.fullmatch(r'[\u3400-\u9fff]{2,4}', w)
                 and not tokenizer.engine.FREQ.get(w, 0)]
        counts.update(words)
        dfs.update(set(words))
    candidates = sorted((dict(word=w, frequency=f, df=dfs[w]) for w, f in counts.items()
                         if f >= 3 and dfs[w] >= 2),
                        key=lambda x: (-x['frequency'], -x['df'], x['word']))
    adopted = [r['word'] for r in candidates[:top_n]]
    words = sorted(set(adopted) | set(seeds))
    write(output/'dictionary-candidates.json', candidates)
    (output/'userdict.txt').write_text(''.join(w+' 100000\n' for w in words))
    (output/'userdict.txt').chmod(0o600)
    return dict(method='jieba HMM tokens absent (positive frequency) from default dictionary; CJK length 2-4; frequency>=3, chunk df>=2; sort frequency desc, df desc, word asc',
                requested_n=top_n, candidates=len(candidates), adopted=len(adopted),
                seeds=len(set(seeds)), unique_words=len(words), user_frequency=100000)


def accepted(row, mode, absolute, coverage):
    if not row['pack'] or row['S1'] < absolute or not row['features']['keyword_count']:
        return False
    f = row['features']
    return f[mode.lower()] if mode in ('K1', 'K2') else f['coverage'] >= coverage


def measure(rows, mode, absolute, coverage):
    pos = [r for r in rows if r['positive']]
    rejected = [r for r in pos if not accepted(r, mode, absolute, coverage)]
    result = dict(abs_min=absolute, coverage_min=coverage, positives=len(pos), false_rejects=len(rejected),
                  false_reject_rate=len(rejected)/len(pos), domains={})
    for typ in ('unseen', 'chat'):
        part = [r for r in rows if r.get('negative_type') == typ]
        result[typ] = dict(n=len(part), emits=sum(accepted(r, mode, absolute, coverage) for r in part))
    for domain in sorted({r['domain'] for r in pos}):
        part = [r for r in pos if r['domain'] == domain]
        result['domains'][domain] = dict(n=len(part), rejects=sum(r in rejected for r in part))
    return result


def scan(rows, mode, combined, target):
    # Same absolute-score quantile grid as phase 6, including zero/off.
    aa = thresholds(rows, 'S1') if combined else [0]
    cc = [0]
    if mode == 'K3':
        values = {r['features']['coverage'] for r in rows}
        cc = sorted({0, 1, *values, *(math.nextafter(v, math.inf) for v in values if v < 1)})
    grid = [measure(rows, mode, a, c) for a in aa for c in cc]
    eligible = [m for m in grid if m['false_reject_rate'] <= target]
    order = lambda m: (m['unseen']['emits']+m['chat']['emits'], m['false_reject_rate'], m['abs_min'], m['coverage_min'])
    best = min(eligible, key=order) if eligible else None
    floor = min(grid, key=lambda m: (m['false_reject_rate'], *order(m)))
    return dict(selected=best, minimum_rejection=floor, grid_size=len(grid))


def evaluate(rows):
    results, holdout = [], []
    for target in (.07, .10):
        for mode in ('K1', 'K2', 'K3'):
            for combined in (False, True):
                label = mode + ('+ABS' if combined else '')
                result = scan(rows, mode, combined, target)
                results.append(dict(target=target, kind=label, **result))
                for parity in (0, 1):
                    train = [r for r in rows if r['group'] % 2 == parity]
                    test = [r for r in rows if r['group'] % 2 != parity]
                    tr = scan(train, mode, combined, target)
                    chosen = tr['selected']
                    te = measure(test, mode, chosen['abs_min'], chosen['coverage_min']) if chosen else None
                    holdout.append(dict(target=target, kind=label, train_parity=parity,
                                        train=tr, test=te, meets_10=te['false_reject_rate'] <= .10 if te else None))
    return dict(results=results, holdout=holdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--stopwords', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional calibration JSON to verify')
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    frozen = read(args.input/'frozen-manifest.json')
    def verify():
        for name, digest in frozen.items():
            assert hashlib.sha256((args.input/name).read_bytes()).hexdigest() == digest, name
    verify()
    corpus, positives, negatives = [read(args.input/(n+'.json')) for n in ('corpus', 'positives', 'negatives')]
    if not positives or not negatives:
        raise ValueError('positive and negative queries are required')
    dictionary = mine(corpus, args.output, args.seeds.read_text().splitlines())
    stopwords = args.stopwords.read_text().splitlines()
    tokenizer = JiebaTokenizer(args.output/'userdict.txt', hmm=True)
    gate = KeytermGate([r['text'] for r in corpus], tokenizer, stopwords)
    print('dictionary ready; collecting bigram', flush=True)
    rows = collect(corpus, positives, negatives, BigramTokenizer(), shared_corpus=True)
    baseline_matches = None
    if args.baseline is not None:
        old = read(args.baseline)['rows']
        if len(rows) != len(old):
            raise ValueError('baseline row count mismatch')
        for actual, expected in zip(rows, old):
            for key in expected:
                if actual[key] != expected[key]:
                    raise ValueError(f'baseline mismatch: row {actual["query_index"]}, field {key}')
        baseline_matches = True
    texts = {r['id']: r['text'] for r in corpus}
    for row in rows:
        q = (positives+negatives)[row['query_index']]
        row.update(domain=q.get('domain'), negative_type=q.get('negative_type'))
        row['features'] = gate.features(q['query'], [texts[i] for i in row['delivered_ids']])
    write(args.output/'rows.json', rows)
    print('scanning gates', flush=True)
    output = evaluate(rows)
    # Default-dictionary ablation isolates the impact of mined words/seeds.
    default_gate = KeytermGate([r['text'] for r in corpus], JiebaTokenizer(hmm=True), stopwords)
    default_rows = [dict(r, features=default_gate.features((positives+negatives)[r['query_index']]['query'],
                         [texts[i] for i in r['delivered_ids']])) for r in rows]
    output['default_dictionary_ablation'] = evaluate(default_rows)
    write(args.output/'default-dictionary-rows.json', default_rows)
    print('collecting jieba-only recall control', flush=True)
    jieba_rows = collect(corpus, positives, negatives, tokenizer, shared_corpus=True)
    def recall_at(rr, k):
        pp = [r for r in rr if r['positive']]
        hits = sum(bool(set(positives[r['query_index']]['gold']) & set(r['lane_ids'][:k])) for r in pp)
        return dict(hits=hits, n=len(pp), rate=hits/len(pp))
    output['recall_control'] = {name: {str(k): recall_at(rr, k) for k in (5, 10)}
                                for name, rr in [('bigram', rows), ('jieba_custom', jieba_rows)]}
    write(args.output/'jieba-recall-rows.json', jieba_rows)
    output.update(dictionary=dictionary, stopword_count=len(set(stopwords)),
                  jieba_version=importlib.metadata.version('jieba'), tokenizer=tokenizer.version,
                  frozen_manifest=frozen, baseline_matches=baseline_matches,
                  no_keywords={typ: sum(not r['features']['keyword_count'] for r in rows if
                                        ('positive' if r['positive'] else r['negative_type']) == typ)
                               for typ in ('positive', 'unseen', 'chat')})
    verify()
    write(args.output/'analysis.json', output)
    print('done; frozen hashes verified', flush=True)


if __name__ == '__main__':
    main()
