"""python -m eval.run --data OUTSIDE_REPO/locomo10.json --cache OUTSIDE_REPO/models --out OUTSIDE_REPO/results"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import resource
import sqlite3
import statistics
import time
from adapters.locomo import LoCoMoAdapter
from memlib.index import SentenceEmbedder, BGEEmbedder, EnglishTokenizer, JiebaTokenizer, BigramTokenizer
from memlib.store import Store
from memlib.recall import Recall, Options

def metrics(gold, ranked):
    gold = set(gold)
    return (len(gold.intersection(ranked[:5]))/len(gold),
            len(gold.intersection(ranked[:10]))/len(gold),
            next((1/i for i,key in enumerate(ranked[:10],1) if key in gold),0),
            int(gold.issubset(ranked[:10])))

def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0
    pos = (len(values)-1)*fraction
    lo = int(pos)
    return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(pos-lo)

def evaluate(recall, question, options):
    at = datetime.fromisoformat(question['asked_at'])
    start = time.perf_counter()
    result = recall.query(question['query'],options,at=at)
    elapsed = (time.perf_counter()-start)*1000
    scores = result['debug']['scores']
    keys = sorted(scores,key=lambda k:(-scores[k]['weighted'],int(k)))
    identities = {str(r['rowid']):r['ns']+':'+r['id'] for r in recall.store.db.execute('SELECT rowid,ns,id FROM items')}
    ranked = [] if result['debug']['timed_out'] else [identities[k] for k in keys]
    delivered = [r['ns']+':'+r['id'] for r in result['items']]
    pack_scores = [scores[k]['weighted'] for k in keys if identities[k] in delivered]
    negative = question.get('should_surface') is False
    r5,r10,mrr,all10 = metrics(question['gold'],ranked) if not negative else (0,0,0,0)
    return dict(qid=question['qid'],type=question['type'],negative=negative,
        recall5=r5,recall10=r10,mrr10=mrr,all_gold10=all10,
        pack_hit=int(bool(set(question.get('gold',[])).intersection(delivered))),
        false_surface=int(negative and bool(delivered)), max_pack_score=max(pack_scores,default=0),
        ms=elapsed,timed_out=result['debug']['timed_out'],ranked=json.dumps(ranked[:10]),pack=json.dumps(delivered))

def main():
    parser=argparse.ArgumentParser()
    for name in ('data','cache','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--model', choices=['minilm','bge'], default='minilm')
    parser.add_argument('--tokenizer',choices=['english','jieba','bigram'],default='english')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    for path in (args.data,args.cache,args.out):
        if path.resolve().is_relative_to(repo):
            raise ValueError('data/model/results must be outside repository')
    if (args.out/'summary.json').exists() or (args.out/'per_question.csv').exists():
        raise ValueError('refusing to overwrite existing scored results')
    args.out.mkdir(parents=True,exist_ok=True)
    model=(BGEEmbedder if args.model == 'bge' else SentenceEmbedder)(args.cache)
    loaded_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stamp=time.perf_counter(); model.encode(['今天一起喝茶。']); cold_ms=(time.perf_counter()-stamp)*1000
    times=[]
    for _ in range(30):
        stamp=time.perf_counter(); model.encode(['What did we talk about yesterday?']); times.append((time.perf_counter()-stamp)*1000)
    tokenizer={'english':EnglishTokenizer,'jieba':JiebaTokenizer,'bigram':BigramTokenizer}[args.tokenizer]()
    configs=[('bm25',False,True,False,60),('vector',True,False,False,60),
             ('rrf',True,True,False,60),('rrf+time',True,True,True,60)]
    configs += [(f'rrf+k={k}',True,True,False,k) for k in (0,10,120)]
    if args.model == 'bge':
        configs=[('vector-bge',True,False,False,60),('rrf-bge',True,True,False,60)]
    rows=[]; inventory=[]; query_encoding_ms=[]
    negatives=[json.loads(line) for line in (repo/'eval/negatives/synthetic.jsonl').read_text().splitlines()]
    samples=json.loads(args.data.read_text())
    wall=time.perf_counter()
    for sample_number,sample in enumerate(samples):
        adapter=LoCoMoAdapter(sample)
        questions=list(adapter.questions(sample['qa']))
        cutoff=datetime.fromisoformat(adapter.asked_at)
        store=Store(':memory:',tokenizer,model)
        # Index records only, never QA/gold or generated summaries. Strict cutoff.
        records=[r for r in adapter.records if datetime.fromisoformat(r.time)<cutoff]
        # Batch index encoding only; query encoding remains real and single-text.
        original_encode = model.encode
        if args.model == 'bge':
            vectors = []
            for offset in range(0,len(records),16):
                vectors.extend(original_encode([r.text for r in records[offset:offset+16]]))
            pending = iter(vectors)
            model.encode = lambda texts: [next(pending) for _ in texts]
        if not store.run_batch('public',adapter.ns,records,'start','end'):
            raise RuntimeError('public indexing failed')
        model.encode = original_encode
        query_vectors = {}
        if args.model == 'bge':
            def record_query(texts):
                stamp = time.perf_counter()
                result = original_encode(texts)
                query_encoding_ms.append((time.perf_counter()-stamp)*1000)
                query_vectors.update(zip(texts,result))
                return result
            model.encode = record_query
        recall=Recall(store,{adapter.ns:adapter})
        local_neg=[dict(q,asked_at=adapter.asked_at) for i,q in enumerate(negatives) if i%len(samples)==sample_number]
        inventory.append(dict(ns=adapter.ns,records=len(records),questions=len(questions),excluded=len(sample['qa'])-len(questions),asked_at=adapter.asked_at))
        for name,vector,keyword,weight,k in configs:
            options=Options(vector=vector,keyword=keyword,time_weight=weight,k=k)
            for question in questions+local_neg:
                rows.append(dict(mode=name,**evaluate(recall,question,options)))
            print(adapter.ns,name,len(questions),'wall_seconds',round(time.perf_counter()-wall,2),flush=True)
        model.encode = original_encode
        if args.model == 'bge':
            with sqlite3.connect(args.out/f'index-{sample_number}.sqlite') as saved:
                store.db.backup(saved)
            (args.out/f'queries-{sample_number}.json').write_text(json.dumps(query_vectors))
        store.close()
    with (args.out/'per_question.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries=[]
    for name,*_ in configs:
        subset=[r for r in rows if r['mode']==name]; pos=[r for r in subset if not r['negative']]; neg=[r for r in subset if r['negative']]
        avg=lambda key: statistics.mean(r[key] for r in pos)
        summaries.append(dict(mode=name,n=len(pos),negative_n=len(neg),recall5=avg('recall5'),recall10=avg('recall10'),mrr10=avg('mrr10'),all_gold10=avg('all_gold10'),pack_hit=avg('pack_hit'),false_surface=statistics.mean(r['false_surface'] for r in neg),negative_score={str(p):percentile([r['max_pack_score'] for r in neg],p) for p in (0,.5,.95,1)},p50_ms=percentile([r['ms'] for r in subset],.5),p95_ms=percentile([r['ms'] for r in subset],.95),timeout_rate=statistics.mean(r['timed_out'] for r in subset)))
    output=dict(model=model.version,tokenizer=tokenizer.version,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),inventory=inventory,summaries=summaries,encoding=dict(first_ms=cold_ms,p50_ms=percentile(times,.5),p95_ms=percentile(times,.95),public_query_n=len(query_encoding_ms),public_query_p50_ms=percentile(query_encoding_ms,.5),public_query_p95_ms=percentile(query_encoding_ms,.95)),loaded_peak_rss_kib=loaded_rss,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.perf_counter()-wall,fragments='SKIPPED: no authored impressions or verified evidence mapping in public text-only protocol')
    (args.out/'summary.json').write_text(json.dumps(output,indent=2))
    lines=['| mode | R@5 | R@10 | MRR@10 | neg false | P50 ms | P95 ms | timeout |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['mode']} | {s['recall5']:.4f} | {s['recall10']:.4f} | {s['mrr10']:.4f} | {s['false_surface']:.2%} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['timeout_rate']:.2%} |")
    (args.out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__': main()
