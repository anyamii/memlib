"""Resumable public-only A then B; smoke outputs never share production paths."""
import argparse
import hashlib
import json
import resource
import statistics
import time
from datetime import datetime
from pathlib import Path
from adapters.locomo import LoCoMoAdapter
from memlib.index import BGEEmbedder, EnglishTokenizer
from memlib.store import Store
from memlib.recall import Recall, Options
from eval.run import evaluate, percentile

DATA_SHA = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def progress(stage, n, total, detail=''):
    print(datetime.now().astimezone().isoformat(), stage, f'{n}/{total}', detail, flush=True)

def summarize(rows):
    result = []
    for mode in dict.fromkeys(r['mode'] for r in rows):
        subset = [r for r in rows if r['mode'] == mode]
        pos = [r for r in subset if not r['negative']]
        neg = [r for r in subset if r['negative']]
        result.append(dict(mode=mode, n=len(pos), negative_n=len(neg),
            **{k: statistics.mean(r[k] for r in pos) for k in ('recall5','recall10','mrr10')},
            false_surface=statistics.mean(r['false_surface'] for r in neg) if neg else None,
            positive_pack_empty=statistics.mean(not json.loads(r['pack']) for r in pos),
            p50_ms=percentile([r['ms'] for r in subset],.5),
            p95_ms=percentile([r['ms'] for r in subset],.95),
            timeout_rate=statistics.mean(r['timed_out'] for r in subset)))
    return result

def main():
    p=argparse.ArgumentParser()
    for key in ('data','cache','out'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); repo=Path(__file__).resolve().parents[1]
    for path in (a.data,a.cache,a.out):
        if path.resolve().is_relative_to(repo): raise ValueError('artifacts must be outside git')
    if a.smoke and a.out.resolve() == (repo.parent/'memlib-artifacts/results-phase3').resolve():
        raise ValueError('smoke requires separate --out')
    a.out.mkdir(parents=True,exist_ok=True)
    data=a.data.read_bytes(); assert hashlib.sha256(data).hexdigest()==DATA_SHA
    protocol=dict(data_sha256=DATA_SHA, model=BGEEmbedder.version,n=50,k=60,max_items=5,max_chars=1200,time_weight=False,smoke=a.smoke,runner=1)
    manifest=a.out/'protocol.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==protocol,'protocol mismatch'
    else: save(manifest,protocol)
    if (a.out/'DONE').exists():
        progress('DONE resume',1,1); return
    start=time.perf_counter(); model=BGEEmbedder(a.cache); encode=model.encode
    loaded=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    samples=json.loads(data); negatives=[json.loads(l) for l in (repo/'eval/negatives/synthetic.jsonl').read_text().splitlines()]
    contexts=[]; encoding=[]; index_seconds=0; index_count=0
    for i,sample in enumerate(samples[:1] if a.smoke else samples):
        adapter=LoCoMoAdapter(sample); questions=list(adapter.questions(sample['qa']))
        records=[r for r in adapter.records if datetime.fromisoformat(r.time)<datetime.fromisoformat(adapter.asked_at)]
        if a.smoke:
            questions=questions[:20]
            gold={g for q in questions for g in q['gold']}
            records=[r for r in records if r.ns+':'+r.id in gold or r in records[:20]]
        qs=questions+[dict(q,asked_at=adapter.asked_at) for j,q in enumerate(negatives) if j%len(samples)==i]
        directory=a.out/f'cache-{i}'; directory.mkdir(exist_ok=True)
        index=directory/'index.sqlite'; legacy=a.out/'stage-A'/f'index-{i}.sqlite'
        if not index.exists() and legacy.exists() and not a.smoke:
            import shutil
            temporary=index.with_suffix('.importing')
            shutil.copy2(legacy,temporary)
            temporary.replace(index)
        store=Store(index,EnglishTokenizer(),model)
        existing=list(store.db.execute('SELECT ns,id,embedder_version,tokenizer_version FROM items'))
        if existing:
            assert {(r['ns'],r['id']) for r in existing}=={(r.ns,r.id) for r in records}
            assert all(r['embedder_version']==model.version and r['tokenizer_version']==store.tokenizer.version for r in existing)
            progress('index resume',len(records),len(records),adapter.ns)
        else:
            vectors=[]; stamp=time.perf_counter()
            for offset in range(0,len(records),16):
                chunk=directory/f'vectors-{offset}.json'
                if not chunk.exists(): save(chunk,encode([r.text for r in records[offset:offset+16]]))
                vectors.extend(json.loads(chunk.read_text()))
                progress('index encode',min(offset+16,len(records)),len(records),adapter.ns)
            pending=iter(vectors); model.encode=lambda texts:[next(pending) for _ in texts]
            try:
                assert store.run_batch('public',adapter.ns,records,'start','end'),'index failed'
            finally: model.encode=encode
            index_seconds+=time.perf_counter()-stamp; index_count+=len(records)
        store.close(); contexts.append((i,adapter,qs,directory,index))
    progress('indexes complete',len(contexts),len(contexts))
    for stage,modes in [('A',[('vector-bge',False,None),('rrf-bge',True,None)]),('B',[(f'rrf-bge-t={t:.2f}',True,t) for t in (.30,.35,.40,.45,.50,.55,.60,.65,.70)])]:
        all_rows=[]
        for i,adapter,qs,directory,index in contexts:
            store=Store(index,EnglishTokenizer(),model); recall=Recall(store,{adapter.ns:adapter})
            queries=directory/'queries.json'; cache=json.loads(queries.read_text()) if queries.exists() else {}
            legacy=a.out/'stage-A'/f'queries-{i}.json'
            if not cache and legacy.exists() and not a.smoke: cache=json.loads(legacy.read_text())
            for mode,keyword,threshold in modes:
                last_encoding=[]
                rows_file=directory/(mode+'.json'); rows=json.loads(rows_file.read_text()) if rows_file.exists() else []
                assert [r['qid'] for r in rows]==[q['qid'] for q in qs[:len(rows)]], 'checkpoint questions differ'
                def query_encode(texts):
                    if stage=='B': return [cache[t] for t in texts]
                    stamp=time.perf_counter(); values=encode(texts)
                    elapsed=(time.perf_counter()-stamp)*1000
                    encoding.append(elapsed); last_encoding.append(elapsed)
                    cache.update(zip(texts,values)); return values
                model.encode=query_encode
                for q in qs[len(rows):]:
                    last_encoding.clear()
                    row=dict(mode=mode,**evaluate(recall,q,Options(keyword=keyword,time_weight=False,vector_min_score=threshold)))
                    if stage=='A': row['encoding_ms']=last_encoding[-1] if last_encoding else None
                    save(queries,cache)
                    rows.append(row); save(rows_file,rows)
                    progress(stage+' '+mode,len(rows),len(qs),adapter.ns)
                save(queries,cache); all_rows.extend(rows)
                progress(stage+' '+mode+' complete',len(rows),len(qs),adapter.ns)
            store.close()
        # Separate A and B artifacts; B uses cached vectors, its latency is replay latency.
        save(a.out/f'stage-{stage}-rows.json',all_rows)
        measured=[r['encoding_ms'] for r in all_rows if r.get('encoding_ms') is not None]
        save(a.out/f'stage-{stage}-summary.json',dict(protocol=protocol,encoding_n=len(measured),encoding_p50_ms=percentile(measured,.5),encoding_p95_ms=percentile(measured,.95),summaries=summarize(all_rows),latency='real single query encoding' if stage=='A' else 'cached vector replay; not online latency'))
        progress('stage '+stage+' complete',len(all_rows),len(all_rows))
    model.encode=encode
    save(a.out/'runtime.json',dict(smoke=a.smoke,wall_seconds=time.perf_counter()-start,index_seconds=index_seconds,index_records=index_count,loaded_peak_rss_kib=loaded,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,query_encoding_n=len(encoding),query_p50_ms=percentile(encoding,.5),query_p95_ms=percentile(encoding,.95)))
    (a.out/'DONE').write_text(datetime.now().astimezone().isoformat()+'\n')
    progress('DONE',1,1)

if __name__=='__main__': main()
