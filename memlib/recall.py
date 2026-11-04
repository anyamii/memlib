"""Ordered, switchable recall stages with independent keyword and vector lanes."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
import re
from difflib import SequenceMatcher
from .deadline import Deadline
from .index import cosine, rrf

@dataclass
class Options:
    near_duplicate_threshold: float = 0.97
    vector_min_score: float | None = None
    n: int = 50
    k: float = 60
    half_life_days: float = 30
    time_strength: float = 0.1
    max_items: int = 5
    max_chars: int = 1200
    timeout_seconds: float = 2
    scope: bool = True
    pre_tombstone: bool = True
    keyword: bool = True
    vector: bool = True
    dedup: bool = True
    fusion: bool = True
    time_weight: bool = True
    fetch: bool = True
    post_tombstone: bool = True
    parent_cap: bool = True
    budget: bool = True

    def __post_init__(self):
        if self.vector_min_score is not None and not -1 <= self.vector_min_score <= 1:
            raise ValueError('invalid vector minimum score')
        if not 0 <= self.near_duplicate_threshold <= 1:
            raise ValueError('invalid near duplicate threshold')
        if self.n < 0 or self.max_items < 0 or self.max_chars < 0 or self.k < 0 or self.half_life_days <= 0:
            raise ValueError("invalid recall limits")

class Recall:
    def __init__(self, store, adapters):
        self.store = store
        self.adapters = adapters
        self.encoder_deadline = Deadline()
        self.source_deadline = Deadline()

    @staticmethod
    def _near_duplicate(a, b, threshold):
        matcher = SequenceMatcher(None,a,b,autojunk=False)
        return (matcher.real_quick_ratio() >= threshold and
                matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold)

    def query(self, text, options=None, namespaces=None, at=None, after_fetch=None):
        o = options or Options()
        start = time.monotonic()
        debug = {'counts':{},'lanes':{},'scores':{},'timed_out':False,'source_errors':0}
        def checkpoint(stage, count):
            debug['counts'][stage] = count
            if time.monotonic() - start >= o.timeout_seconds:
                raise TimeoutError
        try:
            s = self.store
            items = {r['rowid']:dict(r) for r in s.db.execute("SELECT * FROM items")}
            items = {key:r for key,r in items.items() if (not o.scope or namespaces is None or r['ns'] in namespaces) and (not o.pre_tombstone or s.allowed(r))}
            if at is not None:
                cutoff = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
                items = {k:r for k,r in items.items() if (lambda d: d if d.tzinfo else d.replace(tzinfo=timezone.utc))(datetime.fromisoformat(r['date'])) < cutoff}
            checkpoint('1_eligible',len(items))
            for r in items.values():
                if o.keyword and r['tokenizer_version'] != s.tokenizer.version:
                    raise ValueError('index version mismatch: rebuild required')
            # Scope and tombstones are applied before lane top-N, so blocked
            # results cannot use up the candidate quota.
            lanes = []
            if o.keyword:
                tokens = list(dict.fromkeys(s.tokenizer.tokens(text)))
                expression = ' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
                hits = s.db.execute("SELECT rowid,bm25(keyword) AS score FROM keyword WHERE keyword MATCH ? ORDER BY score,rowid", (expression,)).fetchall() if expression else []
                lane = [(r['rowid'],-r['score']) for r in hits if r['rowid'] in items][:o.n]
                lanes.append([key for key,_ in lane])
                debug['lanes']['keyword'] = [{'key':key,'id':items[key]['id'],'rank':i,'score':score} for i,(key,score) in enumerate(lane,1)]
            if o.vector:
                q = self.encoder_deadline.call(lambda: s.embedder.encode([text]), start + o.timeout_seconds)[0]
                lane = []
                for key,r in items.items():
                    if r['embedder_version'] != s.embedder.version or r['tokenizer_version'] != s.tokenizer.version:
                        raise ValueError("index version mismatch: rebuild required")
                    lane.append((key,json.loads(r['vector'])))
                if lane and hasattr(s.embedder, 'similarities'):
                    values = s.embedder.similarities(q,[v for _,v in lane])
                    lane = [(pair[0],score) for pair,score in zip(lane,values)]
                else:
                    lane = [(key,cosine(q,v)) for key,v in lane]
                lane = sorted(lane,key=lambda x:(-x[1],x[0]))[:o.n]
                if o.vector_min_score is not None:
                    lane = [(key, score) for key, score in lane if score >= o.vector_min_score]
                lanes.append([key for key,_ in lane])
                debug['lanes']['vector'] = [{'key':key,'id':items[key]['id'],'rank':i,'score':score} for i,(key,score) in enumerate(lane,1)]
            checkpoint('2_lanes',sum(map(len,lanes)))
            union = [key for lane in lanes for key in lane]
            if o.dedup:
                union = list(dict.fromkeys(union))
            checkpoint('3_dedup',len(union))
            scores = rrf(lanes,o.k) if o.fusion else {key:1/(i+1) for i,key in reversed(list(enumerate(union)))}
            checkpoint('4_rrf',len(scores))
            current = at or datetime.now(timezone.utc)
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            weighted = dict(scores)
            if o.time_weight:
                for key in weighted:
                    date = datetime.fromisoformat(items[key]['date'])
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    age = max(0,(current-date).total_seconds()/86400)
                    weighted[key] *= 1 + o.time_strength * 2**(-age/o.half_life_days)
            ranked = sorted(union,key=lambda key:(-weighted[key],key))
            debug['scores'] = {str(key):{'id':items[key]['id'],'rrf':scores[key],'weighted':weighted[key]} for key in dict.fromkeys(ranked)}
            checkpoint('5_time',len(ranked))
            fetched = []
            # Batch source reads by namespace; no persistent text cache.
            sources = {}
            if o.fetch:
                for ns in {items[key]['ns'] for key in ranked if items[key]['kind']=='message'}:
                    try:
                        ids = [items[key]['id'] for key in ranked if items[key]['ns']==ns]
                        sources[ns] = self.source_deadline.call(lambda: self.adapters[ns].get_by_ids(ids), start + o.timeout_seconds)
                    except TimeoutError:
                        raise
                    except Exception:
                        debug['source_errors'] += 1
                        sources[ns] = {}
                    checkpoint('6_source_progress',len(sources))
                for key in ranked:
                    r = items[key]
                    if r['kind']=='message':
                        record = sources.get(r['ns'],{}).get(r['id'])
                        body = record.text if record and record.ns==r['ns'] and record.id==r['id'] else None
                    else:
                        body = s.fragment_text(r)
                    if body is not None:
                        fetched.append((r,body))
            checkpoint('6_fetch',len(fetched))
            if after_fetch:
                after_fetch()
            fetched = [(r,body) for r,body in fetched if not o.post_tombstone or s.allowed(r)]
            checkpoint('6_post_tombstone',len(fetched))
            capped, parents = [],{}
            for r,body in fetched:
                if r['kind']=='impression_fragment':
                    parent = (r['ns'],r['parent_id'])
                    parents[parent] = parents.get(parent,0)+1
                    if o.parent_cap and parents[parent]>2:
                        continue
                capped.append((r,body))
            checkpoint('7_parent_cap',len(capped))
            unique = []
            normalized = []
            for r,body in capped:
                if o.budget and len(unique) >= o.max_items:
                    break
                norm = re.sub(r"\W+", "", body.casefold())
                if norm and any(self._near_duplicate(norm,old,o.near_duplicate_threshold) for old in normalized):
                    continue
                unique.append((r,body))
                normalized.append(norm)
            capped = unique
            pack,chars = [],0
            for r,body in capped:
                if o.budget:
                    if len(pack)>=o.max_items or chars>=o.max_chars:
                        break
                    body = body[:o.max_chars-chars]
                entry = {'kind':r['kind'],'date':r['date'],'id':r['id'],'ns':r['ns'],'text':body}
                if r['kind']=='impression_fragment':
                    entry['source_note'] = 'AI 当时写的叙述'
                    entry['parent_id'] = r['parent_id']
                pack.append(entry)
                chars += len(body)
            checkpoint('8_budget',len(pack))
            debug['retrieved'] = len(ranked)
            debug['delivered'] = len(pack)
            return {'items':pack,'debug':debug}
        except TimeoutError:
            debug.update(timed_out=True,delivered=0)
            return {'items':[],'debug':debug}
