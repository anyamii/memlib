import argparse
from collections import defaultdict
import json
from pathlib import Path
from .adapters import FixtureAdapter
from .store import Store
from .recall import Recall

DEFAULT_EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'

def build_demo(store, examples):
    adapter = FixtureAdapter(examples/'messages.jsonl')
    groups = defaultdict(list)
    for r in adapter.iter_since(None):
        groups[r.time[:10]].append(r)
    for date,records in groups.items():
        if not store.run_batch(date,adapter.ns,records,records[0].id,records[-1].id):
            raise RuntimeError(f'demo batch failed: {date}')
    for line in (examples/'impressions.jsonl').read_text().splitlines():
        store.put_impression(**json.loads(line))
    return adapter

def main():
    p = argparse.ArgumentParser(description='Synthetic memory library demo')
    p.add_argument('--db',default='runtime/demo.db')
    p.add_argument('--examples',type=Path,default=DEFAULT_EXAMPLES)
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo')
    sub.add_parser('query').add_argument('text')
    args = p.parse_args()
    Path(args.db).parent.mkdir(parents=True,exist_ok=True)
    store = Store(args.db)
    try:
        adapter = build_demo(store,args.examples) if args.command=='demo' else FixtureAdapter(args.examples/'messages.jsonl')
        queries = json.loads((args.examples/'queries.json').read_text()) if args.command=='demo' else [{'query':args.text}]
        for q in queries:
            print(json.dumps({'query':q['query'],'pack':Recall(store,{'fixture':adapter}).query(q['query'])},ensure_ascii=False,indent=2))
    finally:
        store.close()

if __name__ == '__main__':
    main()
