"""Single SQLite store: only impressions have stored prose."""
import json
import sqlite3
import time
from .deadline import Deadline
from datetime import datetime, timezone
from .index import BigramTokenizer, FakeEmbedder, split_spans

def now():
    return datetime.now(timezone.utc).isoformat()

class BatchInputError(ValueError):
    """Safe error code, never contains source text."""

class Store:
    def __init__(self, path, tokenizer=None, embedder=None, operation_timeout=30):
        if sqlite3.sqlite_version_info < (3, 43, 0):
            raise RuntimeError("SQLite >=3.43 required for contentless_delete")
        self.operation_timeout = operation_timeout
        self.encoder_deadline = Deadline()
        self.reader_deadline = Deadline()
        self.tokenizer = tokenizer or BigramTokenizer()
        self.embedder = embedder or FakeEmbedder()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS impressions(
          id TEXT PRIMARY KEY,date TEXT,written_at TEXT,title TEXT,body TEXT,cues TEXT,
          revision INTEGER,status TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS impression_history(
          id TEXT,revision INTEGER,date TEXT,written_at TEXT,title TEXT,body TEXT,cues TEXT,
          status TEXT,created_at TEXT,updated_at TEXT,PRIMARY KEY(id,revision));
        CREATE TABLE IF NOT EXISTS fragments(
          id TEXT PRIMARY KEY,parent_id TEXT,parent_revision INTEGER,seq INTEGER,start INTEGER,end INTEGER);
        CREATE TABLE IF NOT EXISTS items(
          rowid INTEGER PRIMARY KEY,ns TEXT,kind TEXT,id TEXT,date TEXT,parent_id TEXT,
          parent_revision INTEGER,vector TEXT,tokenizer_version TEXT,embedder_version TEXT,
          indexed_at TEXT,UNIQUE(ns,kind,id));
        CREATE VIRTUAL TABLE IF NOT EXISTS keyword USING fts5(tokens,content='',contentless_delete=1);
        CREATE TABLE IF NOT EXISTS batches(
          batch_id TEXT,ns TEXT,range_start TEXT,range_end TEXT,status TEXT
          CHECK(status IN ('pending','success','failed')),error TEXT,tokenizer_version TEXT,
          embedder_version TEXT,created_at TEXT,finished_at TEXT,PRIMARY KEY(ns,batch_id));
        CREATE TABLE IF NOT EXISTS tombstones(
          target_kind TEXT CHECK(target_kind IN ('message','impression','fragment')),
          target_id TEXT,ns TEXT,reason TEXT,created_at TEXT,PRIMARY KEY(ns,target_kind,target_id));
        ''')

    def close(self):
        self.db.close()

    def _remove(self, rowid):
        self.db.execute("DELETE FROM keyword WHERE rowid=?", (rowid,))
        self.db.execute("DELETE FROM items WHERE rowid=?", (rowid,))

    def index(self, ns, kind, id, date, text, parent_id=None, parent_revision=None):
        old = self.db.execute("SELECT rowid FROM items WHERE ns=? AND kind=? AND id=?", (ns,kind,id)).fetchone()
        if old:
            self._remove(old[0])
        v = self.encoder_deadline.call(lambda: self.embedder.encode([text]), time.monotonic()+self.operation_timeout)[0]
        cur = self.db.execute("INSERT INTO items(ns,kind,id,date,parent_id,parent_revision,vector,tokenizer_version,embedder_version,indexed_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ns,kind,id,date,parent_id,parent_revision,json.dumps(v),self.tokenizer.version,self.embedder.version,now()))
        self.db.execute("INSERT INTO keyword(rowid,tokens) VALUES(?,?)", (cur.lastrowid," ".join(self.tokenizer.tokens(text))))

    def run_batch(self, batch_id, ns, records, range_start, range_end, fail=False):
        """Explicit ranges, independent status and atomic index publication per batch."""
        with self.db:
            self.db.execute("INSERT INTO batches VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ns,batch_id) DO UPDATE SET status='pending',error=NULL,finished_at=NULL,range_start=excluded.range_start,range_end=excluded.range_end,tokenizer_version=excluded.tokenizer_version,embedder_version=excluded.embedder_version",
                (batch_id,ns,range_start,range_end,'pending',None,self.tokenizer.version,self.embedder.version,now(),None))
        try:
            records = self.reader_deadline.call(lambda: list(records), time.monotonic()+self.operation_timeout)
            with self.db:
                for r in records:
                    if r.ns != ns:
                        raise BatchInputError("namespace_mismatch")
                    self.index(ns,'message',r.id,r.time,r.text)
                    if fail:
                        raise BatchInputError("synthetic_failure_after_first_record")
                self.db.execute("UPDATE batches SET status='success',finished_at=? WHERE ns=? AND batch_id=?", (now(),ns,batch_id))
        except Exception as exc:
            with self.db:
                # No exception text: third-party errors can contain source prose.
                self.db.execute("UPDATE batches SET status='failed',error=?,finished_at=? WHERE ns=? AND batch_id=?", (str(exc) if isinstance(exc, BatchInputError) else type(exc).__name__,now(),ns,batch_id))
            return False
        return True

    def put_impression(self, id, date, title, body, cues=(), written_at=None):
        with self.db:
            old = self.db.execute("SELECT * FROM impressions WHERE id=?", (id,)).fetchone()
            revision = old['revision'] + (old['body'] != body) if old else 1
            if old and old['body'] != body:
                self.db.execute("INSERT INTO impression_history VALUES(?,?,?,?,?,?,?,?,?,?)", tuple(old[k] for k in ['id','revision','date','written_at','title','body','cues','status','created_at','updated_at']))
            for row in self.db.execute("SELECT rowid FROM items WHERE ns='impressions' AND parent_id=?", (id,)).fetchall():
                self._remove(row[0])
            self.db.execute("DELETE FROM fragments WHERE parent_id=?", (id,))
            self.db.execute("INSERT OR REPLACE INTO impressions VALUES(?,?,?,?,?,?,?,?,?,?)", (id,date,written_at or (old['written_at'] if old else now()),title,body,json.dumps(list(cues),ensure_ascii=False),revision,'active',old['created_at'] if old else now(),now()))
            for seq,(start,end) in enumerate(split_spans(body)):
                fid = f'{id}:r{revision}:{seq}'
                self.db.execute("INSERT INTO fragments VALUES(?,?,?,?,?,?)", (fid,id,revision,seq,start,end))
                self.index('impressions','impression_fragment',fid,date,body[start:end],id,revision)
        return revision

    def tombstone(self, kind, id, ns, reason="user request"):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO tombstones VALUES(?,?,?,?,?)", (kind,id,ns,reason,now()))

    def delete_impression(self, id):
        self.tombstone('impression',id,'impressions')
        with self.db:
            self.db.execute("UPDATE impressions SET status='deleted',updated_at=? WHERE id=?", (now(),id))
            for r in self.db.execute("SELECT rowid FROM items WHERE ns='impressions' AND parent_id=?", (id,)).fetchall():
                self._remove(r[0])
            self.db.execute("DELETE FROM fragments WHERE parent_id=?", (id,))

    def allowed(self, item):
        targets = [('message',item['id'])] if item['kind']=='message' else [('fragment',item['id']),('impression',item['parent_id'])]
        if any(self.db.execute("SELECT 1 FROM tombstones WHERE ns=? AND target_kind=? AND target_id=?", (item['ns'],kind,id)).fetchone() for kind,id in targets):
            return False
        if item['kind'] != 'message':
            p = self.db.execute("SELECT revision,status FROM impressions WHERE id=?", (item['parent_id'],)).fetchone()
            return bool(p and p['revision']==item['parent_revision'] and p['status']=='active')
        return True

    def fragment_text(self, item):
        r = self.db.execute("SELECT f.start,f.end,p.body FROM fragments f JOIN impressions p ON p.id=f.parent_id AND p.revision=f.parent_revision WHERE f.id=? AND p.status='active'", (item['id'],)).fetchone()
        return r['body'][r['start']:r['end']] if r else None
