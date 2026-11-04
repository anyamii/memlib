"""Optional BM25 rejection gate; corpus statistics match SQLite's BM25 lane."""
from dataclasses import dataclass
import math


@dataclass
class Gate:
    abs_min: float = 0
    norm_min: float = 0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (self.abs_min, self.norm_min)):
            raise ValueError('gate thresholds must be finite and nonnegative')

    @property
    def enabled(self):
        return self.abs_min > 0 or self.norm_min > 0

    def reason(self, s1, f, has_hit):
        if not has_hit:
            return 'no_hit'
        if s1 < self.abs_min:
            return 'abs'
        if f < self.norm_min:
            return 'norm'
        return None


def features(db, tokens, lane):
    """Use the whole keyword corpus, as SQLite bm25 does before scope filtering.

    fts5vocab supplies actual indexed terms and document frequencies. Query
    tokens are passed through the same unicode61 tokenizer (including accent
    folding) so repeated equivalent terms contribute only once; OOVs count.
    """
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.gate_query USING fts5(tokens)")
    db.execute("DELETE FROM temp.gate_query")
    db.execute("INSERT INTO temp.gate_query(tokens) VALUES(?)", (' '.join(tokens),))
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.gate_query_vocab USING fts5vocab(temp, gate_query, row)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.gate_vocab USING fts5vocab(main, keyword, row)")
    n = db.execute('SELECT count(*) FROM keyword').fetchone()[0]
    denominator = 0.0
    for (term,) in db.execute('SELECT term FROM temp.gate_query_vocab').fetchall():
        row = db.execute('SELECT doc FROM temp.gate_vocab WHERE term=?', (term,)).fetchone()
        df = row[0] if row else 0
        denominator += max(1e-6, math.log((n-df+0.5)/(df+0.5)))
    s1 = lane[0][1] if lane else 0.0
    return s1, s1 / denominator if denominator else 0.0
