"""Corpus-only substring IDF and optional post-Pack jieba keyword gates.

No private text or query vocabulary is stored in source control. A caller owns
its corpus, tokenizer/user dictionary and stopword set. IDF includes OOV terms.
"""
import math
import re


class KeytermGate:
    def __init__(self, corpus, tokenizer, stopwords=(), mode='K1', coverage_min=0):
        if mode not in ('K1', 'K2', 'K3'):
            raise ValueError('unknown keyterm gate')
        if not math.isfinite(coverage_min) or not 0 <= coverage_min <= 1:
            raise ValueError('coverage_min must be finite and in [0,1]')
        self.corpus = tuple(text.casefold() for text in corpus)
        self.tokenizer = tokenizer
        self.stopwords = frozenset(w.casefold() for w in stopwords)
        self.mode, self.coverage_min = mode, coverage_min
        self._df = {}

    def keywords(self, query):
        # Single CJK characters are too ambiguous here; retain multi-character
        # CJK words and English/number tokens of length >= 2. Stable query order
        # resolves equal-IDF ties; repeated tokens count once.
        words = dict.fromkeys(w.casefold() for w in self.tokenizer.tokens(query))
        return [w for w in words if len(w) >= 2 and
                re.fullmatch(r'[\u3400-\u9fffa-z0-9]+', w) and w not in self.stopwords]

    def idf(self, word):
        if word not in self._df:
            self._df[word] = sum(word in text for text in self.corpus)
        return math.log((len(self.corpus) + 1) / (self._df[word] + 1)) + 1

    def features(self, query, pack_originals):
        words = self.keywords(query)
        weights = {w: self.idf(w) for w in words}
        ranked = sorted(words, key=lambda w: -weights[w])
        texts = [text.casefold() for text in pack_originals]
        covered = {w for w in words if any(w in text for text in texts)}
        mass = sum(weights.values())
        return dict(keyword_count=len(words),
                    k1=bool(ranked and ranked[0] in covered),
                    k2=any(w in covered for w in ranked[:2]),
                    coverage=sum(weights[w] for w in covered)/mass if mass else 0)

    def reason(self, features):
        if not features['keyword_count']:
            return 'no_keywords'
        passes = features[self.mode.lower()] if self.mode != 'K3' else features['coverage'] >= self.coverage_min
        return None if passes else 'keyterm'
