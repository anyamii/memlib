"""Versioned tokenizers, deterministic test vectors and a pinned CPU semantic model."""
import hashlib
import math
import re
from typing import Protocol

class Tokenizer(Protocol):
    version: str
    def tokens(self, text: str) -> list[str]: ...

class BigramTokenizer:
    version = "cjk-bigram-v1"
    def tokens(self, text):
        result = []
        for run in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
            if re.fullmatch(r"[a-z0-9]+", run):
                result.append(run)
            else:
                result.extend([run[i:i+2] for i in range(len(run)-1)] or [run])
        return result

class Embedder(Protocol):
    version: str
    def encode(self, texts: list[str]) -> list[list[float]]: ...

class FakeEmbedder:
    version = "sha256-character-64-v1"
    def encode(self, texts):
        vectors = []
        for text in texts:
            v = [0.0] * 64
            for c in text:
                h = hashlib.sha256(c.encode()).digest()
                v[int.from_bytes(h[:2], "big") % 64] += 1 if h[2] % 2 else -1
            vectors.append(v)
        return vectors

def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("vector dimensions differ")
    denom = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / denom if denom else 0.0

def split_spans(body, limit=400):
    """Offsets use Python Unicode code points; punctuation stays with its sentence."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    for para in re.finditer(r"\S(?:[^\n]|\n(?!\s*\n))*", body):
        start, end = para.span()
        while end > start and body[end-1].isspace():
            end -= 1
        pos = start
        while pos < end:
            stop = min(pos + limit, end)
            if stop < end:
                boundaries = [m.end() + pos for m in re.finditer(r"[。？！.!?]", body[pos:stop])]
                if boundaries:
                    stop = boundaries[-1]
            yield pos, stop
            pos = stop

def rrf(lanes, k=60):
    if k < 0:
        raise ValueError("RRF k must be nonnegative")
    scores = {}
    for lane in lanes:
        for rank, key in enumerate(dict.fromkeys(lane), 1):
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return scores

class EnglishTokenizer:
    version = "english-lower-alnum-v1"
    def tokens(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

class JiebaTokenizer:
    version = "jieba-0.42.1-default-hmm-false+english-v1"
    def __init__(self):
        import importlib.metadata
        import jieba
        if importlib.metadata.version('jieba') != '0.42.1':
            raise ValueError('jieba version mismatch')
        self.engine = jieba.Tokenizer()
    def tokens(self, text):
        result = []
        for run in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.lower()):
            result.extend(self.engine.cut(run, HMM=False))
        return result

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'

class SentenceEmbedder:
    """CPU multilingual semantic model; files must live outside the repository."""
    version = f'{MODEL_NAME}@{MODEL_REVISION}:normalized:256:v1'
    def __init__(self, cache_dir):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION,
                                         cache_folder=str(cache_dir), device='cpu')
        self.model.max_seq_length = 256
        import torch
        torch.set_num_threads(2)
    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def similarities(self, query, vectors):
        import numpy as np
        matrix = np.asarray(vectors, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        return np.divide(matrix @ q, denom, out=np.zeros(len(matrix)), where=denom != 0).tolist()

BGE_MODEL_NAME = 'BAAI/bge-m3'
BGE_MODEL_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'

class BGEEmbedder(SentenceEmbedder):
    """Optional pinned dense-only BGE; MiniLM and fake defaults remain available."""
    version = f'{BGE_MODEL_NAME}@{BGE_MODEL_REVISION}:dense:normalized:512:v1'
    def __init__(self, cache_dir):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(2)
        path = Path(cache_dir) / ('bge-m3-' + BGE_MODEL_REVISION)
        self.model = SentenceTransformer(str(path), device='cpu', local_files_only=True)
        self.model.max_seq_length = 512
