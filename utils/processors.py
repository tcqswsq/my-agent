import re, hashlib
import numpy as np
from langchain_core.documents import Document

def enterprise(docs, min_len=50):
    REGEX_RULES = [re.compile(r'^\s*\d+\s*$'), re.compile(r'^(第?\s*\d+\s*页?).*$', re.I), re.compile(r'保密|confidential|copyright|©|\u00a9', re.I), re.compile(r'^[\s\-_=.]{1,5}$'), re.compile(r'^[^\w\u4e00-\u9fff\s]{3,}$')]
    class Deduplicator:
        def __init__(self): self.seen = set()
        def add(self, text):
            h = hashlib.md5(re.sub(r'\s+', '', text).lower().encode()).hexdigest()
            if h in self.seen: return False
            self.seen.add(h); return True
    dedup = Deduplicator(); out = []
    for doc in docs:
        c = doc.page_content.strip(); m = doc.metadata
        if not c or len(c) < min_len: continue
        if any(p.search(c) for p in REGEX_RULES): continue
        if len(re.findall(r'[\w\u4e00-\u9fff]', c)) / len(c) < 0.4: continue
        if m.get("category") == "Table":
            m.setdefault('text_as_html', c); out.append(Document(page_content=c, metadata=m)); continue
        if not dedup.add(c): continue
        cat = m.get("category")
        m["level"] = "h1" if cat == "Title" and len(c) < 30 else "h2" if cat == "Title" else "list" if cat == "ListItem" else "p"
        out.append(Document(page_content=c, metadata=m))
    return out

def merge(docs, max_len=800):
    merged = []; buf = []; cur = 0; base = {}
    for doc in docs:
        c = doc.page_content; m = doc.metadata
        if not base: base = m.copy(); base["start_page"] = m.get("page_number", 1)
        new_sec = m.get("level") in ["h1", "h2"]
        if new_sec or cur > max_len:
            if buf: merged.append(Document(page_content="\n".join(buf), metadata=base))
            buf = [c]; cur = len(c); base = m.copy(); base["start_page"] = m.get("page_number", 1)
        else:
            buf.append(c); cur += len(c)
            if m.get("page_number"): base["end_page"] = m.get("page_number")
    if buf: merged.append(Document(page_content="\n".join(buf), metadata=base))
    return merged

def fenkuai(docs, embed_model, sim_threshold=0.75):
    if len(docs) <= 1: return docs
    merged = []; buf = [docs[0]]; base = docs[0].metadata.copy()
    embs = embed_model.encode([d.page_content for d in docs])
    # 兼容 tensor 和 numpy
    is_tensor = hasattr(embs, 'cpu')
    for i in range(1, len(docs)):
        d = docs[i]
        if is_tensor:
            from sentence_transformers import util as st_util
            sim = st_util.cos_sim(embs[i], embs[i - 1]).item()
        else:
            # numpy: 手动算 cosine similarity
            a, b = embs[i], embs[i - 1]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        total = len("\n".join(x.page_content for x in buf)) + len(d.page_content)
        if sim >= sim_threshold and total <= 800 and d.metadata.get("category") != "Table":
            buf.append(d)
            if d.metadata.get("page_number"): base["end_page"] = d.metadata["page_number"]
        else:
            merged.append(Document(page_content="\n".join(x.page_content for x in buf), metadata=base))
            buf = [d]; base = d.metadata.copy(); base["start_page"] = d.metadata.get("page_number", 1)
    if buf: merged.append(Document(page_content="\n".join(x.page_content for x in buf), metadata=base))
    return merged