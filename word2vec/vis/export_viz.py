"""
export the trained cbow model for the interactive page (w2v_viz.html)
run after word2vec.py:  python word2vec/vis/export_viz.py  -> writes word2vec/vis/w2v_viz_data.js
"""

import base64
import inspect
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from word2vec import (  # noqa: E402
    ARTIFACT_PREFIX, BATCH_SIZE, CONTEXT_SIZE, EMBED_DIM, EPOCHS, LR, MAX_REVIEWS, MIN_COUNT, WINDOW_SIZE,
    CBOW, Vocab, build_cbow_windows, load_corpus, most_similar, tokenize, train,
)

N_MAP = 2000      # most frequent words placed on the 2D map
N_CLUSTERS = 14
N_PRESETS = 14
OUT_PATH = Path(__file__).resolve().parent / "w2v_viz_data.js"


def quantize(matrix):
    """per-row symmetric int8 so the browser can run the real weights at ~1/4 the size"""
    scale = np.abs(matrix).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(matrix / scale[:, None]).astype(np.int8)
    return {"b64": base64.b64encode(q.tobytes()).decode("ascii"), "scale": np.round(scale, 7).tolist()}


def pick_sample_review(texts, vocab):
    """a short review that also contains a few out-of-vocab words to show dropping"""
    for text in texts:
        toks = tokenize(text)
        oov = sum(t not in vocab for t in toks)
        if 45 <= len(toks) <= 80 and 1 <= oov <= 4:
            return text
    return texts[0]


def pick_presets(model, encoded, words, rng):
    """windows from the corpus where the model's top guess is the true (non-trivial) target"""
    contexts, targets = build_cbow_windows(encoded[:800])
    with torch.no_grad():
        probs = torch.softmax(model(contexts), dim=1)
    p_top, top1 = probs.max(dim=1)
    good = ((top1 == targets) & (targets >= 60) & (p_top > 0.35)).nonzero().flatten().numpy()
    rng.shuffle(good)

    presets, seen = [], set()
    for i in good:
        t = int(targets[i])
        if t in seen:
            continue
        seen.add(t)
        presets.append({
            "context": [words[j] for j in contexts[i].tolist()],
            "target": words[t],
            "p": round(float(p_top[i]), 3),
        })
        if len(presets) == N_PRESETS:
            break
    return presets


def main():
    words = json.load(open(f"{ARTIFACT_PREFIX}_vocab.json"))
    history = json.load(open(f"{ARTIFACT_PREFIX}_history.json"))
    state = torch.load(f"{ARTIFACT_PREFIX}_cbow.pt", map_location="cpu", weights_only=True)

    texts = load_corpus()
    tokenized = [tokenize(t) for t in texts]
    vocab = Vocab(tokenized)
    assert vocab.idx2word == words, "vocab mismatch: re-run word2vec.py first"
    encoded = [vocab.encode(doc) for doc in tokenized]
    counts = Counter(tok for doc in tokenized for tok in doc)
    n_windows = len(build_cbow_windows(encoded)[1])

    model = CBOW(len(words))
    model.load_state_dict(state)
    model.eval()

    E = state["embeddings.weight"].numpy()
    W = state["output.weight"].numpy()
    b = state["output.bias"].numpy()

    print(f"t-SNE + k-means on the top {N_MAP} words ...")
    X = E[:N_MAP] / np.linalg.norm(E[:N_MAP], axis=1, keepdims=True)
    # numpy 2.0 + macOS Accelerate emits spurious matmul warnings on finite inputs
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        clusters = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(X)
    xy = (xy - xy.mean(axis=0)) / np.abs(xy - xy.mean(axis=0)).max()

    rng = np.random.default_rng(0)
    data = {
        "meta": {
            "embed_dim": EMBED_DIM, "window_size": WINDOW_SIZE, "context_size": CONTEXT_SIZE,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "min_count": MIN_COUNT,
            "max_reviews": MAX_REVIEWS, "vocab_size": len(words), "n_windows": n_windows,
            "random_loss": math.log(len(words)),
        },
        "history": history,
        "words": words,
        "counts": [counts[w] for w in words],
        "E": quantize(E),
        "W": quantize(W),
        "b": np.round(b, 4).tolist(),
        "map": {
            "n": N_MAP,
            "xy": np.round(xy, 4).flatten().tolist(),
            "cluster": clusters.tolist(),
        },
        "sample": pick_sample_review(texts, vocab),
        "presets": pick_presets(model, encoded, words, rng),
        "code": {
            obj.__name__: inspect.getsource(obj).rstrip()
            for obj in (tokenize, Vocab, build_cbow_windows, CBOW, train, most_similar)
        },
    }

    with open(OUT_PATH, "w") as f:
        f.write("window.W2V = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {OUT_PATH.name}  ({len(data['presets'])} presets)")


if __name__ == "__main__":
    main()
