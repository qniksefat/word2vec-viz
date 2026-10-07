"""
train a word2vec model on a corpus of text
cbow (continuous bag of words) model

given the words around a position (the context), predict the word in the middle (the target):
    context embeddings -> average -> linear -> softmax over vocab
"""

# install libraries
# pip install datasets torch numpy

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# hyperparameters
EMBED_DIM = 100     # size of the embedding vector
WINDOW_SIZE = 5     # total window: 2 context words on each side + the target in the middle
CONTEXT_SIZE = WINDOW_SIZE // 2
EPOCHS = 10         # iterations
BATCH_SIZE = 1024
LR = 1e-3
MIN_COUNT = 5       # drop words seen fewer times than this
MAX_REVIEWS = 25000  # subset of the 25k reviews to keep training fast on a laptop

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACT_PREFIX = Path(__file__).resolve().parent / "word2vec"  # -> word2vec/word2vec_*.{npy,json,pt}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_corpus(max_reviews=MAX_REVIEWS):
    """corpus of text download from a library (imdb reviews)"""
    dataset = load_dataset("imdb")
    return dataset["train"][:max_reviews]["text"]


def tokenize(text):
    """lowercase, strip html line breaks, keep only word-like tokens"""
    text = text.lower().replace("<br />", " ")
    return [word for word in text.split() if word.isalpha()]


class Vocab:
    """maps words <-> integer ids"""

    def __init__(self, tokenized_docs, min_count=MIN_COUNT):
        counts = Counter(tok for doc in tokenized_docs for tok in doc)
        words = [w for w, c in counts.most_common() if c >= min_count]
        self.idx2word = words
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.counts = counts

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word):
        return word in self.word2idx

    def encode(self, tokens):
        """tokens -> ids, dropping out-of-vocab words"""
        return [self.word2idx[t] for t in tokens if t in self.word2idx]


def build_cbow_windows(encoded_docs, context_size=CONTEXT_SIZE):
    """lets take all the windows ready: returns (contexts [N, 2*c], targets [N])"""
    window = 2 * context_size + 1
    contexts, targets = [], []
    for ids in encoded_docs:
        if len(ids) < window:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(np.asarray(ids), window)
        targets.append(windows[:, context_size])
        contexts.append(np.delete(windows, context_size, axis=1))
    contexts = torch.from_numpy(np.concatenate(contexts)).long()
    targets = torch.from_numpy(np.concatenate(targets)).long()
    return contexts, targets


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class CBOW(nn.Module):
    """the small nn: embedding table + one linear layer"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids):
        # context_ids: [B, 2*c] -> [B, 2*c, D] -> mean over context -> [B, D]
        hidden = self.embeddings(context_ids).mean(dim=1)
        return self.output(hidden)  # logits: [B, V]


def train(model, loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for contexts, targets in loader:
            contexts, targets = contexts.to(DEVICE), targets.to(DEVICE)
            logits = model(contexts)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            n += targets.size(0)
        history.append(total_loss / n)
        print(f"epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}")
    return history


# ---------------------------------------------------------------------------
# inspecting the embeddings
# ---------------------------------------------------------------------------

def get_embeddings(model):
    """[V, D] embedding matrix as numpy"""
    return model.embeddings.weight.detach().cpu().numpy()


def most_similar(word, embeddings, vocab, k=5):
    """nearest neighbours by cosine similarity"""
    if word not in vocab:
        return []
    emb = torch.from_numpy(embeddings)
    sims = torch.cosine_similarity(emb, emb[vocab.word2idx[word]].unsqueeze(0), dim=1)
    best = torch.topk(sims, k + 1).indices[1:]  # skip the word itself
    return [(vocab.idx2word[i], round(sims[i].item(), 3)) for i in best.tolist()]


def save_embeddings(embeddings, vocab, path_prefix=ARTIFACT_PREFIX):
    """saved as .npy + .json so the embed vis file can load them"""
    np.save(f"{path_prefix}_embeddings.npy", embeddings)
    with open(f"{path_prefix}_vocab.json", "w") as f:
        json.dump(vocab.idx2word, f)


def save_model(model, history, path_prefix=ARTIFACT_PREFIX):
    """full weights (embeddings + output layer) and the per-epoch loss curve"""
    torch.save(model.state_dict(), f"{path_prefix}_cbow.pt")
    with open(f"{path_prefix}_history.json", "w") as f:
        json.dump(history, f)


# in embed vis file we visualize the embedding vectors in a 2D space using t-SNE


def main():
    texts = load_corpus()

    # split the text into tokens
    tokenized = [tokenize(t) for t in texts]
    vocab = Vocab(tokenized)
    encoded = [vocab.encode(doc) for doc in tokenized]
    print(f"vocab size: {len(vocab)}")

    contexts, targets = build_cbow_windows(encoded)
    print(f"training windows: {len(targets)}")
    loader = DataLoader(TensorDataset(contexts, targets), batch_size=BATCH_SIZE, shuffle=True)

    model = CBOW(len(vocab))
    history = train(model, loader)

    embeddings = get_embeddings(model)
    for word in ["good", "bad", "movie", "actor", "funny"]:
        print(word, "->", most_similar(word, embeddings, vocab))

    save_embeddings(embeddings, vocab)
    save_model(model, history)


if __name__ == "__main__":
    main()
