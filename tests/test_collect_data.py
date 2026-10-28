"""Tests for streaming statistics (vs brute force) and for the leakage-free document split.

Global context: wrong top-k merging or a leaky split would silently invalidate labels and held-out scores.
"""

import numpy as np
import torch
from scipy import stats as sps

from mamba_neurons.collect import UnitStats, decode_index
from mamba_neurons.data import make_windows, split_documents


def test_unitstats_matches_bruteforce():
    rng = np.random.default_rng(0)
    data = rng.standard_t(df=5, size=(3, 900, 7)).astype(np.float32)  # heavy tails -> nontrivial kurtosis
    acc = UnitStats(n_layers=3, n_units=7, top_k=5, device="cpu")
    for start in range(0, 900, 128):  # uneven final chunk on purpose
        chunk = torch.from_numpy(data[:, start:start + 128])
        idx = torch.arange(start, start + chunk.shape[1])
        for layer in range(3):
            acc.update(layer, chunk[layer], idx)
        acc.count += chunk.shape[1]
    s, st = acc.summary(), acc.state()
    np.testing.assert_allclose(s["mean"], data.mean(1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s["std"], data.std(1), rtol=1e-4)
    np.testing.assert_allclose(s["kurtosis"], sps.kurtosis(data, axis=1, fisher=False), rtol=1e-3)
    np.testing.assert_allclose(s["skew"], sps.skew(data, axis=1), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(s["frac_pos"], (data > 0).mean(1), rtol=1e-6)
    order = np.argsort(-data, axis=1)[:, :5]  # brute-force top-5 positions per (layer, unit)
    np.testing.assert_array_equal(st["top_idx"].numpy(), order)
    np.testing.assert_allclose(st["bot_val"].numpy(), np.sort(data, axis=1)[:, :5])  # most negative first


def test_decode_index_roundtrip():
    rows, pos = decode_index([0, 127, 128, 300], ctx_len=128)
    assert rows.tolist() == [0, 0, 1, 2] and pos.tolist() == [1, 128, 1, 45]


def test_document_split_is_disjoint_and_complete():
    disc, held = split_documents(1000, 0.3, seed=0)
    assert len(set(disc) & set(held)) == 0  # no leakage
    assert len(disc) + len(held) == 1000 and len(held) == 300


class _FakeTok:
    def __call__(self, text):
        return type("Enc", (), {"input_ids": [int(c) for c in text.split()]})()


def test_windows_never_mix_documents():
    texts = [" ".join(str(i * 100 + j) for j in range(n)) for i, n in enumerate([10, 3, 25])]
    w = make_windows(_FakeTok(), texts, ["a", "b", "c"], [0, 1, 2], ctx_len=4, max_windows_per_doc=3, max_tokens=100)
    assert w.tokens.shape[1] == 5 and (w.tokens[:, 0] == 0).all()  # separator + 4 tokens
    assert w.doc.tolist() == [0, 0, 2, 2, 2]  # doc 1 is too short for one window; doc 2 capped at 3 windows
    for row, d in zip(w.tokens, w.doc):
        assert all(t // 100 == d for t in row[1:])  # every token of a window comes from its own document
