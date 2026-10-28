"""Corpus loading, document-level train/held-out split, fixed-length windows and neutral prompts.

Local explanation: documents from the Pile sample `NeelNanda/pile-10k` (https://huggingface.co/datasets/NeelNanda/pile-10k;
Mamba was pretrained on the Pile, Gu & Dao 2023) are shuffled with a fixed seed and split *by document* into a
discovery split (top examples, labels, candidate selection) and a held-out split (detection scoring, steering
prompts, ablations). Each document is tokenized and cut into non-overlapping windows of `ctx_len` tokens, each
prefixed by the <|endoftext|> separator (id 0) the Pile models saw between documents.
Global context: splitting by document (not by window) prevents leakage of the same text into both splits
(Lones 2021, https://arxiv.org/abs/2108.02497, pitfall "do not allow test data to leak into training").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset  # https://huggingface.co/docs/datasets/loading

from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Windows:
    """Token windows of one split. `tokens[:, 0]` is the separator; `doc[j]`, `start[j]` locate window j."""

    tokens: np.ndarray  # (n_windows, ctx_len + 1) int32 token ids
    doc: np.ndarray  # (n_windows,) int32 index into the corpus
    start: np.ndarray  # (n_windows,) int32 token offset of the window inside its document
    sources: list[str]  # Pile subset name per *document* (indexed by `doc`), e.g. "Github"

    @property
    def n_tokens(self) -> int:
        """Analysed tokens (separator excluded)."""
        return int(self.tokens.shape[0] * (self.tokens.shape[1] - 1))


def load_corpus(dataset_name: str) -> tuple[list[str], list[str]]:
    """Return (texts, pile_set_names) for every document of the dataset's train split."""
    ds = load_dataset(dataset_name, split="train")
    sources = [m.get("pile_set_name", "?") for m in ds["meta"]] if "meta" in ds.column_names else ["?"] * len(ds)
    return list(ds["text"]), sources


def split_documents(n_docs: int, heldout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random permutation of document ids cut into (discovery_ids, heldout_ids); disjoint by construction."""
    perm = np.random.default_rng(seed).permutation(n_docs)  # numpy Generator API (https://numpy.org/doc/stable/reference/random/generator.html)
    n_held = int(round(n_docs * heldout_fraction))
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


def make_windows(tokenizer, texts, sources, doc_ids, ctx_len: int, max_windows_per_doc: int, max_tokens: int,
                 sep_id: int = 0) -> Windows:
    """Cut the documents `doc_ids` into full windows of `ctx_len` tokens (shorter tails are dropped)."""
    rows, docs, starts = [], [], []
    budget = max_tokens // ctx_len  # number of windows allowed by the token budget
    for d in doc_ids:  # documents are visited in sorted-id order; the split permutation already randomised them
        ids = tokenizer(texts[d]).input_ids
        for w in range(min(max_windows_per_doc, len(ids) // ctx_len)):  # only complete windows -> no padding
            rows.append([sep_id] + ids[w * ctx_len:(w + 1) * ctx_len])
            docs.append(d)
            starts.append(w * ctx_len)
            if len(rows) >= budget:
                break
        if len(rows) >= budget:
            break
    windows = Windows(np.asarray(rows, np.int32), np.asarray(docs, np.int32), np.asarray(starts, np.int32), list(sources))
    log.info("Built %d windows (%d tokens) from %d documents", len(rows), windows.n_tokens, len(set(docs)))
    return windows


def build_splits(tokenizer, settings) -> dict[str, Windows]:
    """Discovery + held-out windows according to `settings` (budgets, seed, window size)."""
    texts, sources = load_corpus(settings.dataset_name)
    disc_ids, held_ids = split_documents(len(texts), settings.heldout_fraction, settings.seed)
    common = dict(ctx_len=settings.ctx_len, max_windows_per_doc=settings.max_windows_per_doc)
    return {
        "discovery": make_windows(tokenizer, texts, sources, disc_ids, max_tokens=settings.max_discovery_tokens, **common),
        "heldout": make_windows(tokenizer, texts, sources, held_ids, max_tokens=settings.max_heldout_tokens, **common),
    }


def decode_tokens(tokenizer, ids) -> list[str]:
    """Per-token strings whose concatenation equals the decoded text, without splitting multi-byte characters.

    Byte-level BPE (GPT-2 / GPT-NeoX) can cut one UTF-8 character across tokens, so `decode([id])` yields U+FFFD.
    We map each token back to its raw bytes (inverse of `bytes_to_unicode`, as in GPT-2's encoder.py,
    https://github.com/openai/gpt-2/blob/master/src/encoder.py) and feed them to an incremental UTF-8 decoder
    (https://docs.python.org/3/library/codecs.html#codecs.getincrementaldecoder), which assigns each character to the
    token that completes it.
    """
    import codecs

    from transformers.convert_slow_tokenizer import bytes_to_unicode

    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out = []
    for tok in tokenizer.convert_ids_to_tokens([int(i) for i in ids]):
        raw = bytes(byte_of[c] for c in tok) if all(c in byte_of for c in tok) else tok.encode("utf-8")  # special tokens
        out.append(dec.decode(raw))
    if out:
        out[-1] += dec.decode(b"", final=True)  # flush an unfinished trailing character
    return out


def save_windows(windows: Windows, path) -> None:
    """Persist windows as compressed npz (sources stored as a string array)."""
    np.savez_compressed(path, tokens=windows.tokens, doc=windows.doc, start=windows.start, sources=np.asarray(windows.sources))


def load_windows(path) -> Windows:
    """Inverse of `save_windows`."""
    z = np.load(path)
    return Windows(z["tokens"], z["doc"], z["start"], list(z["sources"]))
