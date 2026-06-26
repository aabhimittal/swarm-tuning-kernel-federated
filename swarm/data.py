"""Character-level dataset for TinyGPT.

The corpus is bundled in the package so every node derives an *identical* vocab
(sorted unique characters) without any network round-trip — critical, because the
server and all workers must agree on token ids and ``vocab_size`` for gradients to
be meaningful.

A worker can take a disjoint ``shard`` of the corpus (``--shard i/N``) to simulate
federated, non-IID data spread across the swarm.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")


def load_corpus(path: Optional[str] = None) -> str:
    """Read the training text. Defaults to the bundled corpus.

    ``SWARM_CORPUS`` env var or an explicit ``path`` overrides it (e.g. a larger
    corpus mounted on the server). The vocab is still derived from whatever text is
    loaded, so every node MUST use the same corpus.
    """
    path = path or os.environ.get("SWARM_CORPUS") or _CORPUS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@dataclass
class CharTokenizer:
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # Sorted for determinism across machines/runs.
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


@dataclass
class Dataset:
    data: torch.Tensor  # 1-D LongTensor of token ids
    tokenizer: CharTokenizer
    block_size: int

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(
        self, batch_size: int, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (inputs, targets), each (batch_size, block_size)."""
        n = self.data.size(0) - self.block_size - 1
        if n <= 0:
            raise ValueError(
                f"shard too small ({self.data.size(0)} tokens) for block_size {self.block_size}"
            )
        ix = torch.randint(0, n, (batch_size,), generator=generator)
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix])
        return x, y


def build_dataset(
    block_size: int,
    shard: int = 0,
    num_shards: int = 1,
    corpus_path: Optional[str] = None,
) -> Dataset:
    """Build a Dataset, optionally restricted to one shard of the corpus.

    The tokenizer/vocab is always derived from the *full* corpus so it is identical
    on every node; only the token slice used for sampling is sharded.
    """
    text = load_corpus(corpus_path)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    if num_shards > 1:
        if not (0 <= shard < num_shards):
            raise ValueError(f"shard {shard} out of range for num_shards {num_shards}")
        shard_len = ids.size(0) // num_shards
        start = shard * shard_len
        end = start + shard_len if shard < num_shards - 1 else ids.size(0)
        ids = ids[start:end]

    return Dataset(data=ids, tokenizer=tokenizer, block_size=block_size)
