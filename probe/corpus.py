"""Corpus builder: public domain text for neuron analysis.

Gutenberg plain text, with headers stripped. Falls back to a built-in sample if
the network is unavailable, so the atlas still runs offline.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

BOOKS = [
    ("pride_and_prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
    ("moby_dick", "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("tale_of_two_cities", "https://www.gutenberg.org/files/98/98-0.txt"),
    ("frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
]

FALLBACK = """
The deep ocean is the largest habitat on the planet. Below two hundred meters
sunlight fades to nothing, and the only light left is produced by living things.
A compiler translates source code into machine instructions. It first parses the
text into a tree, then checks that the tree is well formed. The stock market fell
sharply on Tuesday after a major bank reported losses larger than expected.
Photosynthesis converts light into chemical energy, and chlorophyll absorbs red
and blue wavelengths while reflecting green.
"""


def _strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      text, re.IGNORECASE | re.DOTALL)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG",
                    text, re.IGNORECASE)
    if end:
        text = text[:end.start()]
    return text.strip()


def fetch_book(name: str, url: str, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.txt"
    if path.exists() and path.stat().st_size > 10_000:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "neuron-atlas/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", errors="replace")
        text = _strip_gutenberg(raw)
        path.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"  ! {name}: {type(e).__name__} {e}")
        return None


def build_corpus(cache_dir: Path, verbose: bool = True) -> list[str]:
    """Return a list of paragraph-level documents."""
    docs: list[str] = []
    for name, url in BOOKS:
        text = fetch_book(name, url, cache_dir)
        if not text:
            continue
        if verbose:
            print(f"  {name}: {len(text):,} chars")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
        docs.extend(p for p in paras if len(p) > 120)

    if not docs:
        if verbose:
            print("  network unavailable, using built-in fallback text")
        docs = [FALLBACK.strip()]

    if verbose:
        print(f"  {len(docs)} documents, {sum(len(d) for d in docs):,} chars total")
    return docs
