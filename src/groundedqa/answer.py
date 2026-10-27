"""Generate an answer that must cite its sources, or refuse when the sources don't contain it."""
import re
from dataclasses import dataclass, field
from typing import List

from .llm import Ask
from .retrieval import Chunk, Index

NOT_FOUND = "NOT_FOUND"
SYSTEM = (
    "Answer the question using ONLY the numbered sources. Cite the sources you used like [1]. "
    f"If the sources do not contain the answer, reply exactly: {NOT_FOUND}"
)


@dataclass
class Answer:
    text: str
    refused: bool
    citations: List[int] = field(default_factory=list)
    sources: List[Chunk] = field(default_factory=list)


def build_messages(question: str, sources: List[Chunk]):
    body = "\n\n".join(f"[{i}] ({c.doc}) {c.text}" for i, c in enumerate(sources, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sources:\n{body}\n\nQuestion: {question}"}]


class Pipeline:
    def __init__(self, index: Index, ask: Ask, k: int = 4):
        self.index, self.ask, self.k = index, ask, k

    def answer(self, question: str) -> Answer:
        sources = self.index.search(question, self.k)
        text = self.ask(build_messages(question, sources)).strip()
        if NOT_FOUND in text:
            return Answer(NOT_FOUND, True, [], sources)
        cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(sources)})
        # an answer with no valid citation is treated as unsupported, not trusted
        return Answer(text, not cites, cites, sources)
