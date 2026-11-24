"""Eval harness: retrieval hit rate, answer support (LLM judge), refusals on unanswerable questions."""
import argparse
import json
from typing import Dict, List, Optional

from .answer import Answer, Pipeline
from .llm import Ask
from .retrieval import Index

JUDGE = (
    "Sources:\n{sources}\n\nAnswer: {answer}\n\n"
    "Is every factual claim in the answer supported by the sources? Reply with one word: yes or no."
)


def judge_supported(judge: Ask, ans: Answer) -> bool:
    src = "\n".join(c.text for c in ans.sources)
    return judge([{"role": "user", "content": JUDGE.format(sources=src, answer=ans.text)}]).strip().lower().startswith("yes")


def _hit(c: Dict, sources) -> bool:
    return any(c["gold_phrase"].lower() in s.text.lower() for s in sources)


def retrieval_hit_rate(index: Index, cases: List[Dict], k: int = 4) -> float:
    """Retrieval only, no LLM calls: share of answerable cases whose top-k passages contain the gold phrase."""
    answerable = [c for c in cases if c["answerable"]]
    return round(sum(_hit(c, index.search(c["question"], k)) for c in answerable) / len(answerable), 3)


def run_evals(pipe: Pipeline, cases: List[Dict], judge: Ask, records: Optional[List[Dict]] = None) -> Dict[str, float]:
    """cases: {question, answerable, gold_phrase?}. Rates are fractions of the relevant subset.
    Pass a list as `records` to collect one row per case for failure analysis."""
    hits = supported = refused_ok = false_refusals = 0
    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    log = records if records is not None else []
    for c in answerable:
        ans = pipe.answer(c["question"])
        hit = _hit(c, ans.sources)
        ok = None if ans.refused else judge_supported(judge, ans)
        hits += hit
        false_refusals += ans.refused
        supported += bool(ok)
        log.append({"question": c["question"], "answerable": True, "retrieval_hit": hit, "refused": ans.refused,
                    "supported": ok, "answer": ans.text, "docs": [s.doc for s in ans.sources]})
    for c in unanswerable:
        ans = pipe.answer(c["question"])
        refused_ok += ans.refused
        log.append({"question": c["question"], "answerable": False, "refused": ans.refused, "answer": ans.text,
                    "docs": [s.doc for s in ans.sources]})
    rate = lambda n, d: round(n / d, 3) if d else float("nan")
    return {
        "retrieval_hit_rate": rate(hits, len(answerable)),
        "answer_support_rate": rate(supported, len(answerable) - false_refusals),
        "false_refusal_rate": rate(false_refusals, len(answerable)),
        "correct_refusal_rate": rate(refused_ok, len(unanswerable)),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
    }


def main() -> None:
    from .retrieval import load_chunks, tfidf_embed

    p = argparse.ArgumentParser(description="Run evals against a folder of .txt documents.")
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--cases", default="data/eval_cases.json")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--embed", choices=["openai", "tfidf"], default="openai")
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM; no API key needed with --embed tfidf")
    p.add_argument("--out", help="write per-case rows (answers, hits, refusals) to this JSON file")
    a = p.parse_args()
    chunks = load_chunks(a.docs)
    cases = json.load(open(a.cases, encoding="utf-8"))
    if a.embed == "tfidf":
        embed = tfidf_embed([c.text for c in chunks])
    else:
        from .llm import openai_embed
        embed = openai_embed()
    index = Index(chunks, embed)
    if a.retrieval_only:
        print(json.dumps({"retrieval_hit_rate": retrieval_hit_rate(index, cases, a.k), "embed": a.embed, "k": a.k}))
        return
    from .llm import openai_ask
    rows: List[Dict] = []
    result = run_evals(Pipeline(index, openai_ask(), a.k), cases, openai_ask(), rows)
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=2)
    print(json.dumps({**result, "embed": a.embed, "k": a.k}, indent=2))


if __name__ == "__main__":
    main()
