"""Eval harness: retrieval hit rate, answer support (LLM judge), refusals on unanswerable questions."""
import argparse
import json
from typing import Dict, List

from .answer import Answer, Pipeline
from .llm import Ask

JUDGE = (
    "Sources:\n{sources}\n\nAnswer: {answer}\n\n"
    "Is every factual claim in the answer supported by the sources? Reply with one word: yes or no."
)


def judge_supported(judge: Ask, ans: Answer) -> bool:
    src = "\n".join(c.text for c in ans.sources)
    return judge([{"role": "user", "content": JUDGE.format(sources=src, answer=ans.text)}]).strip().lower().startswith("yes")


def run_evals(pipe: Pipeline, cases: List[Dict], judge: Ask) -> Dict[str, float]:
    """cases: {question, answerable, gold_phrase?}. Rates are fractions of the relevant subset."""
    hits = supported = refused_ok = false_refusals = 0
    answerable = [c for c in cases if c["answerable"]]
    unanswerable = [c for c in cases if not c["answerable"]]
    for c in answerable:
        ans = pipe.answer(c["question"])
        hits += any(c["gold_phrase"].lower() in s.text.lower() for s in ans.sources)
        if ans.refused:
            false_refusals += 1
        else:
            supported += judge_supported(judge, ans)
    for c in unanswerable:
        refused_ok += pipe.answer(c["question"]).refused
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
    from .llm import openai_ask, openai_embed
    from .retrieval import Index, load_chunks

    p = argparse.ArgumentParser(description="Run evals against a folder of .txt documents.")
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--cases", default="data/eval_cases.json")
    p.add_argument("--k", type=int, default=4)
    a = p.parse_args()
    pipe = Pipeline(Index(load_chunks(a.docs), openai_embed()), openai_ask(), a.k)
    print(json.dumps(run_evals(pipe, json.load(open(a.cases, encoding="utf-8")), openai_ask()), indent=2))


if __name__ == "__main__":
    main()
