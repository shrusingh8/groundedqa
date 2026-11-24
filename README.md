# groundedqa

Question answering over your own documents (RAG) where **every answer must cite its sources or refuse**,
plus an eval harness that measures whether the answers are actually supported.

The point of the project is the evals: a RAG demo is easy, knowing whether it is *right* is the hard part.

## How it works

```
documents (.txt) -> chunks -> embeddings -> cosine search -> top-k passages
                                                              |
question ---------------------------------------------------> LLM (must cite [n] or reply NOT_FOUND)
                                                              |
                         answer + citations + passages <------+
```

- **Citations are enforced.** An answer with no valid `[n]` citation is treated as a refusal, not trusted.
- **Refusal is a feature.** If the passages don't contain the answer, the model must say `NOT_FOUND`.
- **Model calls sit behind two functions** (`ask`, `embed`), so all logic is tested without an API key.

## Evals

`data/eval_cases.json` holds questions marked answerable or not. The harness reports:

| Metric | Meaning |
|---|---|
| `retrieval_hit_rate` | Did the retrieved passages contain the gold phrase? (retrieval quality, separate from generation) |
| `answer_support_rate` | Of answers given, how many an LLM judge says are fully supported by the passages (a hallucination check) |
| `false_refusal_rate` | Answerable questions the system wrongly refused |
| `correct_refusal_rate` | Unanswerable questions the system correctly refused |

Splitting retrieval from generation shows *where* a failure happens, which is what you fix next.

### Results

**Corpus:** 20 CC BY papers from PubMed Central on clinical uses of LLMs (fetched Nov 2026, ~114K words,
1,272 chunks of 120 words). **Cases:** `data/pmc_cases.json`, 50 hand-written questions: 40 answerable
(2 per paper, each gold phrase verified verbatim in its paper) and 10 unanswerable. Several unanswerable
questions are near misses on purpose, e.g. asking for a Net Promoter Score for the wrong study, so a
system that matches keywords without reading will answer them.

**Retrieval (keyless TF-IDF baseline):**

| Chunk size | Hit rate @1 | @4 (default) | @8 | Right paper in top 4 |
|---|---|---|---|---|
| 120 words | 47.5% | **80.0%** | 87.5% | 100% |
| 60 words | 40.0% | 75.0% | 90.0% | 100% |
| 40 words | 30.0% | 65.0% | 82.5% | 100% |

What the failures show:

- **Every miss found the right paper but the wrong passage.** All 8 misses at k=4 are numeric facts
  (an MAE of 0.825, "1573 sessions", "4321" training records). The question's keywords appear in many
  passages of the same paper, and the one passage holding the number does not stand out.
- **Smaller chunks made it worse, not better.** Cutting chunks to 40 words dropped the hit rate from 80% to
  65%, because a short chunk with the number loses the surrounding words the query matches on.
- **Next fix:** rerank the top 8 to 10 passages (cross-encoder or LLM) instead of shrinking chunks,
  since k=8 already reaches 87.5%.

**Semantic embeddings and generation** (`text-embedding-3-small` + `gpt-4o-mini`): run with an OpenAI key.

```bash
python -m groundedqa.evals --docs data/docs --cases data/pmc_cases.json --out results.json
```

That prints the four metrics above and writes every answer, refusal, and retrieved passage to
`results.json` for failure review.

Reproduce the corpus (downloaded papers stay out of git; each case's `doc` field is its PMC ID):

```bash
python -m groundedqa.fetch_pmc '"large language model"[Title] AND (clinical OR medical)[Title]' --n 20
python -m groundedqa.evals --docs data/docs --cases data/pmc_cases.json --embed tfidf --retrieval-only
```

## Run

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest                                   # 11 tests, no network or key needed

export OPENAI_API_KEY=...                          # never commit keys
python -m groundedqa.evals --docs data/docs --cases data/eval_cases.json
uvicorn groundedqa.api:app_from_env --factory      # POST /ask {"question": "..."}
```

Docker: `docker build -t groundedqa . && docker run -e OPENAI_API_KEY -p 8000:8000 groundedqa`

## Limits (honest list)

- Brute-force cosine search over a numpy matrix: fine for thousands of chunks, not millions.
- Chunking is by words with overlap; it ignores section structure and tables.
- The support check is an LLM judge, which is itself noisy; spot-check its verdicts by hand.
- No auth, rate limiting, or prompt-injection defenses on the API yet.
