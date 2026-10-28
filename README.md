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

_Not yet populated._ Download 15 to 20 CC BY papers from PubMed Central, write 30 to 50 cases in
`data/eval_cases.json`, run the evals, and paste the table here with a short note on the failures found.
The `toy` example cases use a fictional corpus and only demonstrate the format.

```bash
python -m groundedqa.fetch_pmc '"large language model"[Title] AND (clinical OR medical)[Title]' --n 20
```

Downloaded papers stay out of git (see `.gitignore`); their PMC IDs are the file names, so the corpus is reproducible.

## Run

```bash
pip install -r requirements.txt && pip install -e .
python -m pytest                                   # 10 tests, no network or key needed

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
