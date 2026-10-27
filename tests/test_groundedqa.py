from fastapi.testclient import TestClient

from groundedqa.answer import NOT_FOUND, Pipeline
from groundedqa.api import create_app
from groundedqa.evals import run_evals
from groundedqa.retrieval import Chunk, Index, chunk_text, tfidf_embed

# toy, clearly fictional corpus: only here to exercise the code
CHUNKS = [
    Chunk("a.txt", "Zorvex is a fictional enzyme that breaks down quillium in the liver."),
    Chunk("b.txt", "The mareline trial enrolled 240 fictional patients across four sites."),
    Chunk("c.txt", "Plant growth in flurnax soil depends on light and moisture."),
]


def make_pipe(reply):
    idx = Index(CHUNKS, tfidf_embed([c.text for c in CHUNKS]))
    return Pipeline(idx, lambda _m: reply() if callable(reply) else reply, k=2)


def test_chunking_overlaps_and_covers():
    words = " ".join(str(i) for i in range(100))
    cs = chunk_text(words, size=40, overlap=10)
    assert cs[0].split()[-10:] == cs[1].split()[:10]
    assert set(" ".join(cs).split()) == set(words.split())
    assert chunk_text("short", 40, 10) == ["short"]


def test_search_ranks_relevant_chunk_first():
    idx = Index(CHUNKS, tfidf_embed([c.text for c in CHUNKS]))
    assert idx.search("what enzyme breaks down quillium", 1)[0].doc == "a.txt"


def test_answer_keeps_valid_citations_only():
    a = make_pipe("Zorvex breaks down quillium [1][9].").answer("what breaks down quillium")
    assert not a.refused and a.citations == [1]


def test_uncited_answer_is_not_trusted():
    assert make_pipe("Zorvex does it.").answer("what breaks down quillium").refused


def test_explicit_refusal():
    a = make_pipe(NOT_FOUND).answer("who won the 1998 world cup")
    assert a.refused and a.text == NOT_FOUND


def test_run_evals_rates():
    cases = [
        {"question": "what breaks down quillium", "answerable": True, "gold_phrase": "zorvex"},
        {"question": "how many patients in the mareline trial", "answerable": True, "gold_phrase": "240"},
        {"question": "who won the 1998 world cup", "answerable": False},
    ]
    replies = iter(["Zorvex [1].", NOT_FOUND, NOT_FOUND])
    r = run_evals(make_pipe(lambda: next(replies)), cases, judge=lambda _m: "yes")
    assert r["retrieval_hit_rate"] == 1.0
    assert r["false_refusal_rate"] == 0.5
    assert r["answer_support_rate"] == 1.0
    assert r["correct_refusal_rate"] == 1.0


def test_unsupported_answer_is_caught_by_judge():
    cases = [{"question": "what breaks down quillium", "answerable": True, "gold_phrase": "zorvex"}]
    r = run_evals(make_pipe("Invented claim [1]."), cases, judge=lambda _m: "no")
    assert r["answer_support_rate"] == 0.0


def test_api_ask():
    client = TestClient(create_app(make_pipe("Zorvex [1].")))
    body = client.post("/ask", json={"question": "what breaks down quillium"}).json()
    assert body["refused"] is False and body["citations"] == [1] and body["sources"][0]["doc"] == "a.txt"
    assert client.post("/ask", json={}).status_code == 422
