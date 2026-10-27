"""FastAPI wrapper: POST /ask -> answer, citations, and the passages used."""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from .answer import Pipeline


class AskRequest(BaseModel):
    question: str


def create_app(pipe: Pipeline) -> FastAPI:
    app = FastAPI(title="groundedqa")

    @app.post("/ask")
    def ask(req: AskRequest):
        a = pipe.answer(req.question)
        return {"answer": a.text, "refused": a.refused, "citations": a.citations,
                "sources": [{"doc": c.doc, "text": c.text} for c in a.sources]}

    return app


def app_from_env() -> FastAPI:  # uvicorn groundedqa.api:app_from_env --factory
    from .llm import openai_ask, openai_embed
    from .retrieval import Index, load_chunks

    return create_app(Pipeline(Index(load_chunks(os.environ.get("DOCS_DIR", "data/docs")), openai_embed()), openai_ask()))
