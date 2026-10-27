"""The only place that talks to a model API; everything else takes an `ask` / `embed` callable."""
import os
from typing import Callable, Dict, List

import numpy as np

Messages = List[Dict[str, str]]
Ask = Callable[[Messages], str]
Embed = Callable[[List[str]], np.ndarray]


def openai_ask(model: str = "gpt-4o-mini") -> Ask:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return lambda m: client.chat.completions.create(model=model, messages=m, temperature=0).choices[0].message.content


def openai_embed(model: str = "text-embedding-3-small") -> Embed:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(texts: List[str]) -> np.ndarray:
        r = client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in r.data])

    return embed
