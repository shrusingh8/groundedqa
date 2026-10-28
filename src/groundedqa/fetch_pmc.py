"""Download CC BY open-access papers from PubMed Central as plain text into a docs folder.

python -m groundedqa.fetch_pmc "large language model AND clinical" --n 20 --out data/docs
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
XLINK = "{http://www.w3.org/1999/xlink}href"


def _get(endpoint: str, **params) -> bytes:
    with urllib.request.urlopen(EUTILS + endpoint + "?" + urllib.parse.urlencode(params), timeout=30) as r:
        return r.read()


def search(query: str, n: int) -> List[str]:
    term = f"({query}) AND cc by license[filter]"
    return json.loads(_get("esearch.fcgi", db="pmc", term=term, retmax=n, retmode="json"))["esearchresult"]["idlist"]


def _text(el: Optional[ET.Element]) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def jats_to_text(xml: bytes) -> Optional[str]:
    """Title, license, abstract and body paragraphs; None if the full text isn't available."""
    art = ET.fromstring(xml).find(".//article")
    paras = [_text(p) for p in art.findall("./body//p")] if art is not None else []
    if not paras:
        return None
    lic = art.find(".//license")
    license_ = (lic.get(XLINK) or _text(lic)) if lic is not None else "unknown"
    title = _text(art.find(".//article-meta//article-title"))
    abstract = _text(art.find(".//article-meta/abstract"))
    return "\n\n".join([f"Title: {title}", f"License: {license_}", abstract, *paras])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("query")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="data/docs")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for pmc_id in search(a.query, a.n * 2):  # extra ids because some records have no full text
        time.sleep(0.4)  # NCBI allows ~3 requests/second without an API key
        text = jats_to_text(_get("efetch.fcgi", db="pmc", id=pmc_id, rettype="xml"))
        if text:
            (out / f"PMC{pmc_id}.txt").write_text(text, encoding="utf-8")
            saved += 1
            title = text.splitlines()[0][7:90].encode("ascii", "replace").decode()  # Windows consoles choke on unicode
            print(f"PMC{pmc_id}: {title}")
        if saved == a.n:
            break
    print(f"saved {saved} papers to {out}")


if __name__ == "__main__":
    main()
