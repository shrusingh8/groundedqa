from groundedqa.fetch_pmc import jats_to_text

XML = b"""<pmc-articleset><article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
<title-group><article-title>A <italic>Toy</italic> Paper</article-title></title-group>
<permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><p>CC BY</p></license></permissions>
<abstract><p>Short abstract.</p></abstract></article-meta></front>
<body><sec><title>Intro</title><p>First   paragraph.</p><p>Second <bold>bold</bold> one.</p></sec></body></article></pmc-articleset>"""


def test_jats_to_text_extracts_sections():
    t = jats_to_text(XML)
    assert t.startswith("Title: A Toy Paper")
    assert "License: https://creativecommons.org/licenses/by/4.0/" in t
    assert "Short abstract." in t and "First paragraph." in t and "Second bold one." in t


def test_no_body_means_no_text():
    assert jats_to_text(b"<pmc-articleset><article><front/></article></pmc-articleset>") is None
