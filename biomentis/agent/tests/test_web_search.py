"""Tests for web search through Ollama's web search API.

On 2026-10-28 DuckDuckGo began answering the keyless scrapers with its
bot-detection page, so every `search_google` / `keyless_web_search` call
returned "No web results" and a 23-minute CHO-workflow run had only PubMed
abstracts. With OLLAMA_API_KEY set, search now goes through
https://ollama.com/api/web_search first, and `extract_url_content` reuses the
page text a search returned or fetches through https://ollama.com/api/web_fetch.

What these tests pin down (with a stand-in client, no network):

  1.  With a key, Ollama's search is tried first and its results are formatted
      as before (Title / URL / Description)
  2.  Each description is the passage that best matches the query, not the
      page's opening (often navigation)
  3.  A page a search returned is read back without another request
  4.  Otherwise extract_url_content fetches through Ollama, and falls back to
      a direct download when that fails
  5.  Without a key, or when Ollama fails, the scrapers are used as before,
      and the failure is named in the diagnostic

Run with:
    python -m biomentis.agent.tests.test_web_search
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

# --- Tiny test harness (matches test_smoke_e2e.py) -----------------------

_PASSED: list[str] = []
_FAILED: list[tuple[str, str]] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        _PASSED.append(name)
        print(f"  ✓ {name}")
    else:
        _FAILED.append((name, detail))
        print(f"  ✗ {name}: {detail}")


_PAGE = (
    "Home About Contact Login Subscribe Cookie settings. " * 20
    + "In harvest cell culture fluid the mass ratio of host cell protein to antibody is around 1:1, "
    "and Protein A capture removes most HCP. "
    + "Related articles and footer links. " * 30
)


class _FakeClient:
    def __init__(self, fetch_fails: bool = False):
        self.searches: list[str] = []
        self.fetches: list[str] = []
        self.fetch_fails = fetch_fails

    def web_search(self, query, max_results=3):
        self.searches.append(query)
        return SimpleNamespace(results=[
            SimpleNamespace(title="HCP in CHO harvest", url="https://example.org/hcp", content=_PAGE),
            SimpleNamespace(title="Short page", url="https://example.org/short", content="Brief text."),
        ][:max_results])

    def web_fetch(self, url):
        self.fetches.append(url)
        if self.fetch_fails:
            raise RuntimeError("not found (status code: 404)")
        return SimpleNamespace(title="Fetched", content=f"Fetched text of {url}", links=[])


def _patch(lit, client, scrapers=None):
    """Swap the Ollama client and, optionally, the scrapers; return a restore function."""
    saved = {name: getattr(lit, name) for name in (
        "_ollama_web_client", "_search_duckduckgo_html", "_search_duckduckgo_lite",
        "_search_googlesearch_python", "_scrape_url_content")}
    lit._ollama_web_client = lambda: client
    scrapers = scrapers or {}
    for name in ("_search_duckduckgo_html", "_search_duckduckgo_lite", "_search_googlesearch_python"):
        lit.__dict__[name] = scrapers.get(name, lambda *a, **k: [])
    lit._scrape_url_content = scrapers.get("_scrape_url_content", lambda url: f"Scraped text of {url}")
    lit._WEB_PAGE_CACHE.clear()

    def restore():
        for name, value in saved.items():
            setattr(lit, name, value)
        lit._WEB_PAGE_CACHE.clear()

    return restore


def test_ollama_first() -> None:
    print("\n[1-2] Ollama's search first; descriptions are the best-matching passage")
    from biomentis.tool import literature as lit

    client = _FakeClient()
    called = []
    restore = _patch(lit, client, {"_search_duckduckgo_html": lambda *a, **k: called.append("ddg") or []})
    try:
        out = lit.search_google("host cell protein harvest antibody ratio", num_results=2)
    finally:
        restore()
    _check("Ollama was asked", client.searches == ["host cell protein harvest antibody ratio"], str(client.searches))
    _check("the scrapers were not", called == [], str(called))
    _check("same format as before", out.startswith("Title: HCP in CHO harvest\nURL: https://example.org/hcp\nDescription: "),
           out[:200])
    _check("the description is the matching passage", "mass ratio of host cell protein to antibody" in out, out[:900])
    _check("not the navigation at the top", "Cookie settings" not in out.split("Description: ")[1][:80], out[:900])
    _check("a short page is shown whole", "Description: Brief text." in out, out)


def test_extract_reuses_and_fetches() -> None:
    print("\n[3-4] reading a page: from the search, then Ollama fetch, then download")
    from biomentis.tool import literature as lit

    client = _FakeClient()
    restore = _patch(lit, client)
    try:
        lit.search_google("host cell protein", num_results=2)
        page = lit.extract_url_content("https://example.org/hcp")
        _check("a searched page comes back whole", page == _PAGE, page[:100])
        _check("without a fetch", client.fetches == [], str(client.fetches))
        other = lit.extract_url_content("https://example.org/other")
        _check("an unseen page is fetched through Ollama", other == "Fetched text of https://example.org/other", other)
    finally:
        restore()

    client = _FakeClient(fetch_fails=True)
    restore = _patch(lit, client)
    try:
        out = lit.extract_url_content("https://doi.org/10.1/x")
    finally:
        restore()
    _check("a failed fetch falls back to downloading", out == "Scraped text of https://doi.org/10.1/x", out)


def test_fallbacks() -> None:
    print("\n[5] without a key, or when Ollama fails, the scrapers are used")
    from biomentis.tool import literature as lit

    hit = [{"title": "DDG hit", "url": "https://example.org/ddg", "description": "From DuckDuckGo"}]
    restore = _patch(lit, None, {"_search_duckduckgo_html": lambda *a, **k: hit})
    try:
        out = lit.keyless_web_search("anything", num_results=1)
        none_out = None
    finally:
        restore()
    _check("no key: the scrapers answer", "DDG hit" in out, out)

    class _Broken(_FakeClient):
        def web_search(self, query, max_results=3):
            raise RuntimeError("unauthorized (status code: 401)")

    restore = _patch(lit, _Broken())
    try:
        none_out = lit.keyless_web_search("anything", num_results=1)
        page = lit.extract_url_content("https://example.org/x")
    finally:
        restore()
    _check("Ollama failing is named in the diagnostic", "ollama-web-search: unauthorized" in none_out, none_out)
    _check("and every provider is listed", "duckduckgo-html: no results" in none_out, none_out)

    restore = _patch(lit, None)
    try:
        page = lit.extract_url_content("https://example.org/x")
    finally:
        restore()
    _check("no key: pages are downloaded directly", page == "Scraped text of https://example.org/x", page)


def test_tool_description() -> None:
    print("\n[6] the agent is told what search returns")
    from biomentis.tool.tool_description.literature import description

    desc = {d["name"]: d["description"] for d in description}
    _check("search mentions Ollama and extract_url_content",
           "Ollama" in desc["search_google"] and "extract_url_content" in desc["search_google"], desc["search_google"])


def run_all() -> int:
    print("=" * 60)
    print("Web search tests")
    print("=" * 60)
    for test in (test_ollama_first, test_extract_reuses_and_fetches, test_fallbacks, test_tool_description):
        try:
            test()
        except Exception as e:
            _check(f"{test.__name__} ran", False, repr(e))

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASSED)} passed, {len(_FAILED)} failed")
    print("=" * 60)
    for name, detail in _FAILED:
        print(f"  ✗ {name}: {detail}")
    return 0 if not _FAILED else 1


if __name__ == "__main__":
    sys.exit(run_all())
