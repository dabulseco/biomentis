import os
import re
import time
from io import BytesIO
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import PyPDF2
import requests
from bs4 import BeautifulSoup

# `googlesearch` is intentionally NOT imported at module level. The original
# `googlesearch` PyPI package was renamed to `googlesearch-python` in 2024; if
# a user installs the package under its new name the import still works, but
# if they have neither installed, a module-level `from googlesearch import search`
# breaks the entire literature module — taking down query_pubmed, query_arxiv,
# and advanced_web_search_claude with it. The lazy import below only fires when
# that provider is actually reached, and it is no longer the only provider —
# see `keyless_web_search` for the DuckDuckGo backends that need no extra
# package and no API key at all.


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # Fetch publisher page
    response = requests.get(publisher_url, headers=headers)
    if response.status_code != 200:
        log_message = f"Failed to access publisher page for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Parse page content
    soup = BeautifulSoup(response.content, "html.parser")
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return research_log

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for link in supplementary_links:
        file_name = os.path.join(output_dir, link.split("/")[-1])
        file_response = requests.get(link, headers=headers)
        if file_response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(file_response.content)
            downloaded_files.append(file_name)
            research_log.append(f"Downloaded file: {file_name}")
        else:
            research_log.append(f"Failed to download file from {link}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return "\n".join(research_log)


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import arxiv

    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
        results = "\n\n".join([f"Title: {paper.title}\nSummary: {paper.summary}" for paper in client.results(search)])
        return results if results else "No papers found on arXiv."
    except Exception as e:
        return f"Error querying arXiv: {e}"


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

    """
    from scholarly import ProxyGenerator, scholarly

    # Set up a ProxyGenerator object to use free proxies
    # This needs to be done only once per session
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)
    try:
        search_query = scholarly.search_pubs(query)
        result = next(search_query, None)
        if result:
            return f"Title: {result['bib']['title']}\nYear: {result['bib']['pub_year']}\nVenue: {result['bib']['venue']}\nAbstract: {result['bib']['abstract']}"
        else:
            return "No results found on Google Scholar."
    except Exception as e:
        return f"Error querying Google Scholar: {e}"


def _get_ncbi_email() -> str:
    """Resolve the user's NCBI email from config / env. NCBI requires one
    (politely) on every Entrez request and will rate-limit requests that omit it."""
    # Lazy import to keep this module cheap to load
    try:
        from biomentis.config import default_config

        if default_config.ncbi_email:
            return default_config.ncbi_email
    except Exception:
        pass
    return os.getenv("NCBI_EMAIL") or os.getenv("BIOMNI_NCBI_EMAIL") or "your-email@example.com"


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """Query PubMed for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retry attempts with modified queries (default: 3).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    try:
        from pymed import PubMed
    except ImportError:
        return (
            "query_pubmed is unavailable: the 'pymed' package is not installed. "
            "Install it with `pip install pymed==0.8.9` (or `pip install -r "
            "requirements.txt`) and retry. The rest of the literature tools "
            "still work without it."
        )

    try:
        pubmed = PubMed(tool="Biomentis", email=_get_ncbi_email())

        # Initial attempt
        papers = list(pubmed.query(query, max_results=max_papers))

        # Retry with modified queries if no results
        retries = 0
        while not papers and retries < max_retries:
            retries += 1
            # Simplify query with each retry by removing the last word
            simplified_query = " ".join(query.split()[:-retries]) if len(query.split()) > retries else query
            time.sleep(1)  # Add delay between requests
            papers = list(pubmed.query(simplified_query, max_results=max_papers))

        if papers:
            results = "\n\n".join(
                [f"Title: {paper.title}\nAbstract: {paper.abstract}\nJournal: {paper.journal}" for paper in papers]
            )
            return results
        else:
            return "No papers found on PubMed after multiple query attempts."
    except Exception as e:
        return f"Error querying PubMed: {e}"


# ---------------------------------------------------------------------------
# Keyless web search backends
# ---------------------------------------------------------------------------
# Everything below runs with *no* API key and no account: DuckDuckGo's public
# HTML endpoints are scraped with requests + BeautifulSoup (both already hard
# dependencies of this module), with googlesearch-python as a last resort.
#
# Why this exists: `advanced_web_search_claude` used to hard-fail with
# "Error code: 401 - invalid x-api-key" whenever ANTHROPIC_API_KEY was set but
# stale/wrong, and the agent would happily keep reasoning on that error string.
# Now any Anthropic failure -- and an unset key -- degrades to these providers.
#
# Set BIOMENTIS_WEB_SEARCH=keyless to skip Anthropic entirely.

_KEYLESS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo wraps results in /l/?uddg=<encoded>; unwrap to the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href or href.startswith("/l/"):
        target = parse_qs(urlparse(href).query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def _search_duckduckgo_html(query: str, num_results: int, timeout: int = 20) -> list[dict]:
    """Scrape https://html.duckduckgo.com/html/ -- no key, no rate-limit token."""
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query, "kl": "us-en"},
        headers=_KEYLESS_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for node in soup.select("div.result, div.web-result"):
        link = node.select_one("a.result__a")
        if link is None:
            continue
        url = _clean_ddg_url(link.get("href", ""))
        if not url:
            continue
        snippet = node.select_one(".result__snippet")
        results.append(
            {
                "title": link.get_text(" ", strip=True),
                "url": url,
                "description": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= num_results:
            break
    return results


def _search_duckduckgo_lite(query: str, num_results: int, timeout: int = 20) -> list[dict]:
    """Scrape https://lite.duckduckgo.com/lite/ -- different markup, same data.

    Kept as a second provider because the two endpoints are rate-limited
    independently; when the HTML one starts returning an anomaly page this one
    usually still answers.
    """
    resp = requests.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        headers=_KEYLESS_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results, pending = [], None
    for row in soup.select("tr"):
        link = row.select_one("a.result-link")
        if link is not None:
            if pending is not None:
                results.append(pending)
            pending = {
                "title": link.get_text(" ", strip=True),
                "url": _clean_ddg_url(link.get("href", "")),
                "description": "",
            }
        elif pending is not None:
            snippet = row.select_one("td.result-snippet")
            if snippet is not None:
                pending["description"] = snippet.get_text(" ", strip=True)
        if len(results) >= num_results:
            break
    if pending is not None and len(results) < num_results:
        results.append(pending)
    return [r for r in results if r["url"]][:num_results]


def _search_googlesearch_python(query: str, num_results: int, language: str = "en") -> list[dict]:
    """The original googlesearch-python path, now one provider among several."""
    # Lazy import: optional dependency, only needed when the DDG providers fail.
    from googlesearch import search as _google_search

    return [
        {
            "title": res.title,
            "url": res.url,
            "description": res.description,
        }
        for res in _google_search(query, num_results=num_results, lang=language, advanced=True)
    ]


def _format_search_results(results: list[dict]) -> str:
    """Render provider hits in the shape the agent has always seen."""
    return "".join(
        f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nDescription: {r.get('description', '')}\n\n"
        for r in results
    )


def keyless_web_search(query: str, num_results: int = 5, language: str = "en") -> str:
    """Search the web without any API key.

    Tries DuckDuckGo's HTML endpoint, then its Lite endpoint, then
    googlesearch-python, returning the first provider that yields hits.

    Args:
        query: The search query.
        num_results: Number of results to return.
        language: Language code (used by the googlesearch provider).

    Returns:
        Formatted "Title / URL / Description" blocks, or a diagnostic string
        if every provider failed.

    """
    providers = (
        ("duckduckgo-html", lambda: _search_duckduckgo_html(query, num_results)),
        ("duckduckgo-lite", lambda: _search_duckduckgo_lite(query, num_results)),
        ("googlesearch-python", lambda: _search_googlesearch_python(query, num_results, language)),
    )

    failures = []
    for name, run in providers:
        try:
            results = run()
        except ImportError:
            failures.append(f"{name}: not installed")
            continue
        except Exception as e:  # network error, blocked scrape, markup change
            failures.append(f"{name}: {e}")
            continue
        if results:
            print(f"[keyless_web_search] {len(results)} result(s) from {name} for: {query}")
            return _format_search_results(results)
        failures.append(f"{name}: no results")

    return "No web results. Search providers tried -- " + "; ".join(failures)


def search_google(query: str, num_results: int = 3, language: str = "en") -> str:
    """Search the web and return formatted results. No API key required.

    Despite the name (kept for backwards compatibility with the tool registry
    and existing prompts), this now runs the keyless provider chain in
    `keyless_web_search`: DuckDuckGo HTML, DuckDuckGo Lite, then
    googlesearch-python. Google alone was too easy to block — a 429 from the
    scraper used to leave the agent with an empty string.

    Args:
        query (str): The search query (e.g., "protocol text or search question")
        num_results (int): Number of results to return (default: 3)
        language (str): Language code for search results (default: 'en')

    Returns:
        str: Formatted "Title / URL / Description" blocks, one per result.

    """
    return keyless_web_search(query, num_results=num_results, language=language)


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> str:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Behavior:
        - If a usable ANTHROPIC_API_KEY is set, this tool uses Anthropic's
          server-side web_search tool (the original behavior).
        - Otherwise — key unset, key rejected (401/403), `anthropic` not
          installed, or BIOMENTIS_WEB_SEARCH=keyless — it transparently falls
          back to `keyless_web_search`, which scrapes DuckDuckGo and needs no
          API key at all. The LLM sees a string of the same shape, so the ReAct
          loop doesn't have to change.

    This function no longer returns an "Error code: 401" string to the agent:
    a broken key degrades the search instead of poisoning the reasoning trace.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
    """
    import random

    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Explicit opt-out: never touch a paid API for web search.
    if os.getenv("BIOMENTIS_WEB_SEARCH", "").strip().lower() in {"keyless", "free", "local"}:
        return _advanced_web_search_fallback(query, reason="BIOMENTIS_WEB_SEARCH=keyless")

    # Auto-fallback path: no Anthropic key → use the keyless providers so the
    # agent still gets web results on the Ollama default path.
    if not api_key:
        return _advanced_web_search_fallback(query, reason="ANTHROPIC_API_KEY not set")

    # The "use the key for this tool even when chat is on Ollama" path is
    # captured in reminder_consider.md — not implemented yet. When it is,
    # this is where the cost guard will live.
    try:
        import anthropic
    except ImportError:
        return _advanced_web_search_fallback(query, reason="the 'anthropic' package is not installed")

    try:
        from biomentis.config import default_config

        model = default_config.llm
    except ImportError:
        model = "claude-4-sonnet-latest"

    client = anthropic.Anthropic(api_key=api_key)
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            # A bad/expired key, a key without web_search entitlement, or a
            # model the key can't reach will fail identically on every retry —
            # burning ~15s of backoff to arrive at the same 401. Bail out to
            # the keyless path immediately instead.
            if _is_non_retryable_anthropic_error(e):
                return _advanced_web_search_fallback(query, reason=f"Anthropic rejected the request ({e})")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"Error performing web search after {max_retries} attempts: {e}")
            return _advanced_web_search_fallback(query, reason=f"Anthropic web_search failed {max_retries}x ({e})")


def _is_non_retryable_anthropic_error(exc: Exception) -> bool:
    """True when retrying the Anthropic call cannot possibly change the outcome.

    Covers authentication (401), permission (403) and not-found (404, e.g. a
    model the key can't reach) — as opposed to 429/5xx/timeouts, which are
    worth the exponential backoff above.
    """
    status = getattr(exc, "status_code", None)
    if status in (401, 403, 404):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "authentication_error",
            "invalid x-api-key",
            "permission_error",
            "error code: 401",
            "error code: 403",
        )
    )


def _advanced_web_search_fallback(query: str, num_results: int = 5, reason: str = "") -> str:
    """Fallback for `advanced_web_search_claude` when Anthropic is unavailable.

    Routes to `keyless_web_search` (free, no API key) and prepends a short note
    so the LLM can see that this is a degraded result, not a Claude-grade
    multi-round search. The returned string keeps the same shape as the
    Anthropic path so the ReAct loop doesn't need to special-case it.

    Future idea (see reminder_consider.md): extend this to also try
    `query_pubmed` for biomedical queries, and respect a per-tool
    fallback-policy config.
    """
    reason = reason or "Anthropic web_search unavailable"
    note = (
        f"[advanced_web_search_claude: {reason} — falling back to a keyless "
        f"DuckDuckGo search (top {num_results} results). Results are plain "
        "search hits, not a multi-round agentic search; use extract_url_content "
        "on the most promising URLs to read further.]\n\n"
    )
    print(f"[advanced_web_search_claude] {reason} — using keyless web search.")
    try:
        results = keyless_web_search(query, num_results=num_results, language="en")
    except Exception as e:
        return note + f"keyless_web_search failed: {e}"
    if not results:
        return note + "No web results returned."
    return note + results


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
