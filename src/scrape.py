import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
import yaml
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

USER_AGENT = "IndianTaxRAGAssistant/1.0 (contact: hmeshva@gmail.com)"
DELAY_SECONDS = 1.5
MAX_PAGES = 300
MAX_RETRIES = 4
TIMEOUT = 30

last_request_time = {}
robots_cache = {}


def wait_for_host(url):
    host = urlparse(url).netloc
    now = time.time()
    if host in last_request_time:
        remaining = DELAY_SECONDS - (now - last_request_time[host])
        if remaining > 0:
            time.sleep(remaining)
    last_request_time[host] = time.time()


def get_robots_parser(url):
    host = urlparse(url).netloc
    if host in robots_cache:
        return robots_cache[host]
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{host}/robots.txt"
    try:
        wait_for_host(robots_url)
        response = requests.get(
            robots_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
        )
        if response.status_code != 200:
            print(f"Skipping {host}: robots.txt returned HTTP {response.status_code}")
            robots_cache[host] = None
            return None
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        robots_cache[host] = parser
        return parser
    except requests.RequestException as error:
        print(f"Could not read robots.txt for {host}: {error}")
        robots_cache[host] = None
        return None


def can_fetch_url(url):
    parser = get_robots_parser(url)
    if parser is None:
        return False
    allowed = parser.can_fetch(USER_AGENT, url)
    if not allowed:
        print(f"Blocked by robots.txt: {url}")
    return allowed


def fetch_url(url):
    if not can_fetch_url(url):
        return None
    for attempt in range(MAX_RETRIES):
        try:
            wait_for_host(url)
            response = requests.get(
                url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
            )
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                delay = 2 ** attempt
                print(f"HTTP {response.status_code}; retrying in {delay}s")
                time.sleep(delay)
                continue
            if response.status_code != 200:
                print(f"Skipping {url}: HTTP {response.status_code}")
                return None
            return response
        except requests.RequestException as error:
            print(f"Request failed for {url}: {error}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    print(f"Failed after {MAX_RETRIES} attempts: {url}")
    return None


def get_file_extension(response):
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        return ".pdf"
    if "text/html" in content_type or "application/xhtml" in content_type:
        return ".html"
    print(f"Unsupported content type: {content_type}")
    return None


def save_raw_file(url, response, source_metadata=None):
    source_metadata = source_metadata or {}
    extension = get_file_extension(response)
    if extension is None:
        return False

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    content_hash = hashlib.sha256(response.content).hexdigest()
    raw_path = RAW_DIR / f"{url_hash}{extension}"
    sidecar_path = RAW_DIR / f"{url_hash}.json"

    metadata = {
        "url": url,
        "final_url": response.url,
        "status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash,
        "doc_type": source_metadata.get("doc_type") or "",
        "act": source_metadata.get("act") or "",
        "tax_domain": source_metadata.get("tax_domain") or "",
        "as_of": source_metadata.get("as_of") or "",
    }
    if hasattr(metadata["as_of"], "isoformat"):
        metadata["as_of"] = metadata["as_of"].isoformat()

    old_hash = None
    if sidecar_path.exists():
        try:
            old_hash = json.loads(sidecar_path.read_text(encoding="utf-8")).get(
                "content_hash"
            )
        except (json.JSONDecodeError, OSError):
            pass

    if old_hash == content_hash and raw_path.exists():
        print(f"Unchanged: {url}")
    else:
        raw_path.write_bytes(response.content)
        print(f"Saved: {raw_path.name}")

    sidecar_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return True


def find_same_domain_links(url, html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    host = urlparse(url).netloc
    links = set()
    for tag in soup.find_all("a", href=True):
        link, _ = urldefrag(urljoin(url, tag["href"]))
        parsed = urlparse(link)
        if parsed.scheme in ("http", "https") and parsed.netloc == host:
            if parsed.path.lower().endswith((".html", ".htm", ".pdf")):
                links.add(link)
    return links


def load_sources():
    with SOURCES_FILE.open("r", encoding="utf-8") as file:
        sources = yaml.safe_load(file) or []
    if not isinstance(sources, list):
        raise ValueError("sources.yaml must contain a list.")
    return sources


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    queue = [
        {"url": item["url"], "source_metadata": item}
        for item in load_sources()
    ]
    visited = set()
    page_count = 0

    while queue and page_count < MAX_PAGES:
        current = queue.pop(0)
        url = current["url"]
        if url in visited:
            continue
        visited.add(url)
        page_count += 1
        print(f"\nFetching {page_count}: {url}")

        response = fetch_url(url)
        if response is None:
            continue
        if not save_raw_file(url, response, current.get("source_metadata")):
            continue

        if current.get("source_metadata", {}).get("doc_type") == "hub":
            if get_file_extension(response) == ".html":
                inherited = current.get("source_metadata", {})
                for link in sorted(find_same_domain_links(response.url, response.content)):
                    if link not in visited:
                        # Carry only the parent Act/domain. A discovered page's exact
                        # document type/date must be verified rather than guessed.
                        queue.append({
                            "url": link,
                            "source_metadata": {
                                "doc_type": "web_page",
                                "act": inherited.get("act", ""),
                                "tax_domain": inherited.get("tax_domain", ""),
                                "as_of": "",
                            },
                        })

    print("\nScraping completed.")
    print(f"Pages processed: {page_count}")
    print(f"Raw files saved in: {RAW_DIR}")


if __name__ == "__main__":
    main()
