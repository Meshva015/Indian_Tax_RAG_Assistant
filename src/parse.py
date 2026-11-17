
import json
import math
import re
from collections import Counter
from pathlib import Path

import pymupdf
import trafilatura
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
SOURCES_FILE = PROJECT_DIR / "sources.yaml"


def load_sources():
    with SOURCES_FILE.open("r", encoding="utf-8") as file:
        sources = yaml.safe_load(file) or []

    return {item["url"]: item for item in sources}


def normalize_text(text):
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    return text.strip()


def decode_pdf_codes(text):
    """Convert encoded PDF character codes such as /c65 into text."""

    def replace_code(match):
        number = int(match.group(1))

        if 0 <= number <= 255:
            return bytes([number]).decode("cp1252", errors="replace")

        return chr(number)

    return re.sub(r"/c(\d+)", replace_code, text or "")


def remove_maintenance_notices(text):
    phrases = [
        "under maintenance",
        "maintenance activity",
        "technical difficulties",
        "temporarily unavailable",
        "aces archives",
        "aces.gov.in",
        "for all transactions under central excise",
        "use the tab",
    ]

    output = []

    for line in (text or "").splitlines():
        lower = line.lower().strip()

        if lower == "attention":
            continue

        if any(phrase in lower for phrase in phrases):
            continue

        output.append(line)

    return "\n".join(output)


def extract_html(raw_path):
    html = raw_path.read_bytes()
    soup = BeautifulSoup(html, "lxml")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else raw_path.stem
    )

    for element in soup.find_all(
        [
            "nav",
            "footer",
            "header",
            "aside",
            "script",
            "style",
            "noscript",
            "form",
        ]
    ):
        element.decompose()

    main_content = soup.select_one(".gst-registration-faqs")

    if main_content:
        extracted = main_content.get_text(
            separator="\n",
            strip=True,
        )
    else:
        extracted = trafilatura.extract(
            str(soup),
            output_format="txt",
            include_comments=False,
            include_tables=True,
        )

        if not extracted or len(extracted.strip()) < 100:
            extracted = soup.get_text(separator="\n")

    extracted = normalize_text(
        remove_maintenance_notices(extracted)
    )

    return title, extracted, []


def extract_pdf(raw_path):
    pages = []

    # Special extraction for the 1961 Act PDF
    if raw_path.name == "income_tax_act_1961.pdf":
        reader = PdfReader(str(raw_path))
        title = "Income-tax Act, 1961"

        for page_number, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            extracted = decode_pdf_codes(extracted)

            pages.append({
                "page": page_number,
                "text": normalize_text(extracted),
            })

    else:
        document = pymupdf.open(raw_path)
        title = document.metadata.get("title") or raw_path.stem

        for page_number, page in enumerate(document, start=1):
            pages.append({
                "page": page_number,
                "text": normalize_text(page.get_text("text")),
            })

        document.close()

    # Remove repeated headers and footers
    line_counts = Counter()

    for page in pages:
        lines = page["text"].splitlines()

        for line in lines[:3] + lines[-3:]:
            if line.strip():
                line_counts[line.strip()] += 1

    threshold = max(2, math.ceil(len(pages) * 0.5))

    repeated = {
        line
        for line, count in line_counts.items()
        if count >= threshold
    }

    all_text = []
    offsets = []
    position = 0

    for page in pages:
        cleaned = [
            line
            for line in page["text"].splitlines()
            if line.strip() not in repeated
        ]

        page_text = normalize_text("\n".join(cleaned))

        if not page_text:
            continue

        if all_text:
            position += 2

        start = position

        all_text.append(page_text)

        position += len(page_text)

        offsets.append({
            "page": page["page"],
            "start": start,
            "end": position,
        })

    return title, "\n\n".join(all_text), offsets


def parse_sources():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    source_metadata = load_sources()
    sidecars = sorted(RAW_DIR.glob("*.json"))

    # Add metadata for manually downloaded transition FAQ PDF
    transition_url = (
        "https://www.incometaxindia.gov.in/documents/20117/43120/"
        "Updated-FQAs-on-Interplay%26Transitions.pdf"
    )

    transition_pdf = RAW_DIR / "transition_faq.pdf"

    if transition_pdf.exists():
        transition_metadata = {
            "url": transition_url,
            "final_url": transition_url,
            "status": "manual_download",
            "content_type": "application/pdf",
            "doc_type": "faq",
            "act": "IT-2025",
            "tax_domain": "income_tax",
            "as_of": "",
        }

        transition_sidecar = RAW_DIR / "transition_faq.json"

        transition_sidecar.write_text(
            json.dumps(transition_metadata, indent=2),
            encoding="utf-8",
        )

        if transition_sidecar not in sidecars:
            sidecars.append(transition_sidecar)

    print(f"Metadata files found: {len(sidecars)}")

    for sidecar in sidecars:
        try:
            download = json.loads(
                sidecar.read_text(encoding="utf-8-sig")
            )

            url = download["url"]
            file_hash = sidecar.stem

            html_path = RAW_DIR / f"{file_hash}.html"
            pdf_path = RAW_DIR / f"{file_hash}.pdf"

            if sidecar.name == "transition_faq.json":
                raw_path = transition_pdf

            elif html_path.exists():
                raw_path = html_path

            elif pdf_path.exists():
                raw_path = pdf_path

            else:
                print(f"Skipped: raw file not found for {url}")
                continue

            if raw_path.suffix.lower() == ".html":
                title, content, offsets = extract_html(raw_path)

            elif raw_path.suffix.lower() == ".pdf":
                title, content, offsets = extract_pdf(raw_path)

            else:
                print(f"Skipped: unsupported file type: {raw_path.name}")
                continue

            if len(content) < 100:
                print(
                    f"Skipped: only {len(content)} characters: {url}"
                )
                continue

            seed = source_metadata.get(url, {})

            metadata = {}

            for key in ("doc_type", "act", "tax_domain", "as_of"):
                value = download.get(key) or seed.get(key) or ""

                metadata[key] = (
                    value.isoformat()
                    if hasattr(value, "isoformat")
                    else value
                )

            clean_document = {
                "text": content,
                "title": title,
                "page_offsets": offsets,
                "metadata": {
                    "url": url,
                    "final_url": download.get("final_url", ""),
                    "status": download.get("status", ""),
                    "content_type": download.get("content_type", ""),
                    "fetched_at": download.get("fetched_at", ""),
                    **metadata,
                },
            }

            output = CLEAN_DIR / f"{file_hash}.json"

            output.write_text(
                json.dumps(
                    clean_document,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"\nParsed: {raw_path.name}")
            print(f"Characters: {len(content)}")
            print(f"Clean file: {output.name}")
            print(content[:300])
            print("-" * 60)

        except Exception as error:
            print(f"Error processing {sidecar.name}: {error}")

    print(f"\nParsing completed. Clean files: {CLEAN_DIR}")


if __name__ == "__main__":
    parse_sources()
