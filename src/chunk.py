
import hashlib
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
CHUNK_OUTPUT = PROJECT_DIR / "data" / "clean" / "chunks.jsonl"


# General text splitter
GENERAL_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def page_for_offset(offset, page_offsets):
    for item in page_offsets or []:
        if item["start"] <= offset < item["end"]:
            return str(item["page"])
    return ""


def split_faq(text):
    # Detect common FAQ question headings.
    matches = list(re.finditer(
        r"(?im)^(?:Q(?:uestion)?\s*\.?\s*\d+[\).:]?\s*|"
        r"\d+[\).]\s+|(?:Q:)\s*)\S.*$",
        text
    ))

    if len(matches) < 2:
        return GENERAL_SPLITTER.split_text(text)

    pieces = []

    for i, match in enumerate(matches):
        start = match.start()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

    merged = []

    for piece in pieces:
        if merged and len(piece) < 250:
            merged[-1] += "\n\n" + piece
        else:
            merged.append(piece)

    final = []

    for piece in merged:
        if len(piece) > 1500:
            final.extend(GENERAL_SPLITTER.split_text(piece))
        else:
            final.append(piece)

    return final


def split_act_or_rules(text):
    # Find where the schedules begin.
    schedule_match = re.search(
        r"(?im)^\s*SCHEDULE\s+[IVXLCDM0-9]+\b",
        text
    )

    schedule_start = (
        schedule_match.start()
        if schedule_match
        else len(text)
    )

    # Detect section, rule and numbered headings.
    heading_pattern = re.compile(
        r"(?im)^(?=(?:section|sec\.|rule)\s+\d+[A-Za-z]?\b|"
        r"\d{1,3}\.\s)"
    )

    boundaries = []

    for match in heading_pattern.finditer(text):
        heading_text = text[match.start():].lstrip()

        is_explicit_heading = re.match(
            r"(?i)(?:section|sec\.|rule)\s+\d+",
            heading_text
        )

        # Do not split numbered schedule entries.
        if match.start() < schedule_start or is_explicit_heading:
            boundaries.append(match.start())

    parts = []
    start = 0

    for boundary in boundaries:
        if boundary > start:
            part = text[start:boundary].strip()

            if part:
                parts.append(part)

        start = boundary

    remaining = text[start:].strip()

    if remaining:
        parts.append(remaining)

    if len(parts) <= 1:
        return GENERAL_SPLITTER.split_text(text)

    chunks = []

    for part in parts:
        if len(part) > 1500:
            chunks.extend(GENERAL_SPLITTER.split_text(part))
        else:
            chunks.append(part)

    return chunks


def build_chunks():
    records = []

    for path in sorted(CLEAN_DIR.glob("*.json")):

        if path.name == "chunks.jsonl":
            continue

        try:
            doc = json.loads(
                path.read_text(encoding="utf-8")
            )

        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipping {path.name}: {error}")
            continue

        text = (doc.get("text") or "").strip()

        if len(text) < 500:
            print(
                f"Skipping short source ({len(text)} chars): "
                f"{path.name}"
            )
            continue

        meta = doc.get("metadata") or {}

        doc_type = str(meta.get("doc_type") or "")
        title = str(doc.get("title") or path.stem)

        # Split according to document type.
        if doc_type in {"act", "rules"}:
            pieces = split_act_or_rules(text)

        elif doc_type == "faq":
            pieces = split_faq(text)

        else:
            pieces = GENERAL_SPLITTER.split_text(text)

        # Find the first schedule in the original document.
        schedule_match = re.search(
            r"(?im)^\s*SCHEDULE\s+[IVXLCDM0-9]+\b",
            text
        )

        schedule_start = (
            schedule_match.start()
            if schedule_match
            else len(text)
        )

        search_from = 0

        for index, piece in enumerate(pieces):

            if not piece.strip():
                continue

            # Find the original position of this chunk.
            found_at = text.find(
                piece[:100],
                search_from
            )

            if found_at < 0:
                found_at = text.find(piece[:100])

            if found_at >= 0:
                search_from = found_at + len(piece[:100])

            # Detect section metadata only before schedules.
            section_match = None

            if found_at >= 0 and found_at < schedule_start:
                section_match = re.search(
                    r"(?im)^\s*((?:section|sec\.|rule)\s+\d+[A-Za-z]?\b|"
                    r"\d+(?=\.\s*(?:\(\d+\)|In this Act\b|"
                    r"[A-Z][^\n]{0,120}—)))",
                    piece
                )

            section_ref = (
                section_match.group(1).strip()
                if section_match
                else ""
            )

            if section_ref.isdigit():
                section_ref = f"section {section_ref}"

            # Find the PDF page number.
            page = (
                page_for_offset(
                    found_at,
                    doc.get("page_offsets", [])
                )
                if found_at >= 0
                else ""
            )

            # Create a unique chunk ID.
            chunk_id = hashlib.sha1(
                f"{meta.get('url', '')}{index}".encode("utf-8")
            ).hexdigest()

            prefix = title

            if section_ref:
                prefix += f" > {section_ref}"

            chunk_text = f"{prefix}\n\n{piece}"

            records.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source_url": str(meta.get("url") or ""),
                    "title": title,
                    "doc_type": doc_type,
                    "act": str(meta.get("act") or ""),
                    "tax_domain": str(meta.get("tax_domain") or ""),
                    "section_ref": section_ref,
                    "as_of": str(meta.get("as_of") or ""),
                    "page": page,
                    "fetched_at": str(meta.get("fetched_at") or ""),
                    "chunk_id": chunk_id,
                },
            })

    # Save chunks.
    CHUNK_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with CHUNK_OUTPUT.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Chunks created: {len(records)}")
    print(f"Chunk file: {CHUNK_OUTPUT}")

    return records


if __name__ == "__main__":
    build_chunks()
