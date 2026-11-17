
import json
from pathlib import Path
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "clean"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"

# Chroma collection
COLLECTION_NAME = "india_tax"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Chroma batch size
BATCH_SIZE = 5000


def load_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_chunks():
    """Load documents and IDs from chunks.jsonl."""
    chunk_file = CLEAN_DIR / "chunks.jsonl"

    documents = []
    ids = []

    with chunk_file.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            documents.append(
                Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
            )

            ids.append(item["id"])

    return documents, ids


def add_in_batches(vectorstore, documents, ids):
    """Add documents to Chroma in smaller batches."""
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE

        vectorstore.add_documents(
            documents=documents[start:end],
            ids=ids[start:end]
        )

        print(
            f"Indexed batch: {start + 1} to "
            f"{min(end, len(documents))}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    documents, ids = load_chunks()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=load_embeddings(),
    )

    if args.rebuild:
        add_in_batches(vectorstore, documents, ids)

    else:
        existing = set(
            vectorstore.get(include=[])["ids"]
        )

        new_docs = []
        new_ids = []

        for doc, chunk_id in zip(documents, ids):
            if chunk_id not in existing:
                new_docs.append(doc)
                new_ids.append(chunk_id)

        if new_docs:
            add_in_batches(vectorstore, new_docs, new_ids)

    # Display indexing statistics
    counts = Counter(
        doc.metadata.get("act", "")
        for doc in documents
    )

    types = Counter(
        doc.metadata.get("doc_type", "")
        for doc in documents
    )

    print(f"\nIndexed chunks: {len(documents)}")
    print(f"Chunks by act: {dict(counts)}")
    print(f"Chunks by document type: {dict(types)}")
    print("\nIndexing completed successfully!")


if __name__ == "__main__":
    main()
