
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma"

# Chroma collection
COLLECTION_NAME = "india_tax"

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# Load vector database
def get_vectorstore():

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise FileNotFoundError(
            "Vector database not found. "
            "Run python -m src.index --rebuild."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


# Act Router
def choose_acts(question):

    q = question.lower()

    # GST questions
    gst_markers = [
        "gst",
        "goods and services tax",
        "input tax credit",
        "composition scheme",
    ]

    if any(term in q for term in gst_markers):
        return ["GST", "SVLDRS-2019"]

    # Transition questions
    transition_markers = [
        "transition",
        "interplay",
        "which act applies",
        "old act or new act",
    ]

    if any(term in q for term in transition_markers):
        return ["IT-1961", "IT-2025"]

    # Explicit Act references
    if "1961" in q or "old act" in q:
        return ["IT-1961"]

    if "2025" in q or "new act" in q:
        return ["IT-2025"]

    # Assessment Year
    if "assessment year" in q or " ay " in f" {q} ":
        return ["IT-1961"]

    # Tax Year
    if "tax year" in q:
        return ["IT-2025"]

    # Old Act section references
    if any(term in q for term in ["section 80c", "section 192", "80c"]):
        return ["IT-1961"]

    # Ambiguous Income Tax questions
    return ["IT-1961", "IT-2025"]


# Retrieve relevant documents
def retrieve(
    question,
    k=5,
    strategy="similarity",
    tax_domain=None,
    act=None,
):

    store = get_vectorstore()

    # Select Acts
    selected_acts = [act] if act else choose_acts(question)

    # Build metadata filters
    filters = []

    if tax_domain:
        filters.append({"tax_domain": tax_domain})

    if selected_acts:
        filters.append({"act": {"$in": selected_acts}})

    base_filter = None

    if len(filters) == 1:
        base_filter = filters[0]

    elif len(filters) > 1:
        base_filter = {"$and": filters}

    # Detect section number
    q = question.lower()

    section_match = re.search(
        r"\bsection\s+(\d+[a-z]?)\b",
        q,
    )

    if section_match:
        section_number = section_match.group(1)

    elif "short title" in q or "commencement" in q:
        section_number = "1"

    else:
        section_number = None

    # Search exact section first
    if section_number:

        section_filter = {
            "section_ref": f"section {section_number}"
        }

        if base_filter:
            section_filter = {
                "$and": [
                    base_filter,
                    section_filter,
                ]
            }

        docs = store.similarity_search(
            question,
            k=k,
            filter=section_filter,
        )

        if docs:
            return docs

    # Normal similarity or MMR search
    search_kwargs = {
        "k": k,
    }

    if base_filter:
        search_kwargs["filter"] = base_filter

    if strategy.lower() == "mmr":

        search_kwargs.update({
            "fetch_k": 25,
            "lambda_mult": 0.5,
        })

        retriever = store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    else:

        retriever = store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    return retriever.invoke(question)


# Test Act Router
def test_act_router():

    test_cases = [

        (
            "What is Assessment Year?",
            ["IT-1961"],
        ),

        (
            "What is Tax Year?",
            ["IT-2025"],
        ),

        (
            "Explain Section 80C",
            ["IT-1961"],
        ),

        (
            "Explain the Income-tax Act, 2025",
            ["IT-2025"],
        ),

        (
            "What is GST input tax credit?",
            ["GST", "SVLDRS-2019"],
        ),

        (
            "Which Act applies during the transition?",
            ["IT-1961", "IT-2025"],
        ),

    ]

    passed = 0

    for question, expected in test_cases:

        actual = choose_acts(question)

        if actual == expected:

            print(f"PASS: {question}")
            passed += 1

        else:

            print(f"FAIL: {question}")
            print("Expected:", expected)
            print("Actual:", actual)

    print(
        f"\nRouter Tests: {passed}/{len(test_cases)} passed"
    )


# Run tests
if __name__ == "__main__":
    test_act_router()
