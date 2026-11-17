
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retrieve import retrieve


# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=2,
)


# System instructions
SYSTEM = """
You are an Indian tax information assistant.

Follow these rules strictly:

1. Answer using ONLY the numbered context provided.
2. Never use your own knowledge to add tax rules or facts.
3. Add a citation like [1] after every factual claim.
4. If the context does not answer the question, say:
   "The indexed sources do not cover this question."
   Also mention the closest available source.
5. Clearly state which Act or tax domain the answer relies on.
6. Check the as_of date. Warn users if a source may be outdated,
   especially for tax rates and thresholds.
7. Do not provide personal tax calculations or personal tax advice.
8. End every answer with:
   "General information, not tax advice."
"""


# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    ),
])


# Format retrieved documents
def format_docs(docs):

    formatted = []

    for i, doc in enumerate(docs, 1):

        metadata = doc.metadata

        block = (
            f"[{i}]\n"
            f"Content: {doc.page_content}\n"
            f"Title: {metadata.get('title', 'Unknown')}\n"
            f"Section: {metadata.get('section_ref', 'Not available')}\n"
            f"Act: {metadata.get('act', 'Not available')}\n"
            f"As of: {metadata.get('as_of', 'Unknown')}\n"
            f"URL: {metadata.get('source_url', 'Not available')}"
        )

        formatted.append(block)

    return "\n\n".join(formatted)


# Create LangChain chain
chain = prompt | llm | StrOutputParser()


# Extract citations from the answer
def extract_citations(answer, docs):

    citation_numbers = set(
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    )

    sources = []

    for number in sorted(citation_numbers):

        if 1 <= number <= len(docs):

            metadata = docs[number - 1].metadata

            sources.append({
                "citation": f"[{number}]",
                "title": metadata.get("title", ""),
                "section": metadata.get("section_ref", ""),
                "act": metadata.get("act", ""),
                "as_of": metadata.get("as_of", ""),
                "url": metadata.get("source_url", ""),
                "page": metadata.get("page", ""),
            })

    return sources


# Main question-answering function
def answer_question(
    question,
    k=6,
    tax_domain=None,
    act=None
):

    # Step 1: Retrieve relevant documents
    docs = retrieve(
        question,
        k=k,
        strategy="similarity",
        tax_domain=tax_domain,
        act=act,
    )

    if not docs:
        return {
            "answer": (
                "The indexed sources do not cover this question.\n\n"
                "General information, not tax advice."
            ),
            "sources": [],
        }

    # Step 2: Format documents
    context = format_docs(docs)

    # Step 3: Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    # Step 4: Extract cited sources
    sources = extract_citations(answer, docs)

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":

    question = (
        "What is the commencement date of the "
        "Income-tax Act, 2025?"
    )

    result = answer_question(
        question,
        k=5,
        act="IT-2025",
    )

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nCITED SOURCES:\n")

    for source in result["sources"]:
        print(source)
