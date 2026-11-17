import csv
import time
from pathlib import Path

from src.chain import answer_question
from src.retrieve import retrieve

PROJECT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_FILE = PROJECT_DIR / "eval" / "golden.csv"
RESULTS_FILE = PROJECT_DIR / "eval" / "results.csv"


def main():
    rows = list(csv.DictReader(GOLDEN_FILE.open("r", encoding="utf-8-sig")))
    results = []
    answerable = [row for row in rows if row["should_answer"] == "1"]
    hits, reciprocal_ranks = 0, []

    for row in rows:
        question = row["question"]
        start = time.perf_counter()
        try:
            docs = retrieve(question, k=6, strategy="mmr")
            elapsed = time.perf_counter() - start
            expected = row["expected_source_url"]
            urls = [doc.metadata.get("source_url", "") for doc in docs]
            hit = int(bool(expected) and expected in urls)
            if hit:
                rank = urls.index(expected) + 1
                reciprocal_ranks.append(1 / rank)
            elif row["should_answer"] == "1":
                reciprocal_ranks.append(0)
            hits += hit
            response = answer_question(question, k=6, strategy="mmr")
            results.append({
                "question": question,
                "category": row["category"],
                "hit_at_6": hit,
                "latency_seconds_retrieval": round(elapsed, 3),
                "retrieved_urls": " | ".join(urls),
                "response": response.replace("\n", " ")[:4000],
            })
            print(f"{row['category']}: hit@6={hit} | {question}")
        except Exception as error:
            results.append({
                "question": question, "category": row["category"],
                "hit_at_6": 0, "latency_seconds_retrieval": "",
                "retrieved_urls": "", "response": f"ERROR: {error}",
            })

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "question", "category", "hit_at_6",
            "latency_seconds_retrieval", "retrieved_urls", "response"
        ])
        writer.writeheader()
        writer.writerows(results)

    answerable_hits = sum(
        int(item["hit_at_6"]) for item in results
        if item["category"] not in {"out_of_scope", "unanswerable"}
    )
    print("\nEvaluation completed.")
    print(f"Questions: {len(rows)}")
    print(f"Answerable questions: {len(answerable)}")
    print(f"Answerable hit@6: {answerable_hits}/{len(answerable)}")
    print(f"MRR (expected URLs supplied): {sum(reciprocal_ranks)/len(reciprocal_ranks):.3f}"
          if reciprocal_ranks else "MRR: not available")
    print(f"Results saved to: {RESULTS_FILE}")
    print("Note: key-fact coverage, citation correctness, refusal quality, and full experiment comparisons require manual review.")


if __name__ == "__main__":
    main()
