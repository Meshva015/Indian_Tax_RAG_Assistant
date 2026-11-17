# Indian Tax RAG Assistant

A local Retrieval-Augmented Generation (RAG) learning project for general information
about Indian Income Tax and GST, using official-source documents, Chroma, BGE embeddings,
LangChain, and Groq (`openai/gpt-oss-120b`).

## Scope and safety

- Intended for general information about Indian Income Tax and GST.
- It does not calculate personal tax liability or provide personal tax/legal advice.
- Case law, state taxes, customs duty, and topics absent from the downloaded corpus are out of scope.
- Answers must be checked against the cited official source. This project is not a substitute
  for professional advice.
- The 2018 GST FAQ is historical. Check the cited date and current official notifications
  before relying on time-sensitive rates or rules.

## Project structure

```text
Indian_Tax_RAG_Assistant/
├── sources.yaml
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── clean/
│   └── chroma/
├── src/
│   ├── scrape.py
│   ├── parse.py
│   ├── chunk.py
│   ├── index.py
│   ├── retrieve.py
│   ├── chain.py
│   └── app.py
└── eval/
    ├── golden.csv
    └── run_eval.py
```

## 1. Environment

Use Python 3.11 or 3.12. From the project root in PowerShell:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root (do not upload or commit it):

```text
GROQ_API_KEY=your_groq_api_key_here
```

## 2. Run the pipeline

Run each command from the project root:

```powershell
python -m src.scrape
python -m src.parse
python -m src.chunk
python -m src.index --rebuild
python -m src.app
```

Ask one question from the command line:

```powershell
python -m src.app --question "What does the GST FAQ say about registration?"
```

Use similarity retrieval instead of MMR:

```powershell
python -m src.app --question "Explain GST registration" --strategy similarity --k 5
```

Filter by tax domain:

```powershell
python -m src.app --question "Explain GST registration" --tax-domain gst
```

## 3. Evaluation

```powershell
python -m eval.run_eval
```

This creates `eval/results.csv`. The script reports source hit@6 and MRR where an
expected source URL is supplied. Review answers manually for key-fact coverage,
citation correctness, refusal quality, and latency. Do not claim a target score
until you have run the evaluation on your actual corpus.

## 4. Important implementation notes

- The scraper checks `robots.txt`, uses a descriptive User-Agent, waits between
  requests, retries temporary HTTP failures, and caches unchanged content by hash.
- If `robots.txt` cannot be read or returns a non-200 response, the source is skipped.
  Do not bypass the website's access rules.
- The scraper follows one level of same-domain HTML/PDF links from configured hub pages.
  Discovered pages may have blank metadata; add a source entry in `sources.yaml`
  when you can verify its document type, Act, tax domain, and date.
- The parser keeps PDF page offsets. HTML sources do not currently have page numbers.
- Chunking uses heading-aware splitting for Act/rule documents, question-oriented
  splitting for FAQs, and 1,000-character / 150-overlap splitting for general pages.
- Embeddings use `BAAI/bge-small-en-v1.5` on CPU and are normalized.
- Rebuild the local vector database with `python -m src.index --rebuild`.

## 5. Known limitations to resolve before final acceptance

1. The configured Income Tax Department URLs may be blocked by their robots policy
   or may return access errors. The scraper reports and skips them; it does not bypass.
2. The corpus needs the official Income Tax Act 1961 / transition FAQ and at least
   three verified P1 sources from the internship handbook before Act-transition
   answers can be considered complete.
3. `sources.yaml` currently includes the supplied SVLDRS (2019) hub. SVLDRS is a
   legacy dispute-resolution scheme, not ordinary GST guidance. Keep it only if
   your mentor confirms it belongs in the project's scope.
4. The starter evaluation set contains 30 questions, but several questions depend
   on sources that may not yet be ingested. Expected-source URLs are intentionally
   blank when the supplied source list does not identify a verified source.
5. The evaluation script measures retrieval source hits. It does not automatically
   grade semantic correctness, citation correctness, refusal quality, or all six
   configuration experiments required by the handbook. Those results must be
   measured and recorded after the complete official corpus is available.
6. The Act router is a conservative starter heuristic. Verify the assessment-year
   transition rule from the official transition FAQ before relying on automatic
   Act selection. For ambiguous questions, retrieve both Acts where available.

## 6. Troubleshooting

- `ModuleNotFoundError`: activate the project `.venv`, then run
  `python -m pip install -r requirements.txt`.
- Missing Groq key: check that `.env` is in the project root and contains
  `GROQ_API_KEY=...`.
- Missing Chroma database: run chunking and then `python -m src.index --rebuild`.
- No retrieved results: confirm that clean JSON files exist, chunks were created,
  and the correct tax-domain / Act metadata is present.
