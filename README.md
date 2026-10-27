# Club Docs Q&A

A small retrieval-augmented question-answering app over 20 club documents (FAQs, handbook, policies, committee notes). Built from scratch with no RAG framework. Answers come only from the documents, and when the documents don't contain the answer, the app says so.

**Live app:** https://club-docs-rag-bdiqwj59dhm5h7rncrszqk.streamlit.app/

## At a glance

- **What:** retrieval-augmented Q&A over 20 club documents, built without a RAG framework (chromadb + embeddings + a validated JSON answer).
- **Behaviour:** answers only from the documents and cites its source. It says "I don't know" when the answer isn't there, and on multi-part questions it answers the covered part and names the missing one.
- **Measured:** 15/15 on a 15-case eval (14/15 before one prompt change) and 10/10 on 10 unseen hold-out questions, on `gemini-3.5-flash-lite`. Every miss in every run was a generation failure, none were retrieval failures.
- **Be sceptical of:** the corpus is small (see Limitations), the eval sets are small and single-run, and no one but the author has used it yet.

## Problem

Club members ask the same questions repeatedly (trial sessions, refunds, weather cancellations, who to contact), and the answers are spread across many files. A plain chatbot would invent plausible answers. The goal here was an assistant that is **grounded** (cites its source) and **honest** (refuses when the answer isn't in the documents).

## Approach

1. **Chunk** each document and **embed** the chunks (`nomic-embed-text` locally, `gemini-embedding-001` when deployed).
2. **Retrieve** the top-k chunks (k=3) from an in-memory chromadb collection.
3. **Prompt** the model to return JSON: `{found, answer, sources}`. For multi-part questions it answers the parts the documents cover and ends with "Not in the documents: ..." for the rest.
4. **Validate in code** before anything reaches the user: correct shape and types, sources must be among the retrieved files, refusals are normalised, and a failed parse gets one retry and then a safe "I don't know" fallback.
5. **UI:** a minimal Streamlit page (input box and answer). The API key lives in Streamlit secrets, not in the code.

The provider is swappable (`RAG_PROVIDER=ollama|gemini`), so the same code runs locally on llama3.2 and deployed on Gemini.

## Evaluation

15 hand-written cases: 8 normal, 2 multi-document, 3 that must be refused, 2 adversarial (prompt injection, an instruction to ignore the context). Each case lists facts the answer must contain and the sources it must cite, and a check confirms those facts really appear in the cited document. Each failure is tagged **RETRIEVAL** (right chunk never retrieved) or **GENERATION** (retrieved, but answered wrongly). Calls that never reached the model are tagged **INFRA** and the run is marked invalid.

| Change (one at a time) | Model | Score |
| --- | --- | --- |
| Baseline | llama3.2 (local) | 11/15 |
| Rewritten prompt | llama3.2 | 11/15 |
| Quote-grounded citations (cited source derived from the quote)* | llama3.2 | 11/15 |
| Added code check for "who is X" questions with no name in the evidence* | llama3.2 | 12/15 |
| Stronger model on the shipped (baseline) validator | gemini-3.5-flash-lite | **14/15** |
| Retrieve k=5 instead of k=3 | gemini-3.5-flash-lite | 13/15 |
| Prompt rule: answer the parts of a multi-part question the documents cover, and name the missing part | gemini-3.5-flash-lite | **15/15** (one run) |
| Same app, **10 unseen hold-out questions** (4 plain, 3 multi-part, 2 must-refuse, 1 false-premise) | gemini-3.5-flash-lite | **10/10** (one run) |

\* **These two experiments are not in this repository.** The code for them was overwritten in my working folder before the first commit, so the repo contains the earlier, simpler validator (shape and type checks, sources must be among the retrieved files, refusal normalisation, one retry). The 11/15 and 12/15 results were measured at the time and are recorded in `eval_results.json`, but you cannot reproduce them from this code. The 14/15 and 13/15 rows below them were run on the code in this repo (the 14/15 was also measured earlier on the Day-24 code, with the same result).

The deployed app uses `gemini-3.5-flash-lite`, the model behind the 14/15 figure (two complete runs, same single failure). I also tried `gemini-3.6-flash`, but the free-tier quota ran out mid-run three times, so no valid score exists for it. Those runs are kept in `eval_results.json`, marked invalid, rather than deleted. The cases that did reach 3.6-flash showed the same case-15 failure.

What the numbers say:

- **Rewording the prompt moved the score by 0 points** across two runs. The one prompt change that did help (+1, case 15) added a missing *capability*, a partial answer, instead of rewording an existing instruction. In the earlier llama experiments, the one fix that worked came from moving judgment out of the prompt and into code (the who-check, +1), though that code is not in this repo (see the note under the table). Switching to a stronger model fixed two more cases with no code changes.
- **Retrieval never failed:** 0 retrieval failures in every run, with two different embedding models. Every miss was in generation.
- **More context did not help:** k=5 scored one point lower than k=3. With retrieval already perfect, extra chunks only added noise. This is one run on 15 cases, so treat it as weak evidence.

## Tradeoffs

- Moving rules into code makes behaviour predictable and testable (9 unit tests on the shipped validator).
- In the earlier quote-grounded experiment (not in this repo), a single required quote proved only one source per answer and cost some recall (the separate JSON drill went from 13/13 to 11/13).
- Hand-written eval cases are slow to write, but they are what made the above findings visible.

## Limitations

- **The corpus is tiny.** There are 20 short documents, each fitting in a single chunk, so chunking strategy is never really exercised and "0 retrieval failures" is expected with top-3 out of 20. These results say little about retrieval quality on a real, larger or messier corpus.
- **The 15/15 is optimistic.** I wrote the partial-answer rule after seeing case 15 fail, so that eval set is no longer an unseen test for that change. To check, I scored 10 hold-out questions (`eval_holdout.py`) that were written and committed before scoring and never used to tune the prompt: 10/10. That is a better signal, but still a single run on a small set, on one model, and the hold-out questions were drafted with an AI assistant from the same documents, so they are not independent of how the documents are worded. The checker also only looks for key facts, not for fully correct wording.
- The eval set is small (15 cases); one case is about 7 percentage points.
- `gemini-3.6-flash` was not scored (free-tier quota), so there is no comparison against it.
- The app runs on a free API tier. Under heavy use it can hit rate or daily limits and will show a "service problem" message rather than a wrong answer.
- The index is rebuilt in memory on each cold start.

## What I'd do next

1. Write harder hold-out questions: paraphrases, questions that need two documents, and questions where the documents contradict each other, and score them.
2. Score `gemini-3.6-flash` on a paid or higher-quota key and compare it with 3.5-flash-lite.
3. Grow the eval set beyond 15 cases and add paraphrased variants of each question.

## Repository layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI |
| `rag.py` | Chunking, embeddings, retrieval, provider switch, retry/backoff |
| `drill_json.py` | JSON answer prompt, `validate()`, `ask_json()` (historical filename) |
| `eval_cases.py`, `eval_holdout.py` | The 15-case eval set and the 10-question hold-out set |
| `score_eval.py` | Scores a set, tags failures RETRIEVAL / GENERATION / INFRA, appends to `eval_results.json` |
| `test_validate.py` | Unit tests for `validate()` |
| `data/docs/` | The 20 source documents |

## Run locally

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:RAG_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "your-key"
.venv\Scripts\python.exe -m streamlit run app.py
.venv\Scripts\python.exe score_eval.py --label my-run --k 3
.venv\Scripts\python.exe -m pytest test_validate.py
```
