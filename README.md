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

## Larger-corpus test

To test retrieval beyond 20 one-chunk documents, I built `data/docs_large/`: the original 20 plus 14 synthetic documents I wrote with an AI assistant (40 chunks in total). Eleven are short and deliberately overlap in vocabulary with the originals (holiday-camp refunds, adult-section instalments, lost property, tournament weather, privacy) and three are longer multi-chunk documents (constitution, annual report, coach handbook). The same eval questions were re-scored with no change to the app.

| Set | Corpus | Score | Failures |
| --- | --- | --- | --- |
| Main (15) | 20 docs | 15/15 | none |
| Main (15) | 34 docs | 13/15 | 1 **retrieval**, 1 generation |
| Hold-out (10) | 20 docs | 10/10 | none |
| Hold-out (10) | 34 docs | 9/10 | 1 invalid case (see below) |

- **First retrieval failure:** "Who is the club treasurer and what is the kit return policy?" retrieved `faq_05_refunds`, `x03_fundraising` and `x10_lost_property` but not `handbook_03_kit`. The lost-property document ("kit issued by the club ... is returned to the equipment officer") sits closer to "kit return policy" than the real kit rule does, and the app answered from the wrong document instead of refusing. This is the failure mode a tiny corpus hides.
- **Case 9 ("can't afford the subscription")** dropped the word "confidentially" even though the right documents were retrieved. It has wobbled across several runs, so I treat it as noise near the pass/fail boundary.
- **Hold-out case 8 ("dogs") was my own test-data mistake:** the synthetic parking document contained a dog rule, so the question became answerable and the app correctly answered it. The document has been corrected; the original run is kept in `eval_results.json` with a note, and the score above is as recorded.
- The new documents are synthetic and still a small corpus. Real, messier documents would be a harder test.

### Fixing the retrieval failure (one variable at a time)

| Change | Corpus | Main (15) | Hold-out (10) | Effect on the retrieval failure |
| --- | --- | --- | --- | --- |
| none (k=3) | 34 docs | 13/15 | 9/10* | case 15 still fails (wrong document retrieved) |
| retrieve k=5 | 34 docs | 13/15 | not run | **no help**: the kit handbook still never appears in the top 5 |
| split compound questions, one search per part (`SPLIT_QUERIES=1`) | 34 docs | **15/15** | **10/10** | fixed |
| splitting on | 20 docs | 14/15 | invalid (1 rate-limited case) | not needed here; cost one borderline case |

\* the one miss was my own test-data mistake (see above).

- **More context was not the answer; the query was.** "Who is the club treasurer and what is the kit return policy?" is one embedding that mixes two topics, so it lands nearer to refunds and lost-property documents than to the kit handbook. Searching for each part separately (and also for the whole question) fixes it.
- **Caveat: I built the fix after seeing case 15 fail.** The hold-out set (10/10) was not used to design it, which is why that number matters more than the main-set 15/15.
- **It is not free.** Case 9 ("we can't afford the subscription") passes when the model sees exactly the three expected chunks and tends to fail when extra chunks are added (k=5, the larger corpus, and splitting on the small corpus). Splitting adds chunks on compound questions, so on the 20-document corpus it probably cost that borderline case and gained nothing.
- **Decision:** the live app, which uses the 20-document corpus, keeps the original behaviour. Splitting stays an opt-in flag, validated for a larger corpus. It should be turned on when the corpus grows, and re-checked then.

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

1. Test on real, messy documents (not synthetic ones), with paraphrased questions and documents that contradict each other.
2. Make the answer check less brittle: case 9 requires the word "confidential", which a correct answer can omit.
3. Score `gemini-3.6-flash` on a paid or higher-quota key and compare it with 3.5-flash-lite.
4. Grow the eval sets and score several runs per configuration, so single-run noise (like case 9) is visible.

## Repository layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI |
| `rag.py` | Chunking, embeddings, retrieval, provider switch, retry/backoff |
| `drill_json.py` | JSON answer prompt, `validate()`, `ask_json()` (historical filename) |
| `eval_cases.py`, `eval_holdout.py` | The 15-case eval set and the 10-question hold-out set |
| `score_eval.py` | Scores a set, tags failures RETRIEVAL / GENERATION / INFRA, appends to `eval_results.json` |
| `test_validate.py` | Unit tests for `validate()` |
| `data/docs/` | The 20 source documents (what the live app indexes) |
| `data/docs_large/` | The 20 originals plus 14 synthetic documents, for the larger-corpus test (`DOCS_GLOB=data/docs_large/*.txt`) |

## Run locally

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:RAG_PROVIDER = "gemini"; $env:GEMINI_API_KEY = "your-key"
.venv\Scripts\python.exe -m streamlit run app.py
.venv\Scripts\python.exe score_eval.py --label my-run --k 3
.venv\Scripts\python.exe -m pytest test_validate.py
```
