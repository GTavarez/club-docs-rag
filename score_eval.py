"""Day 23 - score the RAG app against the 15 eval cases.

Run:  .venv\Scripts\python.exe score_eval.py --label baseline
      .venv\Scripts\python.exe score_eval.py --label k5 --k 5     (Day 24: change ONE thing)

Each failure is tagged:
  RETRIEVAL  - an expected source file was not in the top-k chunks
  GENERATION - the right chunks were retrieved, but the answer/sources/refusal was wrong
Results are appended to eval_results.json so you can compare before/after.
"""
import argparse
import json
from datetime import datetime

from drill_json import ask_json
from eval_cases import EVAL as MAIN_EVAL
from rag import CHAT_MODEL, EMBED_MODEL, PROVIDER, RETRYABLE, index, retrieve


def has_fact(answer, fact):
    return any(alt.strip().lower() in answer.lower() for alt in fact.split("|"))


def score_case(c, k):
    retrieved = {m["source"] for m, _ in retrieve(c["q"], k)}
    out = ask_json(c["q"], k)
    problems, tag = [], None

    if c["sources"] is None:                      # must refuse
        if out["found"] or out["sources"]:
            problems.append("should have refused")
            tag = "GENERATION"
    else:
        missing = c["sources"] - retrieved
        if missing:
            problems.append(f"not retrieved: {sorted(missing)}")
            tag = "RETRIEVAL"
        if not out["found"]:
            problems.append("refused, but the answer is in the docs")
            tag = tag or "GENERATION"
        else:
            for fact in c["must_say"]:
                if not has_fact(out["answer"], fact):
                    problems.append(f"answer missing '{fact}'")
                    tag = tag or "GENERATION"
            if not (c["sources"] & set(out["sources"])):
                problems.append(f"cited {out['sources']}, wanted one of {sorted(c['sources'])}")
                tag = tag or "GENERATION"
    return {"q": c["q"], "pass": not problems, "tag": tag, "problems": problems,
            "retrieved": sorted(retrieved), "out": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--set", dest="case_set", choices=["main", "holdout"], default="main",
                    help="main = the 15 tuned cases; holdout = unseen cases in eval_holdout.py")
    args = ap.parse_args()

    if args.case_set == "holdout":
        from eval_holdout import EVAL
    else:
        EVAL = MAIN_EVAL
    if not EVAL:
        raise SystemExit("No cases in this set yet. Write them in eval_holdout.py first.")

    index()
    results = []
    for i, c in enumerate(EVAL, 1):
        try:
            r = score_case(c, args.k)
        except RETRYABLE as e:          # quota/outage: NOT a model failure
            r = {"q": c["q"], "pass": False, "tag": "INFRA",
                 "problems": [f"{type(e).__name__}: request never completed"],
                 "retrieved": [], "out": None}
        except Exception as e:
            r = {"q": c["q"], "pass": False, "tag": "GENERATION",
                 "problems": [f"crashed: {type(e).__name__}: {e}"], "retrieved": [], "out": None}
        results.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {i:>2}. {c['q']}")
        if not r["pass"]:
            print(f"        [{r['tag']}] " + "; ".join(r["problems"]))
            print(f"        retrieved: {r['retrieved']}")
            print(f"        got: {json.dumps(r['out'])[:200]}")

    n = sum(r["pass"] for r in results)
    ret = sum(r["tag"] == "RETRIEVAL" for r in results)
    gen = sum(r["tag"] == "GENERATION" for r in results)
    infra = sum(r["tag"] == "INFRA" for r in results)
    if infra:
        print(f"\n*** {infra} case(s) never reached the model (quota/outage). "
              f"This run is NOT a valid quality measurement. ***")
    print(f"\nScore: {n}/{len(results)}   failures -> retrieval: {ret}, generation: {gen}"
          f"   (label={args.label}, k={args.k}, provider={PROVIDER}, model={CHAT_MODEL})")

    try:
        with open("eval_results.json", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    history.append({"when": datetime.now().isoformat(timespec="seconds"), "label": args.label,
                    "k": args.k, "provider": PROVIDER, "chat_model": CHAT_MODEL,
                    "embed_model": EMBED_MODEL, "case_set": args.case_set, "score": n, "total": len(results),
                    "retrieval_fails": ret, "generation_fails": gen,
                    "infra_fails": infra, "valid": infra == 0, "cases": results})
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Saved to eval_results.json")


if __name__ == "__main__":
    main()
