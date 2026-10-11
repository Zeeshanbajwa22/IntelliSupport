import json
import sys
import time
import requests

BASE_URL = "http://127.0.0.1:8000"
USER_ID = "eval_user"
PDF_PATH = "eval_data/sample.pdf"
CASES_PATH = "eval_data/eval_set.json"
PASS_THRESHOLD = 0.8


def upload_pdf():
    with open(PDF_PATH, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/upload",
            data={"user_id": USER_ID},
            files={"file": ("sample.pdf", f, "application/pdf")},
            timeout=300,
        )
    r.raise_for_status()
    print("Uploaded test document:", r.json())


def ask(query):
    r = requests.post(
        f"{BASE_URL}/ask",
        json={"user_id": USER_ID, "query": query},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main():
    with open(CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    # Skip the upload on repeat runs so the same document isn't stored twice
    if "--skip-upload" not in sys.argv:
        upload_pdf()

    results = []
    routing_correct = 0
    answer_correct = 0

    for i, case in enumerate(cases, 1):
        try:
            res = ask(case["query"])
            actual_source = res["source_used"]
            answer = res["answer"]
            error = None
        except Exception as e:
            actual_source, answer, error = "error", "", str(e)

        routing_ok = actual_source == case["expected_source"]
        answer_ok = all(k.lower() in answer.lower() for k in case["keywords"])
        passed = routing_ok and answer_ok

        routing_correct += routing_ok
        answer_correct += answer_ok

        status = "PASS" if passed else "FAIL"
        print(f"[{i:02d}] {status} | expected={case['expected_source']:<9} got={actual_source:<9} | {case['query']}")

        results.append({
            "query": case["query"],
            "expected_source": case["expected_source"],
            "actual_source": actual_source,
            "keywords": case["keywords"],
            "routing_ok": routing_ok,
            "answer_ok": answer_ok,
            "passed": passed,
            "answer": answer,
            "error": error,
        })
        time.sleep(1)  # be gentle with the free API rate limits

    total = len(cases)
    passed_count = sum(r["passed"] for r in results)
    pass_rate = passed_count / total

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Routing accuracy : {routing_correct}/{total} ({routing_correct / total:.0%})")
    print(f"Answer accuracy  : {answer_correct}/{total} ({answer_correct / total:.0%})")
    print(f"Overall pass rate: {passed_count}/{total} ({pass_rate:.0%})")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Full answers saved to eval_results.json")

    if pass_rate < PASS_THRESHOLD:
        print(f"RESULT: FAILED (below {PASS_THRESHOLD:.0%} threshold)")
        sys.exit(1)
    print("RESULT: PASSED")


if __name__ == "__main__":
    main()