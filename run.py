#!/usr/bin/env python3
"""
SelfCheckGPT reproduction runner (AIRIS Research Hackathon, Paper 6).

For each question and each model:
  - 1 greedy answer (temperature 0)
  - N sampled answers (temperature ~0.7, default 5)

Every call is logged as one JSON line in results/runs/<run-id>-<model>.jsonl,
including failed calls. The report must attach these logs.

The runner talks to an OpenAI-compatible chat API (the shared hackathon
server). It needs BASE_URL, API_KEY, MODEL_A and MODEL_B.

Usage:
  python run.py --limit 10            # smoke test, 10 questions
  python run.py                       # full run on the server

Env vars (or pass --base-url / --api-key / --models):
  BASE_URL, API_KEY, MODEL_A, MODEL_B
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone

import requests

PROMPT_TEMPLATE = (
    "Answer the following question with a short answer - just the answer, "
    "no explanation.\n\nQuestion: {question}\nAnswer:"
)


# --------------------------------------------------------------------------
# Answer normalisation + extraction
# --------------------------------------------------------------------------

def normalize(text):
    """Lowercase, strip accents, drop punctuation/commas/spaces -> one token string."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.replace(",", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_answer(raw):
    """Pull the model's final answer out of its raw reply.

    Models are asked for just the answer, but they sometimes add a sentence.
    Rule: take the first non-empty line, strip prefixes like 'Answer:',
    then normalise. If the reply is a sentence, the first line usually holds it.
    """
    if raw is None:
        return ""
    line = ""
    for candidate in raw.strip().splitlines():
        if candidate.strip():
            line = candidate.strip()
            break
    line = re.sub(r"^(answer|the answer is|a:)\s*:?\s*", "", line, flags=re.I)
    return line


def is_correct(raw, gold, aliases):
    """True if the model's answer matches the gold answer or any alias.

    Exact match on normalised text, else containment either way: the gold
    appears inside the reply (handles 'Paris, France'), or the reply appears
    as a whole word/phrase inside the gold (handles a short correct answer
    like 'Yes' or 'Anno Domini' against a sentence gold such as
    'Yes, sharks can get cancer' — the TruthfulQA case, where the gold is a
    sentence and the model gives a short answer).
    """
    extracted = normalize(extract_answer(raw))
    if not extracted:
        return False
    targets = [normalize(gold)] + [normalize(a) for a in aliases]
    if extracted in targets:
        return True
    for target in targets:
        if not target:
            continue
        if target in extracted:
            return True
        # reply inside gold, as a whole word/phrase (word boundaries),
        # so 'yes' matches 'yes sharks can get cancer' but not 'yesterday'.
        if re.search(r'(?<!\w)' + re.escape(extracted) + r'(?!\w)', target):
            return True
    return False


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

class OpenAIBackend:
    """Any OpenAI-compatible /chat/completions endpoint (shared server, etc.)."""

    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def call(self, model, prompt, temperature, max_tokens=1024):
        url = self.base_url + "/chat/completions"
        headers = {"Authorization": "Bearer " + self.api_key}
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        if not content.strip():
            # Qwen3 "thinking" models put their reasoning in
            # message.reasoning_content; with too small a max_tokens the
            # visible answer comes back empty. An empty reply is missing
            # data, not a wrong answer: raise so the retry logic runs and
            # the log records an error instead of a score.
            raise ValueError("empty content from model")
        return content


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def call_with_retry(backend, model, prompt, temperature, max_tokens=1024):
    """Call the model, retrying on rate limits / transient 5xx (max 3 tries)."""
    delay = 5
    for attempt in range(3):
        try:
            return backend.call(model, prompt, temperature, max_tokens), None
        except Exception as exc:  # noqa: BLE001 - log and retry, report if fatal
            error = str(exc)
            if attempt == 2:
                return None, error
            time.sleep(delay)
            delay *= 2
    return None, "unreachable"


def main():
    parser = argparse.ArgumentParser(description="Run the SelfCheckGPT reproduction.")
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--sample-temp", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max tokens per reply (Qwen3 thinking models need 1000+)")
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all questions, else first N")
    parser.add_argument("--models", default="",
                        help="comma-separated model names (overrides env)")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between calls (rate-limit courtesy)")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [os.environ.get("MODEL_A", ""), os.environ.get("MODEL_B", "")]
        models = [m for m in models if m]
    if not models:
        print("error: no models given. Set MODEL_A/MODEL_B or pass --models.",
              file=sys.stderr)
        sys.exit(1)

    base_url = args.base_url or os.environ.get("BASE_URL", "")
    api_key = args.api_key or os.environ.get("API_KEY", "")
    if not base_url or not api_key:
        print("error: this runner needs BASE_URL and API_KEY.",
              file=sys.stderr)
        sys.exit(1)
    backend = OpenAIBackend(base_url, api_key)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_dir = os.path.join(args.out, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "backend": "openai",
        "models": models,
        "n_samples": args.n_samples,
        "sample_temp": args.sample_temp,
        "max_tokens": args.max_tokens,
        "n_questions": len(questions),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out, f"manifest-{run_id}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    for model in models:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        log_path = os.path.join(runs_dir, f"{run_id}-{safe}.jsonl")
        n_calls = 0
        with open(log_path, "w") as log:
            for q in questions:
                # 1 greedy answer (temperature 0)
                prompt = PROMPT_TEMPLATE.format(question=q["question"])
                raw, error = call_with_retry(backend, model, prompt, 0.0,
                                             args.max_tokens)
                log.write(json.dumps({
                    "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(),
                    "model": model, "question_id": q["id"], "kind": "greedy",
                    "temperature": 0.0, "prompt": prompt, "raw": raw,
                    "extracted": extract_answer(raw) if raw else "",
                    "correct": is_correct(raw, q["answer"], q.get("aliases", []))
                    if raw else False,
                    "error": error,
                }) + "\n")
                n_calls += 1
                time.sleep(args.delay)
                # N sampled answers (temperature ~0.7)
                for i in range(args.n_samples):
                    raw, error = call_with_retry(
                        backend, model, prompt, args.sample_temp,
                        args.max_tokens)
                    log.write(json.dumps({
                        "run_id": run_id,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "model": model, "question_id": q["id"],
                        "kind": "sample", "sample_idx": i,
                        "temperature": args.sample_temp, "prompt": prompt,
                        "raw": raw,
                        "extracted": extract_answer(raw) if raw else "",
                        "correct": is_correct(raw, q["answer"],
                                              q.get("aliases", []))
                        if raw else False,
                        "error": error,
                    }) + "\n")
                    n_calls += 1
                    time.sleep(args.delay)
        print(f"{model}: {n_calls} calls logged to {log_path}")

    print(f"done. run_id={run_id}. Analyse with: python analyze.py --runs-dir {runs_dir}")


if __name__ == "__main__":
    main()
