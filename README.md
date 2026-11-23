# SelfCheckGPT reproduction — AIRIS Research Hackathon (Paper 6)

Reproduces the central claim of Manakul et al., "SelfCheckGPT"
(EMNLP 2023): sampled responses that agree with each other indicate the
model knows the concept; sampled responses that diverge indicate
hallucination. Consistency predicts factuality, with no external database.

Extension: cross-model consistency — does agreement **across two different
models** predict correctness better than agreement within one model, and
does it catch the confidently-wrong cases (all samples agree, answer wrong)
that single-model SelfCheckGPT is blind to?

## How it runs

Run from a MacBook on the college LAN — the server at 10.7.25.115 is not
reachable from outside that network.

```
export BASE_URL="http://10.7.25.115:30400/v1"
export API_KEY="<team key - type it in your own terminal, never into a file>"
export MODEL_A="qwen3-30b-a3b"
export MODEL_B="qwen3-32b"

# Smoke test first (10 questions), then the full run
python run.py --limit 10
python run.py
python analyze.py --runs-dir results/runs
```

NOTE: both models are Qwen3 "thinking" models, so run.py uses
max_tokens=1024 by default (under 1000 the answer may come back empty -
the reasoning lives in message.reasoning_content). If you get HTTP 429,
raise --delay (e.g. --delay 2) and the retry logic handles the rest.

One command runs the whole experiment per model: 1 greedy answer
(temperature 0) + 5 samples (temperature 0.7) for each of the 60 questions
in `questions.json` — 360 calls per model, 720 total. The extension
(cross-model agreement) reuses those same samples: 0 extra calls.

## Files

- `questions.json` — 60 short factual questions with verified answers:
  15 easy, 20 medium, 25 hard / misconception traps. Every answer is a
  single verifiable fact; aliases list accepted variants (e.g. Berne for Bern).
- `run.py` — the experiment runner. Logs **every** call as one JSON line in
  `results/runs/<run-id>-<model>.jsonl`, including failed calls.
- `analyze.py` — per-model accuracy, consistency-vs-correctness split, AUC
  of consistency as a correctness predictor, cross-model agreement + AUC,
  confidently-wrong cases, and failure buckets. Writes
  `results/analysis.json` and `results/analysis.md`.
- `results/` — run logs and analysis output (gitignored; attach the logs to
  the report as required).

## Scoring rule

The model is asked for just the answer. Its reply is normalised
(lowercase, accents stripped, punctuation/commas dropped) and compared
against the gold answer and its aliases — exact match, or containment
("Paris, France" counts for "Paris"). Borderline cases are visible in the
raw logs, so they can be hand-checked in the failure analysis.

## Notes for the report

- **Dataset:** the paper's own dataset is `wiki_bio_gpt3_hallucination`
  (238 GPT-3 passages about people from WikiBio, with sentence-level human
  factuality annotations — HuggingFace
  `potsawee/wiki_bio_gpt3_hallucination`). Our reproduction uses our own 60
  factual questions instead: same central claim (consistency signals
  factuality), simpler setup, already built and tested. State this
  adaptation openly in the report.
- **Logs must be real and complete:** every call is logged with a timestamp,
  the exact prompt, the raw reply and any error — including failed calls.
  Never edit a log by hand, never delete a failed run.
- This is an adaptation of Paper 6, stated honestly: the original used
  WikiBio passages + human annotation + BERTScore; we test the same core
  claim on short factual QA with exact-match agreement (the n-gram
  variant's spirit).
- Disclosure: this code and the question set were built with the help of
  Rain/Muse (an AI tool). List it in the submission disclosure, with what
  it was used for.
- Never put the server key in a file or in git. Pass it via env vars only.
