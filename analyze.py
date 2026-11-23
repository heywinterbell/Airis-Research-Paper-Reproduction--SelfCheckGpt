#!/usr/bin/env python3
"""
Analyse SelfCheckGPT reproduction logs (AIRIS Research Hackathon, Paper 6).

Reads the JSONL run logs in results/runs/ and reports:
  - per model: accuracy, consistency vs correctness split, AUC and AUC-PR
    of consistency as a correctness predictor
  - cross-model: agreement across the two models' samples, its AUC, and how
    many confidently-wrong cases (5/5 within-model agreement, wrong answer)
    cross-model disagreement catches
  - majority vote (Extension 2): per-model and cross-model accuracy of the
    most common sampled answer vs the greedy answer
  - failure buckets with real examples (for the report's hand analysis)

Usage:
  python analyze.py --runs-dir results/runs --questions questions.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import is_correct, normalize  # noqa: E402 - same scoring rule as the runner


def auc(scores, labels):
    """AUC via Mann-Whitney U. scores = consistency, labels = 1 correct / 0 wrong."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def average_precision(scores, labels):
    """Average precision (AUC-PR approx). scores = consistency,
    labels = 1 correct / 0 wrong. Uses distinct score thresholds so tied
    scores are counted together (with only 6 possible consistency values
    there are many ties)."""
    total_pos = sum(labels)
    if total_pos == 0 or total_pos == len(labels):
        return None
    ap = 0.0
    prev_recall = 0.0
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 0)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / total_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def load_logs(runs_dir):
    """Return {(model, question_id): {'greedy': rec, 'samples': [rec, ...]}}."""
    data = defaultdict(lambda: {"greedy": None, "samples": []})
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(runs_dir, name)) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["model"], rec["question_id"])
                if rec["kind"] == "greedy":
                    data[key]["greedy"] = rec
                else:
                    data[key]["samples"].append(rec)
    return data


def main():
    parser = argparse.ArgumentParser(description="Analyse SelfCheckGPT runs.")
    parser.add_argument("--runs-dir", default="results/runs")
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = {q["id"]: q for q in json.load(f)}
    data = load_logs(args.runs_dir)
    models = sorted({m for m, _ in data})
    qids = sorted({q for _, q in data})

    report = {"models": models, "n_questions": len(qids), "per_model": {},
              "cross_model": {}, "confidently_wrong": [], "failures": {}}
    lines = []
    lines.append("# SelfCheckGPT reproduction - analysis")
    lines.append("")
    lines.append(f"Models: {', '.join(models)} | Questions: {len(qids)}")
    lines.append("")

    # ---- per model ---------------------------------------------------------
    per_model_cons = {}
    for model in models:
        scores, labels, rows, mv_labels = [], [], [], []
        for qid in qids:
            entry = data.get((model, qid))
            if not entry or entry["greedy"] is None:
                continue
            greedy = entry["greedy"]
            samples = [s for s in entry["samples"] if s["raw"] is not None]
            if not samples:
                continue
            greedy_norm = normalize(greedy["extracted"])
            agree = sum(1 for s in samples
                        if normalize(s["extracted"]) == greedy_norm)
            consistency = agree / len(samples)
            scores.append(consistency)
            labels.append(1 if greedy["correct"] else 0)
            rows.append((qid, consistency, greedy["correct"], agree,
                         len(samples)))
            # Majority vote (Extension 2): the most common answer across
            # the samples. Ties prefer the greedy answer if it is tied,
            # else the first alphabetically (deterministic).
            norm_answers = [normalize(s["extracted"]) for s in samples]
            norm_answers = [a for a in norm_answers if a]
            if norm_answers:
                counts = Counter(norm_answers)
                top = max(counts.values())
                tied = [a for a, c in counts.items() if c == top]
                if len(tied) == 1:
                    majority = tied[0]
                elif greedy_norm in tied:
                    majority = greedy_norm
                else:
                    majority = sorted(tied)[0]
                mv_labels.append(1 if is_correct(
                    majority, questions[qid]["answer"],
                    questions[qid].get("aliases", [])) else 0)
        per_model_cons[model] = rows
        acc = sum(labels) / len(labels) if labels else 0
        mv_acc = sum(mv_labels) / len(mv_labels) if mv_labels else None
        model_auc = auc(scores, labels)
        model_ap = average_precision(scores, labels)
        hi = [y for (q, c, y, a, n) in rows if c == 1.0]
        lo = [y for (q, c, y, a, n) in rows if c <= 0.4]
        hi_rate = sum(hi) / len(hi) if hi else None
        lo_rate = sum(lo) / len(lo) if lo else None
        report["per_model"][model] = {
            "accuracy": round(acc, 3),
            "auc_consistency": round(model_auc, 3) if model_auc else None,
            "auc_pr_consistency": (round(model_ap, 3)
                                  if model_ap is not None else None),
            "majority_vote_accuracy": (round(mv_acc, 3)
                                       if mv_acc is not None else None),
            "n_5_of_5_agree": len(hi),
            "correct_when_5_of_5": round(hi_rate, 3) if hi_rate is not None else None,
            "n_2_or_fewer_agree": len(lo),
            "correct_when_2_or_fewer": round(lo_rate, 3) if lo_rate is not None else None,
        }
        lines.append(f"## {model}")
        lines.append(f"- Greedy accuracy: {acc:.1%} ({sum(labels)}/{len(labels)})")
        lines.append(f"- AUC of consistency as a correctness predictor: "
                     f"{model_auc:.3f}" if model_auc else "- AUC: n/a")
        lines.append(f"- AUC-PR of consistency: "
                     f"{model_ap:.3f}" if model_ap is not None
                     else "- AUC-PR: n/a")
        lines.append(f"- Majority-vote accuracy: {mv_acc:.1%} "
                     f"({sum(mv_labels)}/{len(mv_labels)})"
                     if mv_acc is not None
                     else "- Majority-vote accuracy: n/a")
        lines.append(f"- 5/5 agree: {len(hi)} questions, "
                     f"{hi_rate:.1%} correct" if hi_rate is not None
                     else "- 5/5 agree: none")
        lines.append(f"- 2/5 or fewer agree: {len(lo)} questions, "
                     f"{lo_rate:.1%} correct" if lo_rate is not None
                     else "- 2/5 or fewer agree: none")
        lines.append("")

    # ---- cross-model -------------------------------------------------------
    if len(models) == 2:
        a, b = models
        cross_scores, cross_labels, caught, total_cw = [], [], 0, 0
        for qid in qids:
            ea, eb = data.get((a, qid)), data.get((b, qid))
            if not ea or not eb or ea["greedy"] is None or eb["greedy"] is None:
                continue
            sa = [normalize(s["extracted"]) for s in ea["samples"] if s["raw"]]
            sb = [normalize(s["extracted"]) for s in eb["samples"] if s["raw"]]
            if not sa or not sb:
                continue
            all_samples = sa + sb
            majority = max(set(all_samples), key=all_samples.count)
            cross = sum(1 for s in all_samples if s == majority) / len(all_samples)
            # Label = is the majority answer (the answer cross-model agreement
            # points to) correct? That mirrors the per-model setup, where the
            # label is whether the greedy answer the score points to is right.
            cross_scores.append(cross)
            cross_labels.append(1 if is_correct(
                majority, questions[qid]["answer"],
                questions[qid].get("aliases", [])) else 0)
            # confidently-wrong case: model A 5/5 agree but wrong
            ga = normalize(ea["greedy"]["extracted"])
            if len(sa) == 5 and all(s == ga for s in sa) and not ea["greedy"]["correct"]:
                total_cw += 1
                if normalize(eb["greedy"]["extracted"]) != ga:
                    caught += 1
                    report["confidently_wrong"].append({
                        "question_id": qid,
                        "question": questions[qid]["question"],
                        "gold": questions[qid]["answer"],
                        "model_a_greedy": ea["greedy"]["raw"],
                        "model_b_greedy": eb["greedy"]["raw"],
                        "caught_by_cross_model": True,
                    })
        cross_auc = auc(cross_scores, cross_labels)
        # Cross-model majority-vote accuracy: cross_labels already records
        # whether the majority answer across all 10 samples is correct.
        cross_mv_acc = (sum(cross_labels) / len(cross_labels)
                        if cross_labels else None)
        cross_ap = average_precision(cross_scores, cross_labels)
        report["cross_model"] = {
            "auc_cross_agreement": round(cross_auc, 3) if cross_auc else None,
            "auc_pr_cross_agreement": (round(cross_ap, 3)
                                       if cross_ap is not None else None),
            "confidently_wrong_cases": total_cw,
            "caught_by_cross_model": caught,
            "majority_vote_accuracy": (round(cross_mv_acc, 3)
                                       if cross_mv_acc is not None else None),
        }
        lines.append("## Cross-model agreement")
        lines.append(f"- AUC of cross-model agreement: "
                     f"{cross_auc:.3f}" if cross_auc else "- AUC: n/a")
        lines.append(f"- AUC-PR of cross-model agreement: "
                     f"{cross_ap:.3f}" if cross_ap is not None
                     else "- AUC-PR: n/a")
        lines.append(f"- Confidently-wrong cases (5/5 agree, wrong): {total_cw}")
        lines.append(f"- Caught by cross-model disagreement: {caught}")
        lines.append(f"- Cross-model majority-vote accuracy: "
                     f"{cross_mv_acc:.1%} "
                     f"({sum(cross_labels)}/{len(cross_labels)})"
                     if cross_mv_acc is not None
                     else "- Cross-model majority-vote accuracy: n/a")
        lines.append("")

    # ---- failure buckets ---------------------------------------------------
    buckets = defaultdict(list)
    for model in models:
        for qid, consistency, correct, agree, n in per_model_cons[model]:
            if consistency == 1.0 and not correct:
                buckets["high-consistency-but-wrong"].append((model, qid))
            elif consistency <= 0.4 and correct:
                buckets["low-consistency-but-correct"].append((model, qid))
            elif consistency <= 0.4 and not correct:
                buckets["low-consistency-and-wrong"].append((model, qid))
            elif consistency == 1.0 and correct:
                buckets["high-consistency-and-correct"].append((model, qid))
    report["failures"] = {k: [{"model": m, "question_id": q,
                               "question": questions[q]["question"],
                               "gold": questions[q]["answer"]}
                              for m, q in v]
                          for k, v in buckets.items()}
    lines.append("## Failure buckets")
    for name, items in buckets.items():
        lines.append(f"- {name}: {len(items)}")
        for model, qid in items[:3]:
            lines.append(f"  - {model} {qid}: {questions[qid]['question']} "
                         f"(gold: {questions[qid]['answer']})")
    lines.append("")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "analysis.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {os.path.join(args.out, 'analysis.json')} and analysis.md")


if __name__ == "__main__":
    main()
