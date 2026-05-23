"""Scorer integration and reporting helpers."""

import os
import re
import statistics
import subprocess

SUBSETS = [
    "ALL",
    "S2",
    "S3",
    "S13",
    "S15",
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "Seen Only",
    "Unseen Only",
    "Single Candidate Only",
    "Multiple Candidates Only",
    "Multiple Candidates Seen Only",
    "Multiple Candidates Unseen Only",
]


def load_id_to_pos(tsv_file):
    """Load ID -> POS mapping from a TSV file (id in column 0, pos in column 2)."""
    id_to_pos = {}
    if not os.path.exists(tsv_file):
        return id_to_pos
    
    try:
        with open(tsv_file, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 2:
                    sample_id = parts[0]
                    pos = parts[2]
                    if pos:
                        id_to_pos[sample_id] = pos.upper()
    except Exception as e:
        print(f"Warning: Failed to load POS mapping from {tsv_file}: {e}")
    
    return id_to_pos


def compile_scorer_if_needed(scorer_java_path):
    scorer_java_path = os.path.abspath(scorer_java_path)
    if not os.path.exists(scorer_java_path):
        print(f"Warning: Scorer.java not found at {scorer_java_path}; skipping gold-standard report")
        return None

    scorer_dir = os.path.dirname(scorer_java_path)
    scorer_class = os.path.join(scorer_dir, "Scorer.class")
    if os.path.exists(scorer_class):
        return scorer_dir

    result = subprocess.run(
        ["javac", scorer_java_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: Failed to compile Scorer.java: {result.stderr.strip()}")
        return None
    return scorer_dir


def evaluate_with_gold_standard(
    predictions_file,
    gold_dir,
    scorer_dir,
    concept_id_to_index,
    uk_id_to_concept_id,
    tsv_file=None,
):
    """Evaluate predictions against SemEval gold subsets using Scorer.java."""
    id_to_pos = load_id_to_pos(tsv_file) if tsv_file else {}
    
    gold_standards = [
        {"title": "ALL", "gold_file": "UKC.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "S2", "gold_file": "UKC.gold.key.txt", "id_prefixes": {"senseval2"}, "pos_tags": None},
        {"title": "S3", "gold_file": "UKC.gold.key.txt", "id_prefixes": {"senseval3"}, "pos_tags": None},
        {"title": "S13", "gold_file": "UKC.gold.key.txt", "id_prefixes": {"semeval2013"}, "pos_tags": None},
        {"title": "S15", "gold_file": "UKC.gold.key.txt", "id_prefixes": {"semeval2015"}, "pos_tags": None},
        {"title": "NOUN", "gold_file": "UKC.gold.key.txt", "id_prefixes": None, "pos_tags": {"NOUN"}},
        {"title": "VERB", "gold_file": "UKC.gold.key.txt", "id_prefixes": None, "pos_tags": {"VERB"}},
        {"title": "ADJ", "gold_file": "UKC.gold.key.txt", "id_prefixes": None, "pos_tags": {"ADJ"}},
        {"title": "ADV", "gold_file": "UKC.gold.key.txt", "id_prefixes": None, "pos_tags": {"ADV"}},
        {"title": "Seen Only", "gold_file": "UKC.in.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "Unseen Only", "gold_file": "UKC.out.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "Single Candidate Only", "gold_file": "UKC.single.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "Multiple Candidates Only", "gold_file": "UKC.multi.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "Multiple Candidates Seen Only", "gold_file": "UKC.multi.in.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
        {"title": "Multiple Candidates Unseen Only", "gold_file": "UKC.multi.out.test.gold.key.txt", "id_prefixes": None, "pos_tags": None},
    ]

    if scorer_dir is None or not os.path.isdir(gold_dir) or not os.path.exists(predictions_file):
        return {}

    predicted_ids = set()
    with open(predictions_file, "r", encoding="utf-8") as pf:
        for line in pf:
            parts = line.strip().replace("\r", "").split()
            if parts:
                predicted_ids.add(parts[0])

    filtered_dir = os.path.join(os.path.dirname(predictions_file), "filtered_gold")
    os.makedirs(filtered_dir, exist_ok=True)

    results = {}
    for spec in gold_standards:
        title = spec["title"]
        gold_file = spec["gold_file"]
        id_prefixes = spec["id_prefixes"]
        pos_tags = spec["pos_tags"]

        gold_path = os.path.join(gold_dir, gold_file)
        if not os.path.exists(gold_path):
            continue

        filtered_lines = []
        total_gold_lines = 0

        with open(gold_path, "r", encoding="utf-8", errors="ignore") as gf:
            for line in gf:
                line = line.strip().replace("\r", "")
                if not line:
                    continue
                total_gold_lines += 1
                parts = line.split()
                gid = parts[0]

                if id_prefixes is not None:
                    gid_prefix = gid.split(".", 1)[0]
                    if gid_prefix not in id_prefixes:
                        continue

                if pos_tags is not None and id_to_pos:
                    gid_pos = id_to_pos.get(gid)
                    if gid_pos not in pos_tags:
                        continue

                gold_concepts = []
                for tok in parts[1:]:
                    try:
                        uk_id = int(tok)
                        concept_id = uk_id_to_concept_id.get(uk_id)
                        if concept_id is not None:
                            gold_concepts.append(int(concept_id))
                    except Exception:
                        pass

                if gid not in predicted_ids:
                    continue
                if not any(gc in concept_id_to_index for gc in gold_concepts):
                    continue

                converted_line = " ".join([gid] + [str(gc) for gc in gold_concepts])
                filtered_lines.append(converted_line)

        filtered_path = os.path.join(filtered_dir, gold_file)
        with open(filtered_path, "w", encoding="utf-8") as outf:
            for row in filtered_lines:
                outf.write(row + "\n")

        effective_size = len(filtered_lines)
        coverage_gold = (effective_size / total_gold_lines * 100.0) if total_gold_lines > 0 else 0.0
        coverage_pred = (effective_size / len(predicted_ids) * 100.0) if len(predicted_ids) > 0 else 0.0

        metrics = {
            "effective_size": effective_size,
            "coverage_gold_pct": coverage_gold,
            "coverage_predicted_pct": coverage_pred,
            "gold_total": total_gold_lines,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        result = subprocess.run(
            ["java", "-cp", scorer_dir, "Scorer", filtered_path, predictions_file],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            output = result.stdout
            p_match = re.search(r"P=\s*([\d.]+)%", output)
            r_match = re.search(r"R=\s*([\d.]+)%", output)
            f1_match = re.search(r"F1=\s*([\d.]+)%", output)
            if p_match and r_match and f1_match:
                metrics["precision"] = float(p_match.group(1))
                metrics["recall"] = float(r_match.group(1))
                metrics["f1"] = float(f1_match.group(1))

        results[title] = metrics

    return results


def print_scores_table(title, scores):
    print(f"\n{title}")
    print(f"{'Subset':35} {'Gold':>8} {'Evaluated':>10} {'Coverage%':>10} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 95)
    for subset in SUBSETS:
        m = scores.get(subset, {})
        print(
            f"{subset:35} "
            f"{int(m.get('gold_total', 0)):>8} "
            f"{int(m.get('effective_size', 0)):>10} "
            f"{float(m.get('coverage_gold_pct', 0.0)):>10.1f} "
            f"{float(m.get('precision', 0.0)):>8.1f} "
            f"{float(m.get('recall', 0.0)):>8.1f} "
            f"{float(m.get('f1', 0.0)):>8.1f}"
        )


def _scores_table_lines(title, scores):
    lines = [
        title,
        f"{'Subset':35} {'Gold':>8} {'Evaluated':>10} {'Coverage%':>10} {'P':>8} {'R':>8} {'F1':>8}",
        "-" * 95,
    ]

    for subset in SUBSETS:
        m = scores.get(subset, {})
        lines.append(
            f"{subset:35} "
            f"{int(m.get('gold_total', 0)):>8} "
            f"{int(m.get('effective_size', 0)):>10} "
            f"{float(m.get('coverage_gold_pct', 0.0)):>10.1f} "
            f"{float(m.get('precision', 0.0)):>8.1f} "
            f"{float(m.get('recall', 0.0)):>8.1f} "
            f"{float(m.get('f1', 0.0)):>8.1f}"
        )

    return lines


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def aggregate_scores(scores_list):
    if not scores_list:
        return {}

    aggregated = {}
    for subset in SUBSETS:
        gold_total = int(scores_list[0].get(subset, {}).get("gold_total", 0))
        effective_values = []
        coverage_values = []
        precision_values = []
        recall_values = []
        f1_values = []

        for scores in scores_list:
            metrics = scores.get(subset, {})
            effective_values.append(float(metrics.get("effective_size", 0.0)))
            coverage_values.append(float(metrics.get("coverage_gold_pct", 0.0)))
            precision_values.append(float(metrics.get("precision", 0.0)))
            recall_values.append(float(metrics.get("recall", 0.0)))
            f1_values.append(float(metrics.get("f1", 0.0)))

        effective_mean, effective_std = _mean_std(effective_values)
        coverage_mean, coverage_std = _mean_std(coverage_values)
        precision_mean, precision_std = _mean_std(precision_values)
        recall_mean, recall_std = _mean_std(recall_values)
        f1_mean, f1_std = _mean_std(f1_values)

        aggregated[subset] = {
            "gold_total": gold_total,
            "effective_size_mean": effective_mean,
            "effective_size_std": effective_std,
            "coverage_gold_pct_mean": coverage_mean,
            "coverage_gold_pct_std": coverage_std,
            "precision_mean": precision_mean,
            "precision_std": precision_std,
            "recall_mean": recall_mean,
            "recall_std": recall_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
        }

    return aggregated


def _format_mean_std(mean, std, decimals=1):
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def print_aggregate_scores_table(title, scores_list):
    aggregated = aggregate_scores(scores_list)
    print(f"\n{title}")
    if not aggregated:
        print("No scores available.")
        return

    print(
        f"{'Subset':35} {'Gold':>8} {'Evaluated':>18} {'Coverage%':>18} "
        f"{'P':>18} {'R':>18} {'F1':>18}"
    )
    print("-" * 135)

    for subset in SUBSETS:
        m = aggregated.get(subset, {})
        print(
            f"{subset:35} "
            f"{int(m.get('gold_total', 0)):>8} "
            f"{_format_mean_std(m.get('effective_size_mean', 0.0), m.get('effective_size_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('coverage_gold_pct_mean', 0.0), m.get('coverage_gold_pct_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('precision_mean', 0.0), m.get('precision_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('recall_mean', 0.0), m.get('recall_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('f1_mean', 0.0), m.get('f1_std', 0.0)):>18}"
        )


def _aggregate_scores_table_lines(title, scores_list):
    aggregated = aggregate_scores(scores_list)
    lines = [title]
    if not aggregated:
        lines.append("No scores available.")
        return lines

    lines.append(
        f"{'Subset':35} {'Gold':>8} {'Evaluated':>18} {'Coverage%':>18} "
        f"{'P':>18} {'R':>18} {'F1':>18}"
    )
    lines.append("-" * 135)

    for subset in SUBSETS:
        m = aggregated.get(subset, {})
        lines.append(
            f"{subset:35} "
            f"{int(m.get('gold_total', 0)):>8} "
            f"{_format_mean_std(m.get('effective_size_mean', 0.0), m.get('effective_size_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('coverage_gold_pct_mean', 0.0), m.get('coverage_gold_pct_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('precision_mean', 0.0), m.get('precision_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('recall_mean', 0.0), m.get('recall_std', 0.0)):>18} "
            f"{_format_mean_std(m.get('f1_mean', 0.0), m.get('f1_std', 0.0)):>18}"
        )

    return lines


def append_aggregate_report(report_file, title, scores_list):
    with open(report_file, "a", encoding="utf-8") as f:
        for line in _aggregate_scores_table_lines(title, scores_list):
            f.write(line + "\n")
        f.write("\n\n")


def append_evaluation_report(report_file, epoch, eval_scores, test_scores):
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch}\n")
        f.write("=" * 95 + "\n")

        if eval_scores:
            for line in _scores_table_lines("Eval Set Coverage & Scores", eval_scores):
                f.write(line + "\n")
        else:
            f.write("Eval Set Coverage & Scores\n")
            f.write("No eval scores available.\n")

        f.write("\n")

        if test_scores:
            for line in _scores_table_lines("Test Set Coverage & Scores", test_scores):
                f.write(line + "\n")
        else:
            f.write("Test Set Coverage & Scores\n")
            f.write("No test scores available.\n")

        f.write("\n\n")


def log_scores_to_wandb(prefix, scores, wandb_module):
    if not scores:
        return
    for category, metrics in scores.items():
        if not metrics:
            continue
        wandb_module.log(
            {
                f"{prefix}/precision/{category}": float(metrics.get("precision", 0.0)),
                f"{prefix}/recall/{category}": float(metrics.get("recall", 0.0)),
                f"{prefix}/f1/{category}": float(metrics.get("f1", 0.0)),
                f"{prefix}/coverage_gold_pct/{category}": float(metrics.get("coverage_gold_pct", 0.0)),
            }
        )


def log_aggregate_scores_to_wandb(prefix, scores_list, wandb_module):
    aggregated = aggregate_scores(scores_list)
    if not aggregated:
        return

    for category, metrics in aggregated.items():
        wandb_module.log(
            {
                f"{prefix}/mean/precision/{category}": float(metrics.get("precision_mean", 0.0)),
                f"{prefix}/std/precision/{category}": float(metrics.get("precision_std", 0.0)),
                f"{prefix}/mean/recall/{category}": float(metrics.get("recall_mean", 0.0)),
                f"{prefix}/std/recall/{category}": float(metrics.get("recall_std", 0.0)),
                f"{prefix}/mean/f1/{category}": float(metrics.get("f1_mean", 0.0)),
                f"{prefix}/std/f1/{category}": float(metrics.get("f1_std", 0.0)),
                f"{prefix}/mean/coverage_gold_pct/{category}": float(metrics.get("coverage_gold_pct_mean", 0.0)),
                f"{prefix}/std/coverage_gold_pct/{category}": float(metrics.get("coverage_gold_pct_std", 0.0)),
            }
        )
