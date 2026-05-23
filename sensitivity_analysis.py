"""sensitivity_analysis.py - GAS weight perturbation analysis."""

import json
import logging
import statistics
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/processed.json")
OUTPUT_FILE = Path("data/sensitivity_analysis.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sensitivity")


def score_experience(years, no_exp_signal):
    if no_exp_signal:
        return 100
    if years is None:
        return 70
    if years <= 0:
        return 100
    if years == 1:
        return 70
    if years == 2:
        return 35
    if years == 3:
        return 10
    return 0


def score_degree(degree):
    if degree is None or degree == "bachelor":
        return 100
    if degree == "master":
        return 40
    if degree == "phd":
        return 0
    return 80


def score_tech_breadth(tech_count):
    if tech_count <= 3:
        return 100
    if tech_count <= 5:
        return 85
    if tech_count <= 7:
        return 65
    if tech_count <= 10:
        return 40
    return 15


def score_language_tone(posting):
    return 85 if posting.get("no_experience_signal") else 65


def score_salary(posting):
    has_min = posting.get("salary_min") is not None
    has_max = posting.get("salary_max") is not None
    is_predicted = posting.get("salary_is_predicted") in (True, 1, "1")
    if has_min and has_max and not is_predicted:
        return 100
    if has_min or has_max:
        return 60
    return 0


def compute_gas(posting, weights):
    components = {
        "experience": score_experience(
            posting.get("years_experience"),
            posting.get("no_experience_signal", False),
        ),
        "degree":              score_degree(posting.get("degree_required")),
        "tech_breadth":        score_tech_breadth(posting.get("tech_count", 0)),
        "language":            score_language_tone(posting),
        "salary_transparency": score_salary(posting),
    }
    return sum(components[k] * weights[k] for k in weights)


def categorise(posting, _gas):
    if posting.get("is_off_topic"):
        return "off_topic"
    if posting.get("is_bootcamp"):
        return "bootcamp"
    years = posting.get("years_experience")
    if years is not None and years >= 2:
        return "paradox"
    if (years is not None and years <= 1) or posting.get("no_experience_signal"):
        return "genuine_junior"
    return "ambiguous"


BASELINE_WEIGHTS = {
    "experience":          0.35,
    "degree":              0.20,
    "tech_breadth":        0.20,
    "language":            0.15,
    "salary_transparency": 0.10,
}

DELTAS = [-0.10, -0.05, +0.05, +0.10]


def perturb(weights, target, delta):
    new_weights = dict(weights)
    new_weights[target] += delta
    others = [k for k in weights if k != target]
    redistribute = -delta / len(others)
    for key in others:
        new_weights[key] += redistribute
    return new_weights


def compute_stats(postings, weights):
    gas_values = []
    categories = []
    for posting in postings:
        gas = compute_gas(posting, weights)
        gas_values.append(gas)
        categories.append(categorise(posting, gas))

    category_counter = Counter(categories)
    total = len(postings)

    return {
        "gas_mean":   round(statistics.mean(gas_values), 2),
        "gas_median": round(statistics.median(gas_values), 2),
        "categories": {
            k: round(v / total * 100, 2) for k, v in category_counter.items()
        },
        "total": total,
    }


def main():
    if not INPUT_FILE.exists():
        log.error("%s not found. Run nlp_pipeline.py first.", INPUT_FILE)
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        postings = json.load(f)

    log.info("Loaded %d postings for sensitivity analysis", len(postings))

    baseline = compute_stats(postings, BASELINE_WEIGHTS)
    log.info("Baseline:")
    log.info("  Weights:    %s", BASELINE_WEIGHTS)
    log.info("  GAS mean:   %s", baseline["gas_mean"])
    log.info("  Categories: %s", baseline["categories"])

    scenarios = {}

    log.info("Perturbations:")
    for target in BASELINE_WEIGHTS:
        log.info("  Perturbing '%s':", target)
        scenarios[target] = {}
        for delta in DELTAS:
            new_weights = perturb(BASELINE_WEIGHTS, target, delta)
            if any(w < 0 for w in new_weights.values()):
                continue
            stats = compute_stats(postings, new_weights)
            scenarios[target][f"{delta:+.2f}"] = stats
            log.info("    %s%+.2f: GAS=%s cats=%s",
                     target, delta, stats["gas_mean"], stats["categories"])

    all_shifts = []
    for target, perturbed in scenarios.items():
        for delta, stats in perturbed.items():
            for category in baseline["categories"]:
                base_pct = baseline["categories"].get(category, 0)
                new_pct = stats["categories"].get(category, 0)
                all_shifts.append({
                    "target": target,
                    "delta": delta,
                    "category": category,
                    "baseline": base_pct,
                    "new": new_pct,
                    "shift_abs": round(abs(new_pct - base_pct), 2),
                })

    max_shift = max(all_shifts, key=lambda x: x["shift_abs"])
    avg_shift = round(sum(s["shift_abs"] for s in all_shifts) / len(all_shifts), 2)

    log.info("Summary:")
    log.info("  Scenarios tested:    %d", sum(len(v) for v in scenarios.values()))
    log.info("  Max single shift:    %s%%", max_shift["shift_abs"])
    log.info("    (category: %s, perturbation: %s%s)",
             max_shift["category"], max_shift["target"], max_shift["delta"])
    log.info("  Average shift:       %s%%", avg_shift)

    output = {
        "baseline": {
            "weights": BASELINE_WEIGHTS,
            "stats": baseline,
        },
        "scenarios": scenarios,
        "summary": {
            "max_shift_percent": max_shift["shift_abs"],
            "max_shift_details": max_shift,
            "average_shift_percent": avg_shift,
            "n_scenarios": sum(len(v) for v in scenarios.values()),
        },
    }

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
