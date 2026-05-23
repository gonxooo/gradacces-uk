"""scoring.py - Graduate Accessibility Score and aggregate statistics."""

import json
import re
import logging
import statistics
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/processed.json")
SCORED_FILE = Path("data/scored.json")
STATS_FILE = Path("data/aggregate_stats.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scoring")


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
    if degree is None:
        return 100
    if degree == "bachelor":
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


INCLUSIVE_PATTERNS = [
    r"no\s+(?:prior\s+)?experience\s+required",
    r"training\s+(?:will\s+be\s+)?provided",
    r"mentorship",
    r"structured\s+(?:training|programme|program)",
    r"alongside\s+experienced\s+engineers",
    r"career\s+(?:starter|changer|journey)",
    r"supportive\s+(?:team|environment)",
    r"recent\s+graduate",
]

EXCLUSIONARY_PATTERNS = [
    r"hit\s+the\s+ground\s+running",
    r"proven\s+track\s+record",
    r"extensive\s+experience",
    r"significant\s+experience",
    r"deep\s+knowledge",
    r"expert(?:ise)?\s+in",
    r"mastery\s+of",
    r"\barchitect\b",
]


def score_language_tone(text):
    score = 70
    for pattern in INCLUSIVE_PATTERNS:
        if re.search(pattern, text, re.I):
            score += 8
    for pattern in EXCLUSIONARY_PATTERNS:
        if re.search(pattern, text, re.I):
            score -= 10
    return max(0, min(100, score))


def score_salary_transparency(posting):
    has_min = posting.get("salary_min") is not None
    has_max = posting.get("salary_max") is not None
    is_predicted = posting.get("salary_is_predicted") in (True, 1, "1")
    if has_min and has_max and not is_predicted:
        return 100
    if has_min or has_max:
        return 60
    return 0


WEIGHTS = {
    "experience":          0.35,
    "degree":              0.20,
    "tech_breadth":        0.20,
    "language":            0.15,
    "salary_transparency": 0.10,
}


def compute_gas(posting):
    category = posting.get("category", "ambiguous")

    if category == "bootcamp":
        return {"components": {}, "gas": 15.0, "tier": "bootcamp_disguised"}
    if category == "off_topic":
        return {"components": {}, "gas": 0.0, "tier": "off_topic"}

    title = posting.get("title") or ""
    description = posting.get("description_clean", "")
    combined_text = description + " " + title

    components = {
        "experience": score_experience(
            posting.get("years_experience"),
            posting.get("no_experience_signal", False),
        ),
        "degree":              score_degree(posting.get("degree_required")),
        "tech_breadth":        score_tech_breadth(posting.get("tech_count", 0)),
        "language":            score_language_tone(combined_text),
        "salary_transparency": score_salary_transparency(posting),
    }
    gas = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)

    if category == "paradox":
        tier = "paradox"
    elif gas >= 75:
        tier = "genuine_accessible"
    elif gas >= 55:
        tier = "ambiguous_moderate"
    else:
        tier = "difficult"

    return {"components": components, "gas": round(gas, 1), "tier": tier}


def safe_mean(values):
    return statistics.mean(values) if values else None


def compute_aggregates(scored):
    total = len(scored)
    if total == 0:
        return {}

    category_counter = Counter(p["category"] for p in scored)
    tier_counter = Counter(p["_scoring"]["tier"] for p in scored)

    real_roles = [
        p for p in scored
        if p["category"] in ("genuine_junior", "paradox", "ambiguous")
    ]
    n_real = len(real_roles)

    years_reported = [
        p["years_experience"] for p in real_roles
        if p.get("years_experience") is not None
    ]
    years_counter = Counter(years_reported)

    demanding = sum(
        1 for p in real_roles if (p.get("years_experience") or 0) >= 2
    )
    paradox_rate = (demanding / n_real * 100) if n_real else 0

    genuine_count = category_counter.get("genuine_junior", 0)
    bootcamp_count = category_counter.get("bootcamp", 0)
    genuine_rate = genuine_count / total * 100
    bootcamp_rate = bootcamp_count / total * 100

    tech_counter = Counter()
    for posting in real_roles:
        tech_counter.update(posting.get("technologies", []))

    region_counter = Counter()
    region_gas = {}
    for posting in real_roles:
        area = posting.get("location_area") or []
        if len(area) >= 2:
            region = area[1]
            region_counter[region] += 1
            region_gas.setdefault(region, []).append(
                posting["_scoring"]["gas"]
            )
    region_avg_gas = {
        region: round(safe_mean(values), 1)
        for region, values in region_gas.items()
        if len(values) >= 3
    }

    salaries_min = [p["salary_min"] for p in real_roles if p.get("salary_min")]
    salaries_max = [p["salary_max"] for p in real_roles if p.get("salary_max")]
    salary_mean_min = round(safe_mean(salaries_min), 0) if salaries_min else None
    salary_mean_max = round(safe_mean(salaries_max), 0) if salaries_max else None

    real_gas_values = [p["_scoring"]["gas"] for p in real_roles]
    gas_mean = round(safe_mean(real_gas_values), 1) if real_gas_values else 0
    gas_median = round(statistics.median(real_gas_values), 1) if real_gas_values else 0

    return {
        "total_postings": total,
        "category_distribution": dict(category_counter),
        "category_percentages": {
            k: round(v / total * 100, 1) for k, v in category_counter.items()
        },
        "real_software_roles": n_real,
        "gas_mean_real": gas_mean,
        "gas_median_real": gas_median,
        "genuine_junior_rate_percent": round(genuine_rate, 1),
        "bootcamp_disguised_rate_percent": round(bootcamp_rate, 1),
        "paradox_rate_percent": round(paradox_rate, 1),
        "paradox_numerator": demanding,
        "paradox_denominator": n_real,
        "years_experience_distribution": dict(sorted(years_counter.items())),
        "tier_distribution": dict(tier_counter),
        "top_20_technologies": tech_counter.most_common(20),
        "top_regions": region_counter.most_common(10),
        "region_avg_gas": region_avg_gas,
        "salary_mean_min": salary_mean_min,
        "salary_mean_max": salary_mean_max,
    }


def main():
    if not INPUT_FILE.exists():
        log.error("%s not found. Run nlp_pipeline.py first.", INPUT_FILE)
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        postings = json.load(f)
    log.info("Loaded %d processed postings", len(postings))

    scored = []
    for posting in postings:
        record = dict(posting)
        record["_scoring"] = compute_gas(posting)
        scored.append(record)

    stats = compute_aggregates(scored)

    log.info("Total postings analysed: %d", stats["total_postings"])
    log.info("Breakdown:")
    for category, percentage in sorted(
        stats["category_percentages"].items(), key=lambda x: -x[1]
    ):
        count = stats["category_distribution"][category]
        log.info("  %-18s: %4d (%5.1f%%)", category, count, percentage)

    log.info("Headline figures:")
    log.info("  Genuine graduate-accessible roles:      %s%%", stats["genuine_junior_rate_percent"])
    log.info("  Training programmes disguised as jobs:  %s%%", stats["bootcamp_disguised_rate_percent"])
    log.info("  Graduate paradox rate (real roles):     %s%%", stats["paradox_rate_percent"])
    log.info("  Mean GAS (real roles only):             %s", stats["gas_mean_real"])
    log.info("  Median GAS (real roles only):           %s", stats["gas_median_real"])
    if stats.get("salary_mean_min"):
        log.info("  Mean salary min:     GBP %s", f"{stats['salary_mean_min']:,.0f}")
        log.info("  Mean salary max:     GBP %s", f"{stats['salary_mean_max']:,.0f}")

    log.info("Top 5 technologies:")
    for tech, count in stats["top_20_technologies"][:5]:
        log.info("  %-14s %d", tech, count)

    SCORED_FILE.parent.mkdir(exist_ok=True)
    with SCORED_FILE.open("w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    with STATS_FILE.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", SCORED_FILE)
    log.info("Saved -> %s", STATS_FILE)


if __name__ == "__main__":
    main()
