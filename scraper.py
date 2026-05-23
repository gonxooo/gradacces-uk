"""scraper.py - Adzuna API collector for UK junior software postings."""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import requests

APP_ID = os.getenv("ADZUNA_APP_ID", "YOUR_APP_ID_HERE")
APP_KEY = os.getenv("ADZUNA_APP_KEY", "YOUR_APP_KEY_HERE")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/gb/search/{page}"

CATEGORIES = ["it-jobs", "engineering-jobs", "scientific-qa-jobs"]

SEARCH_TERMS = [
    "graduate software engineer", "junior software engineer",
    "graduate developer", "junior developer",
    "graduate programmer", "trainee software engineer",
    "trainee developer", "graduate web developer",
    "junior web developer", "graduate backend developer",
    "junior backend developer", "graduate frontend developer",
    "junior frontend developer", "graduate full stack",
    "junior full stack", "entry level software",
    "entry level developer", "graduate python developer",
    "graduate java developer", "junior python developer",
    "junior java developer", "graduate .net developer",
    "graduate software engineer no experience",
    "graduate scheme software", "software graduate programme",
    "software engineer intern", "junior c# developer",
    "junior javascript developer", "junior typescript developer",
    "first software engineering job", "placement software developer",
    "associate software engineer", "software developer graduate",
    "apprentice software",
]

PAGES_PER_TERM = 5
RESULTS_PER_PAGE = 50
MAX_DAYS_OLD = 365
REQUEST_DELAY_SECONDS = 4.0

OUTPUT_DIR = Path("data")
OUTPUT_RAW = OUTPUT_DIR / "raw_postings.json"
OUTPUT_FILTERED = OUTPUT_DIR / "raw_postings_filtered.json"

JUNIOR_KEYWORDS = [
    "junior", "graduate", "entry level", "entry-level", "trainee",
    "apprentice", "intern", "internship", "placement", "early career",
    "new grad", "associate",
]

SENIOR_EXCLUDE = [
    " senior", "senior ", "snr ", "sr.", "sr ",
    "lead ", " lead", "principal", "staff engineer",
    "head of", "director", " manager", "manager ",
    "architect", "consultant", "expert",
    " iii", " iv", "level 5", "level 6",
    "chief", "vp ", "cto", "founder",
    "5+ years", "7+ years", "10+ years",
    "tech lead", "team lead",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scraper")


def fetch_page(term, page, category):
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": term,
        "category": category,
        "max_days_old": MAX_DAYS_OLD,
        "full_time": 1,
    }
    try:
        response = requests.get(
            BASE_URL.format(page=page), params=params, timeout=20
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        log.error("Request failed for '%s' page %d (%s): %s",
                  term, page, category, exc)
        return None


def normalise(raw, term, category):
    return {
        "id": raw.get("id"),
        "title": (raw.get("title") or "").strip(),
        "description": (raw.get("description") or "").strip(),
        "company": (raw.get("company") or {}).get("display_name"),
        "location_area": (raw.get("location") or {}).get("area", []),
        "location_display": (raw.get("location") or {}).get("display_name"),
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "salary_min": raw.get("salary_min"),
        "salary_max": raw.get("salary_max"),
        "salary_is_predicted": raw.get("salary_is_predicted"),
        "category_tag": (raw.get("category") or {}).get("tag"),
        "category_label": (raw.get("category") or {}).get("label"),
        "contract_type": raw.get("contract_type"),
        "contract_time": raw.get("contract_time"),
        "created": raw.get("created"),
        "adzuna_url": raw.get("redirect_url"),
        "_search_term": term,
        "_search_category": category,
        "_collected_at": datetime.utcnow().isoformat() + "Z",
    }


def is_junior(posting):
    title = (posting.get("title") or "").lower()
    if not title:
        return False, "empty title"
    if not any(keyword in title for keyword in JUNIOR_KEYWORDS):
        return False, "no junior keyword"
    for excluded in SENIOR_EXCLUDE:
        if excluded in title:
            return False, f"excluded by '{excluded.strip()}'"
    return True, "accepted"


def collect():
    if APP_ID == "YOUR_APP_ID_HERE":
        log.error("Set ADZUNA_APP_ID and ADZUNA_APP_KEY before running.")
        return

    seen_ids = set()
    all_postings = []

    total_calls = len(SEARCH_TERMS) * len(CATEGORIES) * PAGES_PER_TERM
    call_index = 0

    log.info("Starting collection: %d terms x %d categories x %d pages",
             len(SEARCH_TERMS), len(CATEGORIES), PAGES_PER_TERM)

    for category in CATEGORIES:
        log.info("Category: %s", category)

        for term in SEARCH_TERMS:
            log.info("  '%s'", term)

            for page in range(1, PAGES_PER_TERM + 1):
                call_index += 1
                log.info("    [%d/%d] page %d", call_index, total_calls, page)

                response = fetch_page(term, page, category)
                if response is None:
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                results = response.get("results", [])
                if not results:
                    break

                added = 0
                for raw in results:
                    posting_id = raw.get("id")
                    if posting_id and posting_id not in seen_ids:
                        seen_ids.add(posting_id)
                        all_postings.append(normalise(raw, term, category))
                        added += 1

                log.info("    added %d new (total: %d)", added, len(all_postings))
                time.sleep(REQUEST_DELAY_SECONDS)

    kept = []
    rejection_reasons = {}
    for posting in all_postings:
        accepted, reason = is_junior(posting)
        if accepted:
            kept.append(posting)
        else:
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    log.info("Raw postings: %d", len(all_postings))
    log.info("After filter: %d", len(kept))
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", reason, count)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with OUTPUT_RAW.open("w", encoding="utf-8") as f:
        json.dump(all_postings, f, ensure_ascii=False, indent=2)
    with OUTPUT_FILTERED.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    log.info("Saved -> %s", OUTPUT_RAW)
    log.info("Saved -> %s", OUTPUT_FILTERED)


if __name__ == "__main__":
    collect()
