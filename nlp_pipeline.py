"""nlp_pipeline.py - Feature extraction and five-category classification."""

import json
import re
import logging
from pathlib import Path
from html import unescape
from collections import Counter

INPUT_FILE = Path("data/raw_postings_filtered.json")
OUTPUT_FILE = Path("data/processed.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nlp")


TECH_DICTIONARY = {
    "javascript": [r"\bjavascript\b"],
    "typescript": [r"\btypescript\b"],
    "python":     [r"\bpython\b"],
    "java":       [r"\bjava\b(?!script)"],
    "c#":         [r"c#", r"\.net", r"\bcsharp\b"],
    "c++":        [r"c\+\+", r"\bcpp\b"],
    "c":          [r"\bc/c\+\+", r"\bc\b\s+(?:and|or|language|programming)"],
    "go":         [r"\bgolang\b"],
    "rust":       [r"\brust\b"],
    "php":        [r"\bphp\b"],
    "ruby":       [r"\bruby\b"],
    "kotlin":     [r"\bkotlin\b"],
    "swift":      [r"\bswift\b"],
    "scala":      [r"\bscala\b"],
    "dart":       [r"\bdart\b"],
    "sql":        [r"\bsql\b"],
    "react":      [r"\breact\.?js\b", r"\breact\b"],
    "vue":        [r"\bvue\.?js\b", r"\bvue\b"],
    "angular":    [r"\bangular\b"],
    "svelte":     [r"\bsvelte\b"],
    "nextjs":     [r"\bnext\.?js\b"],
    "html":       [r"\bhtml5?\b"],
    "css":        [r"\bcss3?\b", r"\bsass\b", r"\btailwind\b"],
    "nodejs":     [r"\bnode\.?js\b", r"\bnodejs\b"],
    "express":    [r"\bexpress\.?js\b"],
    "spring":     [r"\bspring\s+boot\b", r"\bspring\b"],
    "django":     [r"\bdjango\b"],
    "flask":      [r"\bflask\b"],
    "laravel":    [r"\blaravel\b"],
    "android":    [r"\bandroid\b"],
    "ios":        [r"\bios\b"],
    "flutter":    [r"\bflutter\b"],
    "postgresql": [r"\bpostgres(?:ql)?\b"],
    "mysql":      [r"\bmysql\b"],
    "mongodb":    [r"\bmongo(?:db)?\b"],
    "redis":      [r"\bredis\b"],
    "aws":        [r"\baws\b", r"amazon\s+web\s+services"],
    "azure":      [r"\bazure\b"],
    "gcp":        [r"\bgcp\b", r"google\s+cloud"],
    "docker":     [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "git":        [r"\bgit(?:hub|lab)?\b"],
    "linux":      [r"\blinux\b", r"\bunix\b"],
    "rest":       [r"\brest(?:ful)?\s+api\b"],
    "graphql":    [r"\bgraphql\b"],
    "agile":      [r"\bagile\b", r"\bscrum\b"],
    "ci/cd":      [r"ci/cd", r"continuous\s+integration"],
    "embedded":   [r"\bembedded\b"],
    "wireless":   [r"\bwireless\b", r"\b5g\b", r"\blte\b"],
}

_COMPILED_TECH = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in TECH_DICTIONARY.items()
}


BOOTCAMP_SIGNALS = [
    r"skills\s+shortages\s+in\s+the\s+it\s+sector",
    r"industry\s+recognised\s+certifications",
    r"career\s+seekers\s+and\s+career\s+changers",
    r"invest\s+(?:some\s+)?time\s+and\s+money",
    r"training\s+and\s+placement\s+programme",
    r"skills\s+bootcamp",
    r"govt[- ]funded",
    r"government[- ]funded\s+(?:skills\s+)?bootcamp",
    r"fully[- ]funded\s+(?:cyber\s+security\s+)?course",
    r"ncfe\s+certificate",
    r"(?:pay|charge|fee|cost|invest|£\s*\d+)\s+(?:for\s+)?(?:the\s+)?(?:training|course|programme)",
    r"training\s+programme\s+(?:is\s+)?made\s+for\s+you",
    r"build\s+the\s+foundations\s+for\s+a\s+new\s+career",
    r"help\s+place\s+graduates\s+from\s+this\s+programme",
]

BOOTCAMP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in BOOTCAMP_SIGNALS]


YEARS_PATTERNS = [
    re.compile(r"seniority\s*:?\s*[^.]*?\(\s*(\d+)\s*[-\u2013]\s*\d+\s*years?", re.I),
    re.compile(
        r"(\d+)\s*[-\u2013]\s*\d+\s*years?\b[^.]{0,100}?"
        r"(?:experience|commercial|industry|professional|working|software\s+development)",
        re.I | re.DOTALL,
    ),
    re.compile(
        r"(\d+)\s*[-\u2013]\s*\d+\s*years?\b\s*"
        r"(?:commercial|industry|professional|relevant)",
        re.I,
    ),
    re.compile(
        r"(\d+)\s*\+?\s*years?\b[^.]{0,60}?"
        r"(?:experience|commercial|industry|professional|working|hands[- ]on|software)",
        re.I | re.DOTALL,
    ),
    re.compile(r"(?:minimum|at\s+least|minimum\s+of)\s+(\d+)\s*\+?\s*years?\b", re.I),
    re.compile(
        r"(?:experience|commercial)\s+(?:of|with|in|using)\s+"
        r"(?:at\s+least\s+)?(\d+)\s*\+?\s*years?\b",
        re.I,
    ),
    re.compile(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?pqe\b", re.I),
    re.compile(
        r"\b(one|two|three|four|five)\s+years?\b[^.]{0,40}?"
        r"(?:experience|commercial)",
        re.I,
    ),
]

SPELLED = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


NO_EXP_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"no\s+(?:prior\s+)?(?:commercial\s+)?experience\s+(?:required|needed|necessary|essential)",
        r"no\s+previous\s+experience",
        r"fresh\s+graduate",
        r"recent\s+graduate",
        r"new\s+graduate",
        r"training\s+(?:will\s+be\s+)?provided",
        r"full\s+training",
        r"we\s+will\s+train",
        r"structured\s+(?:training|programme|program)",
        r"early[- ]career",
        r"from\s+university",
        r"supportive\s+environment",
        r"entry[- ]level\s+position",
        r"career\s+(?:journey|path|starter|changer)",
        r"kick[- ]?start\s+your\s+career",
        r"alongside\s+(?:senior|experienced)\s+engineers?",
        r"mentorship",
        r"learn\s+(?:on\s+the\s+job|and\s+grow)",
        r"investment\s+in\s+(?:your|their)",
        r"curiosity\s+and\s+motivation\s+to\s+learn",
    ]
]


DEGREE_PATTERNS = {
    "phd":      re.compile(r"\b(?:ph\.?\s*d|doctorate|doctoral)\b", re.I),
    "master":   re.compile(r"\b(?:master(?:[\'\u2019]s)?|msc|m\.sc\b|mba)\b", re.I),
    "bachelor": re.compile(
        r"\b(?:bachelor(?:[\'\u2019]s)?|bsc|b\.sc\b|beng|honours\s+degree|"
        r"degree\s+in|degree\s+level|university\s+degree|relevant\s+degree)\b",
        re.I,
    ),
}


OFF_TOPIC_PATTERNS = [re.compile(p, re.I) for p in [
    r"customer\s+service",
    r"account(?:s|ing|ant|ancy)\b",
    r"\bsales\b",
    r"\bteacher\b",
    r"recruit(?:er|ment)",
    r"\bmarketing\b",
    r"\bhr\b|human\s+resources",
]]

SOFTWARE_PATTERNS = [re.compile(p, re.I) for p in [
    r"software\s+(?:engineer|developer|development)",
    r"\bdeveloper\b",
    r"programm(?:er|ing)",
    r"full[- ]stack|backend|front[- ]end",
    r"\bcoding\b",
]]


def clean_text(text):
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("â€¦", "...").replace("Ã¢â‚¬â€œ", "-")
    return re.sub(r"\s+", " ", text).strip()


def is_bootcamp(text):
    matches = sum(1 for pattern in BOOTCAMP_PATTERNS if pattern.search(text))
    return matches >= 2


def extract_years_experience(text):
    found = []
    for pattern in YEARS_PATTERNS:
        for match in pattern.finditer(text):
            try:
                raw = match.group(1).lower()
                value = SPELLED[raw] if raw in SPELLED else int(raw)
                if 0 <= value <= 20:
                    found.append(value)
            except (ValueError, IndexError):
                continue
    return min(found) if found else None


def mentions_no_experience(text):
    return any(pattern.search(text) for pattern in NO_EXP_PATTERNS)


def extract_technologies(text):
    found = set()
    for name, patterns in _COMPILED_TECH.items():
        for pattern in patterns:
            if pattern.search(text):
                found.add(name)
                break
    return sorted(found)


def extract_degree(text):
    if DEGREE_PATTERNS["phd"].search(text):
        return "phd"
    if DEGREE_PATTERNS["master"].search(text):
        return "master"
    if DEGREE_PATTERNS["bachelor"].search(text):
        return "bachelor"
    return None


def is_off_topic(title, description):
    combined = f"{title} {description}".lower()
    off_count = sum(1 for p in OFF_TOPIC_PATTERNS if p.search(combined))
    soft_count = sum(1 for p in SOFTWARE_PATTERNS if p.search(combined))
    return off_count > 0 and soft_count == 0


def classify(posting):
    title = posting.get("title") or ""
    description = posting.get("description_clean", "")
    combined = f"{title} . {description}"

    if posting["is_off_topic"]:
        return "off_topic"
    if is_bootcamp(combined):
        return "bootcamp"

    years = posting["years_experience"]
    if years is not None and years >= 2:
        return "paradox"
    if years is not None and years <= 1:
        return "genuine_junior"
    if posting["no_experience_signal"]:
        return "genuine_junior"
    return "ambiguous"


def process(posting):
    title = posting.get("title") or ""
    description = clean_text(posting.get("description", ""))
    combined = f"{title} . {description}"

    enriched = dict(posting)
    enriched["description_clean"]    = description
    enriched["description_length"]   = len(description)
    enriched["years_experience"]     = extract_years_experience(combined)
    enriched["no_experience_signal"] = mentions_no_experience(combined)
    enriched["technologies"]         = extract_technologies(combined)
    enriched["tech_count"]           = len(enriched["technologies"])
    enriched["degree_required"]      = extract_degree(combined)
    enriched["is_off_topic"]         = is_off_topic(title, description)
    enriched["is_bootcamp"]          = is_bootcamp(combined)
    enriched["category"]             = classify(enriched)
    return enriched


def main():
    if not INPUT_FILE.exists():
        log.error("%s not found", INPUT_FILE)
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        postings = json.load(f)
    log.info("Loaded %d strict-junior postings", len(postings))

    processed = [process(p) for p in postings]

    category_counts = Counter(p["category"] for p in processed)
    total = len(processed)

    log.info("Classification:")
    for category in ["bootcamp", "off_topic", "paradox", "genuine_junior", "ambiguous"]:
        count = category_counts.get(category, 0)
        percentage = count / total * 100 if total else 0
        log.info("  %-18s: %4d  (%5.1f%%)", category, count, percentage)

    n_years = sum(1 for p in processed if p["years_experience"] is not None)
    n_no_exp = sum(1 for p in processed if p["no_experience_signal"])
    n_tech = sum(1 for p in processed if p["technologies"])
    n_degree = sum(1 for p in processed if p["degree_required"])
    log.info("Extraction stats:")
    log.info("  With stated years of experience:    %4d (%.1f%%)", n_years, n_years / total * 100)
    log.info("  With 'no experience' signal:        %4d (%.1f%%)", n_no_exp, n_no_exp / total * 100)
    log.info("  With >=1 recognised technology:     %4d (%.1f%%)", n_tech, n_tech / total * 100)
    log.info("  With degree requirement:            %4d (%.1f%%)", n_degree, n_degree / total * 100)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    log.info("Saved -> %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
