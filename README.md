# GradAccess UK

An empirical NLP-based analysis of UK junior software engineering job postings, with a public diagnostic tool for graduates.

This repository contains the full pipeline, dataset and interactive web dashboard that accompany the BSc (Hons) Software Engineering dissertation submitted to York St John University under module COM6016M.

## Findings (summary)

From a corpus of 335 UK postings labelled "junior", "graduate" or "trainee", collected through the Adzuna public REST API:

- **20.9%** are paid training programmes (bootcamps), not employment offers
- **17.6%** are unambiguously accessible to recent graduates
- **8.4%** of real software roles exhibit the "graduate paradox" by demanding 2+ years of experience
- The corpus contains more bootcamp listings (70) than genuine entry-level jobs (59)

A Logistic Regression baseline on TF-IDF features achieves 93.4% accuracy (macro F1 0.869) in five-fold cross-validation against the rule-based labels, supporting the internal consistency of the classification framework.

## Repository structure

```
.
├── scraper.py                # Stage 1: Adzuna API collection
├── nlp_pipeline.py           # Stage 2: feature extraction
├── scoring.py                # Stage 3: GAS scoring and classification
├── ml_baseline.py            # Stage 4a: ML validation baseline
├── sensitivity_analysis.py   # Stage 4b: GAS weight sensitivity test
├── data/                     # Inputs, intermediate JSON and outputs
└── dashboard/                # React + Vite interactive dashboard
```

## Installation

### Python pipeline

```bash
pip install requests scikit-learn numpy
```

Python 3.10 or higher is required.

### Dashboard

```bash
cd dashboard
npm install
```

Node 18 or higher is required.

## Running the pipeline

The pipeline is fully reproducible from a free-tier Adzuna account. Set your credentials as environment variables:

```bash
# Linux / macOS
export ADZUNA_APP_ID=your_id
export ADZUNA_APP_KEY=your_key

# Windows PowerShell
$env:ADZUNA_APP_ID="your_id"
$env:ADZUNA_APP_KEY="your_key"
```

Then run the four stages in order:

```bash
python scraper.py
python nlp_pipeline.py
python scoring.py
python ml_baseline.py
python sensitivity_analysis.py
```

The full collection takes around 35 minutes; the remaining stages run in under 2 minutes.

## Running the dashboard

```bash
cd dashboard
npm run dev
```

Then open http://localhost:5173.

The dashboard has two tabs:

- **Overview** displays the corpus-level findings (category distribution, headline rates, top technologies, regional breakdown, sortable explorable table)
- **Live Analyser** lets a user paste any job posting and receive an instant classification with the same logic that produced the corpus-level findings

## Methodology

A summary of the methodology is provided here. The full account is in Chapter 3 of the accompanying dissertation.

**Strict junior filter.** Postings are retained only if the title contains a junior keyword (junior, graduate, entry level, trainee, apprentice, intern, placement, early career, new grad, associate) and contains none of the senior-exclusion markers (senior, lead, principal, staff, head, director, manager, architect, consultant, "5+ years", and so on).

**Five-category taxonomy.** Each retained posting is assigned to exactly one of:

- **bootcamp** – matches at least 2 of 14 distinctive training-provider patterns
- **off_topic** – not actually a software role
- **paradox** – demands 2+ years of prior experience despite the junior label
- **genuine_junior** – states no experience required or 1 year or less
- **ambiguous** – none of the above

**Graduate Accessibility Score (GAS).** A 0-100 score for each non-bootcamp non-off-topic posting:

```
GAS = 0.35 * S(experience)
    + 0.20 * S(degree)
    + 0.20 * S(tech_breadth)
    + 0.15 * S(language)
    + 0.10 * S(salary_transparency)
```

Weights are validated through a sensitivity analysis of 20 perturbations of +/-5% and +/-10%, which shows zero shift in category distribution.

## Reproducibility

This repository depends only on public APIs and open-source libraries. No proprietary dataset is used and no commercial subscriptions are required. An independent researcher can replicate the entire study from scratch using a free Adzuna account.

## Licence

This project is released under the MIT License.

## Citation

If you use this code or dataset, please cite:

> Ponce, G. (2026). *Beyond the Graduate Label: An Empirical NLP-Based Analysis of Requirements Inflation and Bootcamp Misclassification in UK Junior Software Engineering Postings*. BSc dissertation, York St John University.
