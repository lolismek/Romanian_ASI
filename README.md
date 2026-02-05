# Romanian ASI Benchmark - Project Status

**Last Updated:** 2026-02-05

## Overview

The Romanian ASI (Affective State Identification) Benchmark extracts natural "I feel [state]" expressions from Romanian text corpora, following the MASIVE paper methodology. The goal is to create a dataset of how Romanian speakers naturally express their emotional states.

---

## Current Data Collection Status

### Summary

| Source | Records Processed | ASI Candidates | Status |
|--------|-------------------|----------------|--------|
| Small Datasets (5 sources) | 79,658 | 4,282 | ✅ Complete |
| FULG (web crawl) | 405,000 | 21,184 | ⏸️ Paused (resumable) |
| Filmot/YouTube | - | - | ❌ Blocked |
| **Total** | **484,658** | **25,466** | |

### 1. Small Datasets (Complete)

**Source:** 5 Romanian NLP datasets merged into `data/merged_corpus.jsonl`

| Dataset | Records | ASI Matches | Match Rate |
|---------|---------|-------------|------------|
| RoSent | 28,946 | 2,481 | 8.6% |
| LaRoSeDa | 15,000 | 1,017 | 6.8% |
| PoPreRo | 28,107 | 467 | 1.7% |
| RED v2 | 5,449 | 186 | 3.4% |
| RED v1 | 2,156 | 131 | 6.1% |

**Output:** `data/asi_candidates.jsonl` (4,282 samples)

### 2. FULG Dataset (Paused - Resumable)

**Source:** FULG web crawl (150B tokens, 289GB) via HuggingFace streaming

**Current Progress:**
- Records streamed: 405,000
- Candidates extracted: 21,184
- Match rate: 7.0%
- Unique domains: 2,792

**Resume command:**
```bash
python -m scripts.fulg.extract_candidates --resume
```

**Output:** `data/fulg_asi_candidates.jsonl` (21,184 samples)

### 3. Filmot/YouTube (Blocked)

**Status:** ❌ Blocked by Cloudflare bot protection

Filmot.com (YouTube subtitle search) uses aggressive bot detection that blocks Playwright browsers even with stealth mode. The pattern matching and transcript fetching components work, but search automation is blocked.

**Potential workarounds:**
1. Manual filmot search → export video IDs → use transcript phase
2. yt-dlp channel crawling for known Romanian YouTube channels
3. Different automation (undetected-chromedriver, Puppeteer)

---

## Extracted Data Analysis

### Emotion Distribution (Combined: 25,466 samples)

| Emotion | Small Datasets | FULG | Total | % |
|---------|----------------|------|-------|---|
| Joy | 1,730 | 7,967 | 9,697 | 31% |
| Trust | 864 | 7,170 | 8,034 | 26% |
| Sadness | 803 | 4,966 | 5,769 | 19% |
| Anticipation | 514 | 3,896 | 4,410 | 14% |
| Fear | 295 | 2,957 | 3,252 | 11% |
| Surprise | 601 | 1,285 | 1,886 | 6% |
| Anger | 214 | 1,201 | 1,415 | 5% |
| Disgust | 38 | 293 | 331 | 1% |

### Pattern Usage

| Pattern | Count | % | Example |
|---------|-------|---|---------|
| sunt_adj_present | 13,352 | 54% | "sunt fericit" |
| mie_short | 2,692 | 11% | "mi-e frică" |
| am_noun_present | 1,517 | 6% | "am teamă" |
| am_fost_adj_perfect | 2,254 | 9% | "am fost surprins" |
| ma_simt_present | 1,197 | 5% | "mă simt bine" |
| eram_adj_imperfect | 1,156 | 5% | "eram trist" |
| Other patterns | ~2,500 | 10% | Various |

**Note:** Primary "mă simt" patterns are ~9% of matches. The "sunt" (I am) pattern dominates, which is typical for written Romanian.

### Top Seed Words (FULG)

| Word | Count | Emotion |
|------|-------|---------|
| bine | 2,104 | joy, trust |
| sigur/sigură | 2,501 | trust |
| dor | 1,268 | sadness, anticipation |
| frică | 648 | fear |
| curios/curioasă | 1,030 | anticipation |
| fericit/fericită | 705 | joy |
| mulțumit/mulțumită | 782 | joy |

### Source Categories (FULG only)

| Category | Count | % |
|----------|-------|---|
| other | 11,311 | 58% |
| blog | 7,137 | 36% |
| wiki | 517 | 3% |
| news | 346 | 2% |
| forum | 200 | 1% |
| social | 51 | <1% |

### Context Quality (FULG)

| Metric | Value |
|--------|-------|
| Median context length | 585 chars |
| Average context length | 627 chars |
| Under 500 chars | 33.6% |
| 500-1000 chars | 57.0% |
| Over 1000 chars | 6.4% |

---

## Project Structure

```
Romanian_ASI/
├── CLAUDE.md                 # Development guide for Claude Code
├── ROMANIAN_ASI_PLAN.md      # Original research plan
├── PROJECT_STATUS.md         # This file
│
├── data/
│   ├── merged_corpus.jsonl           # 79K records from 5 datasets
│   ├── asi_candidates.jsonl          # 4,282 samples (small datasets)
│   ├── asi_candidates.stats.json
│   ├── fulg_asi_candidates.jsonl     # 21,184 samples (FULG)
│   ├── fulg_extraction_checkpoint.json  # Resume point
│   ├── fulg_extraction_analysis.json # Detailed statistics
│   └── emotion_seed.json             # 511 curated affective words
│
├── scripts/
│   ├── ro_asi/                       # Core extraction pipeline
│   │   ├── pattern_matcher.py        # 18 Romanian "I feel" patterns
│   │   ├── curated_affective_states.py  # 511 emotion words
│   │   ├── extract_candidates.py     # Small dataset extraction
│   │   ├── merge_datasets.py         # Dataset merger
│   │   └── load_roemolex.py          # RoEmoLex lexicon loader
│   │
│   ├── fulg/                         # FULG streaming extraction
│   │   └── extract_candidates.py     # HuggingFace streaming + context extraction
│   │
│   └── filmot/                       # YouTube extraction (blocked)
│       ├── searcher.py               # Playwright-based search
│       ├── transcript_fetcher.py     # youtube-transcript-api wrapper
│       └── extract_candidates.py     # 3-phase pipeline
│
└── small_datasets/                   # Source datasets
    ├── LaRoSeDa/
    ├── PoPreRo/
    ├── RED/
    └── RoSent/
```

---

## Technical Details

### Pattern Matching

18 Romanian patterns organized into two categories:

**Primary (adjectives with "mă simt"):**
- `mă simt [adj]` - present
- `m-am simțit [adj]` - perfect
- `mă simțeam [adj]` - imperfect

**Secondary (nouns with "am", "mi-e"):**
- `sunt [adj]` - present (most common)
- `mi-e [noun]` - dative short form
- `am [noun]` - have + emotion noun
- `îmi este [noun]` - dative formal

### Emotion Lexicon

511 manually curated Romanian affective words:
- 348 adjectives (fericit/fericită, trist/tristă, etc.)
- 150 emotion nouns (frică, bucurie, tristețe, etc.)
- 15 state adverbs (bine, rău, groaznic, etc.)

Each word mapped to Plutchik's 8 basic emotions.

### Diacritics Handling

Romanian has 5 special characters (ă, â, î, ș, ț) that are often omitted in informal text. The pattern matcher normalizes both text and patterns to handle:
- `mă simt` = `ma simt`
- `frică` = `frica`
- `mulțumit` = `multumit`

---

## Commands

```bash
# Activate environment
source venv/bin/activate

# Small datasets extraction (complete)
python -m scripts.ro_asi.extract_candidates

# FULG extraction (resume from checkpoint)
python -m scripts.fulg.extract_candidates --resume

# FULG extraction (fresh start with limits)
python -m scripts.fulg.extract_candidates --max-samples 50000

# Test pattern matcher
python -m scripts.ro_asi.pattern_matcher
```

---

## Next Steps

1. **Resume FULG extraction** to reach 50K samples target
2. **Explore YouTube alternatives** (yt-dlp channel crawling, manual filmot export)
3. **Quality review** of extracted samples
4. **Annotation** for benchmark validation
5. **Train/test split** for final benchmark

---

## Files Summary

| File | Size | Description |
|------|------|-------------|
| `merged_corpus.jsonl` | 32 MB | 79,658 source records |
| `asi_candidates.jsonl` | 4 MB | 4,282 small dataset samples |
| `fulg_asi_candidates.jsonl` | 37 MB | 21,184 FULG samples |
| `emotion_seed.json` | 40 KB | 511 curated emotion words |
| `fulg_extraction_analysis.json` | 12 KB | Detailed FULG statistics |
