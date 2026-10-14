# The Multi-Tenant Document Index Factory — Design Document

## 0. Introduction

Business-critical knowledge sits in places an AI platform can't reach at scale:

- **Document vault:** tens of thousands of client and operational documents, many of them scans with no text layer.
- **SharePoint policy libraries:** drafts, current and superseded versions side by side.
- **Local and team drives:** near-duplicates of both, with unclear ownership.

The target is **one shared, governed retrieval index**. Several business domains (tenants) write into it, a central AI platform with about 1,000 users reads from it, and every user only sees what they're entitled to see.

This document answers three questions:

1. **Task 1:** When is a parsed page fit to enter the index, and how should it be chunked?
2. **Task 2:** Which framework, technologies and metadata carry documents from source to index?
3. **Task 3:** How do several tenants share one physical index without seeing each other's data?

**Evidence.** Task 1's thresholds were set on a 48-page slice of ParseBench covering scans, multi-column, simple text, table, chart and layout pages. Each page was parsed the cheap way: the PDF's own text layer, with OCR as the fallback. The result was scored against ParseBench's ground-truth rules.

---

## 1. Task 1 — Defining "fit for retrieval"

### 1.1 Principle

An index is fit for retrieval when a question that the documents can answer returns the passage that answers it, with the words and numbers as written. Two consequences follow:

- **Missing, invented or wrong content blocks a page.** Retrieval can't find what isn't there, and a wrong number that *is* found becomes a confident wrong answer.
- **Lost structure only degrades a page.** The right words are still in the chunk; it just ranks lower or carries noise.

Errors cost different amounts. A wrongly blocked page costs a reviewer a few minutes. A wrongly accepted page costs a user a wrong answer, and nobody notices. So the blocking checks favour **precision**, to keep the review queue worth reading. **Recall** comes from cheap automatic re-parsing, not from sending more pages to people.

### 1.2 The five ParseBench dimensions

| Dimension | Decision | Why |
|---|---|---|
| **Content faithfulness** (text present, correct, in order, no invented text) | **Block** | Missing or garbled text can't be retrieved, and wrong digits produce wrong answers. Reading order is the exception: it only degrades, because the words are still there. |
| **Tables** | **Block** if a table holds numbers but nothing usable was extracted; otherwise **degrade** | A numeric table flattened into loose numbers has lost which value belongs to which row. Imperfect structure with correct cells is still answerable. |
| **Charts** | **Degrade** | Chart values are invisible to a text index, but the caption and surrounding text are still there. Charts are flagged and routed to enrichment. |
| **Semantic formatting** (headings, strikethrough, emphasis) | **Degrade** | Headings decide *how* a page is chunked (§1.4), not *whether* it's indexed. Struck-through text is removed and kept as version metadata. Emphasis doesn't matter. |
| **Visual grounding** (layout, bounding boxes) | **Irrelevant** for retrieval | Boxes help a viewer highlight a source passage. The page number is enough to cite it. |

### 1.3 The gate

Each page gets one outcome. The most severe one applies.

| Outcome | When | Then |
|---|---|---|
| **Re-parse** | A different parser or setting is likely to fix it | Try the next parser in the ladder, then gate the page again |
| **Quarantine** | The source itself is poor, so re-parsing won't help | Not indexed; goes to the owning domain's review queue |
| **Reject** | Nothing to index (blank page, exact duplicate) | Logged only |
| **Accept with flags** | Only degrading problems | Indexed; the flags become chunk metadata |
| **Accept** | No problems | Indexed |

**Parser ladder, cheapest first:** PDF text layer → second PDF engine → OCR with the detected language → layout-aware parser → vision-language model.

**Thresholds.** The gate can't see ground truth in production, so each blocking dimension is enforced through signals it *can* compute. The thresholds come from our test results.

| Dimension | Gate signal and threshold | Reasoning from the test slice |
|---|---|---|
| Content faithfulness | **Re-parse** when:<br>• the page yields < 50 characters<br>• the text layer contains < 80% of the words OCR reads confidently<br>• > 20% of the page's ink isn't covered by extracted text | Every page that tripped these thresholds was genuinely bad. Two cases show why re-parsing works:<br>• a broken text layer scored 0% of required words, while OCR on the same page scored 83%<br>• one PDF returned nothing from the first engine, and a second engine read it fully |
| Content faithfulness | **Quarantine** when:<br>• mean OCR confidence < 80<br>• > 20% of words are in no dictionary | All scans below confidence 80 were bad. Between 80 and 94, only two in three were bad, which is too noisy for a person's queue. Clean pages had 0–10% unknown words (names, jargon). |
| Tables | **Re-parse** with a layout-aware parser when a page has rows of numbers but no table was extracted. **Quarantine** if totals don't add up. | Table similarity was ≥ 0.99 whenever a table was extracted (one exception at 0.51) and 0 when it wasn't, so detection mattered more than tuning. One table went undetected just below the threshold. For numeric-heavy sources, run the table parser always instead of trying to predict when it's needed. |
| Charts, formatting, visual grounding | No blocking threshold; flag only | Plain-text output scored 0 on chart values and about 0 on headings across the slice. That's a routing and chunking decision, not a quality failure. |

A page counts as **bad** when fewer than 90% of required words are found, fewer than 90% of digits are correct, or table similarity is below 0.70. The 0.70 sits in the empty gap between tables extracted well and tables not extracted.

**Result on the slice:** the gate blocked 12 pages, 10 of them truly bad: **precision 0.83, recall 0.71**. The misses pointed to fixes, not threshold changes:

- numbers misread on a scan with high OCR confidence → a second OCR engine for numeric documents
- an undetected table → always run the table parser
- words hyphenated across line breaks → join them before chunking

48 pages are enough to set the shape of the gate, but not its final numbers. In production, thresholds are re-fitted per source, using reviewer decisions from the quarantine queue.

**Document level:** a document is indexed with its accepted pages and carries its parse coverage as metadata. Policy documents are the exception: one with a quarantined page isn't published, because a missing clause can change what a policy permits.

### 1.4 Chunking

One fixed window for everything splits tables from their headers and policy clauses from the sections that give them meaning. So the strategy follows the document type:

| Document type | Strategy |
|---|---|
| Policies and procedures | **By heading.** One section per chunk (300–800 tokens), split long sections at paragraphs, and prefix each chunk with its heading path (`Policy › Section › Subsection`). This needs a structure-preserving parser or Word's own styles, because plain text loses the headings. |
| Scanned vault documents | **By page**, split at paragraphs above ~800 tokens. OCR yields no reliable headings, and the page is the natural citation. |
| Tables | **One table per chunk**, with title and header. Large tables split into row groups that repeat the header. Numeric tables are also stored as structured rows (Task 2). |
| Charts | Caption plus surrounding text, optionally with a generated description marked as derived content |
| Slides | **One slide per chunk:** title, body and speaker notes |

**For every type:** join words hyphenated across lines, strip headers and footers (keep them as metadata), drop struck-through or deleted text, and remove near-duplicates before chunking.

---

## 2. Task 2 — The ETL framework and pipeline proposal

*To be written.*

## 3. Task 3 — One index, two tenants, entitlement-aware retrieval

*To be written.*
