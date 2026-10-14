"""
ParseBench quick inspection — Task 1, Steps 2 and 3

Picks a small sample of ParseBench pages, runs them through free parsers, and runs
quality DETECTORS that need no "correct" text, the way a production gate would.
Every page is then scored against ParseBench's own rules (its answer key), so you
can measure how well each detector predicts a page that really is badly parsed.

Parsers:
  - PyMuPDF   : reads the PDF's existing text layer (fast; finds nothing on scans)
  - Tesseract : OCR on a rendered image of the page (works on scans; makes mistakes)
  - Docling   : optional (--docling, pip install docling), layout-aware, rebuilds tables

Detectors (no ground truth needed), grouped by what the gate does with them:
  re-parse    a different parser or setting is likely to fix it
    empty output        nothing extracted (font the PDF engine can't decode, or a blank page)
    wrong script        OCR language doesn't match the script on the page
    rotated             page is sideways or upside down
    text layer suspect  OCR confidently reads words the PDF's text layer lacks
    possible omission   ink on the page that no extracted word covers
    table missing       rows of numbers, but no table was extracted
  quarantine  re-parsing won't help; a person decides
    low OCR confidence  the scan itself is poor
    garbled             text doesn't look like words in any known language
    digit disagreement  two OCR readings disagree on the numbers
    table issues        header missing, numeric columns not numeric, totals don't add up
  accept with flags  page is indexable; the flag changes chunking or metadata
    multicolumn         reading-order risk
    header/footer       page furniture to strip before chunking
    strikethrough       struck text to drop or tag (superseded content)
    figure              charts/pictures whose numbers won't reach a text index
    version marker      'draft', 'superseded', 'version 2.1' ... (lifecycle metadata)

Ground truth (ParseBench rules, scored with parse_bench's own scorer):
  gt_words_score      required words/sentences present        (content faithfulness)
  gt_digits_score     digits present in the right amounts
  gt_order_score      reading order of sentence pairs
  gt_header_footer_score, gt_extra_score (no invented/duplicated text)
  gt_format_score     bold/titles/strikeout (plain text output can't pass these)
  gt_table_grits      table structure + content (GriTS), table pages only
  gt_chart_score      chart data points, which need a table in the output
  gt_label            bad / ok, from GT_THRESHOLDS (blank for layout-only pages)

Outputs (in --out, default ./inspection):
  review.html   open in a browser: page image, flags, ground truth, every parser's output
  signals.csv   one row per page: signals, flags, suggested gate, ground truth, and two
                columns for you: your_label (ok / bad, overrides gt_label) and your_notes

Usage:
  python ParseTest.py                                # 4 pages per group, 24 total
  python ParseTest.py --per-group 8 --seed 7
  python ParseTest.py --rules-dir data/test          # the local 12-page slice
  python ParseTest.py --local-dir my_pdfs            # your own PDFs/images (no ground truth)
  python ParseTest.py --evaluate inspection/signals.csv
  python ParseTest.py --check-docx policy.docx       # Word strikethrough/tracked changes
"""

import argparse
import contextlib
import html
import io
import json
import logging
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image

REPO = "llamaindex/ParseBench"
TEXT_GROUPS = ["ocr", "multicolumns", "simple"]  # tags in text_content.jsonl
OTHER_GROUPS = ["table", "chart", "layout"]       # one rule file each
RULE_FILES = ["text_content", "text_formatting", "table", "chart", "layout"]
DPI = 300
SCALE = DPI / 72  # PDF points -> pixels

# Starting thresholds. These are guesses to be calibrated with --evaluate,
# not final values. Change them here and re-run.
THRESHOLDS = {
    "ocr_mean_conf_min": 80,          # average OCR word confidence (0-100)
    "garble_share_max": 0.20,         # share of words not recognised as real words
    "odd_char_share_max": 0.02,       # share of characters that are junk symbols
    "digit_agreement_min": 0.95,      # digit overlap between two OCR readings
    "layer_recall_min": 0.80,         # share of OCR's words the text layer also has
    "uncovered_ink_max": 0.20,        # share of ink pixels no extracted word covers
    "table_type_break_max": 0.10,     # share of numeric-column cells that aren't numbers
    "numeric_row_share_max": 0.30,    # share of text rows that look like table rows
    "figure_area_max": 0.15,          # share of the page covered by pictures/charts
    "osd_orientation_conf_min": 5,    # Tesseract's confidence that the page is rotated
}

# What a page has to score on ParseBench's rules to count as "fit to index".
# Words and digits are what retrieval needs; order, formatting and charts are
# reported but don't make a page bad on their own (they degrade, see Task 1).
GT_THRESHOLDS = {
    "gt_words_score": 0.90,
    "gt_digits_score": 0.90,
    "gt_table_grits": 0.70,
}

# Your Step 3 policy: what each flag does to the page.
REPARSE = ["flag_empty_output", "flag_wrong_script", "flag_rotated", "flag_text_layer_suspect",
           "flag_possible_omission", "flag_table_missing"]
QUARANTINE = ["flag_low_ocr_conf", "flag_garbled", "flag_digit_disagreement",
              "flag_table_issues"]
DEGRADING = ["flag_multicolumn", "flag_header_footer", "flag_strikethrough",
             "flag_figure", "flag_version_marker"]


# =============================================================== picking pages

def has_tag(tags, tag):
    return isinstance(tags, list) and tag in tags


def load_rules(rules_dir):
    """All five ParseBench rule files, from a local folder or the Hub."""
    rules = {}
    for name in RULE_FILES:
        if rules_dir:
            path = Path(rules_dir) / f"{name}.jsonl"
            if not path.exists():
                continue
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(REPO, f"{name}.jsonl", repo_type="dataset")
        rules[name] = pd.read_json(path, lines=True)
    return rules


def pick_parsebench_pages(rules, per_group, seed):
    """Return (group, path_in_repo) pairs, random but reproducible via the seed."""
    rng = random.Random(seed)
    picks = []
    text_rules = rules["text_content"]
    for tag in TEXT_GROUPS:
        pages = sorted(text_rules[text_rules["tags"].apply(has_tag, args=(tag,))]["pdf"].unique())
        picks += [(tag, p) for p in rng.sample(pages, min(per_group, len(pages)))]
    for group in OTHER_GROUPS:
        if group in rules:
            pages = sorted(rules[group]["pdf"].unique())
            picks += [(group, p) for p in rng.sample(pages, min(per_group, len(pages)))]
    return picks


def download(path_in_repo):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, path_in_repo, repo_type="dataset"))


# =============================================================== parsers

def parse_pymupdf(page):
    """Text layer, word boxes (in pixels) and any tables PyMuPDF can find."""
    text = page.get_text()
    boxes = [(w[0] * SCALE, w[1] * SCALE, w[2] * SCALE, w[3] * SCALE)
             for w in page.get_text("words")]
    tables = []
    try:
        tables = [t.extract() for t in page.find_tables().tables]
    except Exception:
        pass  # table detection is best-effort
    return text, boxes, tables


def parse_tesseract(image, lang):
    """OCR text rebuilt line by line, word confidences, words with boxes, and lines."""
    d = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    lines, confs, words_ = {}, [], []
    for i, word in enumerate(d["text"]):
        if not word.strip():
            continue
        x0, y0 = d["left"][i], d["top"][i]
        x1, y1 = x0 + d["width"][i], y0 + d["height"][i]
        words_.append((word, (x0, y0, x1, y1)))
        confs.append(float(d["conf"][i]))
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        ln = lines.setdefault(key, {"words": [], "box": [x0, y0, x1, y1]})
        ln["words"].append(word)
        b = ln["box"]
        ln["box"] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    line_list = [{"text": " ".join(l["words"]), "n_words": len(l["words"]), "box": l["box"]}
                 for l in lines.values()]
    text = "\n".join(l["text"] for l in line_list)
    return text, confs, words_, line_list


_docling = None

def parse_docling(path):
    """Markdown output plus tables as lists of rows (header row first)."""
    global _docling
    if _docling is None:
        from docling.document_converter import DocumentConverter
        _docling = DocumentConverter()
    doc = _docling.convert(str(path)).document
    tables = []
    for t in doc.tables:
        try:
            df = t.export_to_dataframe(doc=doc)
        except TypeError:  # older Docling versions
            df = t.export_to_dataframe()
        df = df.fillna("").astype(str)
        tables.append([list(map(str, df.columns))] + df.values.tolist())
    return doc.export_to_markdown(), tables


# =============================================================== detectors

def overlap(a, b):
    """Share of items two outputs agree on (0 = nothing in common, 1 = identical)."""
    total = max(sum(a.values()), sum(b.values()))
    return round(sum((a & b).values()) / total, 3) if total else None


def recall(found, reference):
    """Share of the reference reading that the other reading also contains."""
    total = sum(reference.values())
    return round(sum((found & reference).values()) / total, 3) if total else None


def words(text):
    return Counter(w.lower() for w in re.findall(r"\w+", text))


def digits(text):
    return Counter(re.findall(r"\d", text))


# ---- what kind of text layer is this? ----------------------------------------

def text_layer_kind(page):
    """'none', 'native' or 'invisible_ocr'.

    Scanners and OCR tools lay invisible text (render mode 3) over the page
    image. That layer is someone else's OCR of unknown quality, so the page is
    really a scan and gets judged like one.
    """
    try:
        spans = page.get_texttrace()
    except Exception:
        return "none"
    chars = sum(len(s["chars"]) for s in spans)
    if chars < 50:
        return "none"
    hidden = sum(len(s["chars"]) for s in spans if s["type"] == 3)
    return "invisible_ocr" if hidden / chars > 0.8 else "native"


# ---- script and rotation: is OCR running with the right settings? ------------

LANG_SCRIPT = {"eng": "Latin", "fra": "Latin", "deu": "Latin", "nld": "Latin", "spa": "Latin",
               "ita": "Latin", "por": "Latin", "swe": "Latin", "rus": "Cyrillic",
               "ukr": "Cyrillic", "ell": "Greek", "ara": "Arabic", "heb": "Hebrew",
               "chi_sim": "Han", "chi_tra": "Han", "jpn": "Japanese", "kor": "Hangul"}


def detect_script_and_rotation(image):
    """Tesseract's orientation and script detection (needs osd.traineddata)."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return None, None, None  # too little text to tell
    return osd["script"], osd["rotate"], osd["orientation_conf"]


# ---- garble: does it look like language? ------------------------------------

DICTIONARIES = ["en", "nl", "fr", "de", "es", "it", "pt"]  # pyspellchecker languages tried
_spell = {}
VOWELS = set("aeiouyàáâäèéêëìíîïòóôöùúûü")
NORMAL_PUNCT = set(".,;:!?'\"()[]{}-–—/\\%€$£&@#*+=<>_|§°’‘“”…·• ")


def spell_checker(lang):
    if lang not in _spell:
        from spellchecker import SpellChecker
        _spell[lang] = SpellChecker(language=lang)
    return _spell[lang]


def garble_signals(text):
    """Share of words that aren't real words, the language that fits best, and
    share of junk characters.

    Words are checked against each dictionary in DICTIONARIES and the best fit
    wins, so a Dutch page isn't 'garbled' for not being English. When no
    dictionary fits (Swedish, say), it falls back to a rough 'does it look
    pronounceable' rule and the language is 'unknown'. Short all-caps tokens
    (ICC, ENT, MAAA) are codes and acronyms, not words, and are skipped. Names
    and jargon still count as unknown: expect a few percent on clean pages.
    """
    tokens = [t.lower() for t in re.findall(r"[^\W\d_]{3,}", text)
              if not (t.isupper() and len(t) <= 5)]
    chars = [c for c in text if not c.isspace()]
    odd = sum(1 for c in chars if not c.isalnum() and c not in NORMAL_PUNCT)
    odd_share = round(odd / len(chars), 4) if chars else None
    if not tokens:
        return None, None, odd_share

    unique = set(tokens)
    best_lang, best = None, None
    for lang in DICTIONARIES:
        unknown = spell_checker(lang).unknown(unique)
        share = sum(1 for t in tokens if t in unknown) / len(tokens)
        if best is None or share < best:
            best_lang, best = lang, share
    if best > 0.5:  # no dictionary speaks this language
        best_lang = "unknown"
        best = sum(1 for t in tokens
                   if not (set(t) & VOWELS)
                   or re.search(r"[^aeiouy\W]{5,}", t)
                   or re.search(r"(.)\1\1", t)) / len(tokens)
    return round(best, 3), best_lang, odd_share


# ---- ink coverage: is there ink no extracted word covers? --------------------

def long_runs(mask, length, axis):
    """Pixels that sit in an unbroken run of at least `length` ink pixels along
    `axis`: table borders, underlines, form lines. Letters are never that long."""
    m = np.moveaxis(mask, axis, -1)
    n = m.shape[-1]
    if n < length:
        return np.zeros_like(mask)
    cs = np.concatenate([np.zeros(m.shape[:-1] + (1,), np.int32), m.cumsum(-1, dtype=np.int32)], -1)
    full = (cs[..., length:] - cs[..., :-length]) == length  # window starting at j is all ink
    fc = np.concatenate([np.zeros(m.shape[:-1] + (1,), np.int32), full.cumsum(-1, dtype=np.int32)], -1)
    p = np.arange(n)
    lo = np.clip(p - length + 1, 0, full.shape[-1])
    hi = np.clip(p + 1, 0, full.shape[-1])
    covered = (fc[..., hi] - fc[..., lo]) > 0
    return np.moveaxis(covered, -1, axis)


def uncovered_ink_share(image, word_boxes, ignore_boxes=(), pad=4):
    """Share of dark pixels not inside any extracted word box.

    Ruled lines are removed first. Pictures and logos on scans still count as
    'uncovered', so expect false alarms on those pages.
    """
    g = np.asarray(image.convert("L"))
    ink = g < 128
    ink &= ~(long_runs(ink, DPI // 2, 1) | long_runs(ink, DPI // 2, 0))
    if ink.sum() == 0:
        return None
    covered = np.zeros_like(ink)
    for x0, y0, x1, y1 in list(word_boxes) + list(ignore_boxes):
        covered[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad] = True
    return round(float((ink & ~covered).sum() / ink.sum()), 3)


def pdf_ignore_boxes(page, ocr_boxes, min_words=5):
    """Areas of a text-layer PDF that aren't text: thin lines, and embedded images
    that OCR finds (almost) no words in, i.e. real pictures. Images that DO contain
    words (a pasted screenshot, a scanned page inside a PDF) are kept, so text the
    text layer misses there counts as uncovered ink."""
    boxes = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = (v * SCALE for v in info["bbox"])
        words_inside = sum(1 for b in ocr_boxes
                           if x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1)
        if words_inside < min_words:
            boxes.append((x0, y0, x1, y1))
    for d in page.get_drawings():
        r = d["rect"]
        if r.height < 3 or r.width < 3:  # rules, borders, underlines
            boxes.append((r.x0 * SCALE, r.y0 * SCALE, r.x1 * SCALE, r.y1 * SCALE))
    return boxes


# ---- figures: content a text index won't see ---------------------------------

def figure_area_share(page, word_boxes):
    """Share of the page covered by pictures and filled vector shapes (charts)
    with no text-layer words on them. Text-layer PDFs only; on a scan the whole
    page is one picture."""
    w, h = int(page.rect.width) + 1, int(page.rect.height) + 1
    mask = np.zeros((h, w), bool)
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        if (x1 - x0) * (y1 - y0) < 0.9 * w * h:  # skip full-page backgrounds
            mask[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)] = True
    for d in page.get_drawings():
        r = d["rect"]
        if d.get("fill") and r.width > 3 and r.height > 3 and r.width * r.height < 0.9 * w * h:
            mask[max(0, int(r.y0)):int(r.y1), max(0, int(r.x0)):int(r.x1)] = True
    for x0, y0, x1, y1 in word_boxes:
        mask[int(y0 / SCALE):int(y1 / SCALE) + 1, int(x0 / SCALE):int(x1 / SCALE) + 1] = False
    return round(float(mask.mean()), 3)


# ---- columns: reading-order risk ---------------------------------------------

def estimate_columns(lines, page_width):
    """Rough column count from where text lines start horizontally."""
    real = [l for l in lines if l["n_words"] >= 3]
    narrow = [l for l in real if (l["box"][2] - l["box"][0]) < 0.55 * page_width]
    if len(real) < 6 or len(narrow) < 0.5 * len(real):
        return 1
    bins = Counter(int(l["box"][0] / (page_width / 12)) for l in narrow)
    big = sorted(b for b, n in bins.items() if n >= max(3, 0.15 * len(narrow)))
    clusters = sum(1 for i, b in enumerate(big) if i == 0 or b - big[i - 1] > 1)
    return max(1, clusters)


# ---- header / footer lines ---------------------------------------------------

PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/|van|sur)\s*\d{1,4})?$", re.I)


def header_footer_lines(doc, lines, page_height, margin=0.08):
    """Short lines in the top/bottom margin, plus lines repeated across pages."""
    found = [l["text"] for l in lines
             if l["n_words"] <= 12 and (l["box"][3] < margin * page_height
                                        or l["box"][1] > (1 - margin) * page_height)]
    found += [l["text"] for l in lines if PAGE_NUMBER.match(l["text"].strip())]
    if len(doc) > 1:  # repetition across pages (text layer only), digits normalised
        counts = Counter()
        for p in doc:
            counts.update({re.sub(r"\d+", "#", s.strip()) for s in p.get_text().splitlines() if s.strip()})
        found += [s for s, n in counts.items() if n >= max(2, 0.6 * len(doc))]
    return sorted(set(found))


# ---- strikethrough and version markers ---------------------------------------

def detect_strikethrough(page):
    """Text struck through in a PDF: StrikeOut annotations, or thin horizontal
    lines drawn through the middle of a text span (underlines sit lower and are
    ignored). Only works with a text layer; scans need image analysis or a VLM."""
    struck = []
    for a in page.annots():
        if a.type[1] == "StrikeOut":
            struck.append(page.get_textbox(a.rect).strip())

    rules = [d["rect"] for d in page.get_drawings()
             if d["rect"].height <= 2.5 and d["rect"].width >= 4]
    if rules:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    h = y1 - y0
                    if h <= 0 or not span["text"].strip():
                        continue
                    for r in rules:
                        mid = (r.y0 + r.y1) / 2
                        across = min(x1, r.x1) - max(x0, r.x0)
                        # A strike is drawn for this text: it covers most of the span
                        # and doesn't run far past it. Section rules and table borders
                        # that happen to cross a heading are much wider than the text.
                        if (y0 + 0.3 * h <= mid <= y0 + 0.75 * h and across > 0.8 * (x1 - x0)
                                and r.width < 1.2 * (x1 - x0) + 4):
                            struck.append(span["text"].strip())
                            break
    return struck


# Lifecycle words in English, Dutch and French. Metadata, not a quality problem:
# a superseded policy parses fine but must not be served as current. Only
# headers, footers and the title area are searched: in body text 'repealed' or
# 'withdrawn' is usually content (a law describing what it repeals), not status.
VERSION_MARKER = re.compile(
    r"\b(draft|superseded|obsolete|withdrawn|repealed|revoked|for review|"
    r"ontwerp|vervallen|ingetrokken|projet|brouillon|abrog[ée]e?|"
    r"version\s+\d[\d.]*|versie\s+\d[\d.]*|rev(?:ision)?\.?\s*\d+|v\d+\.\d+)\b", re.I)


def version_markers(text):
    return sorted({m.group(0).lower() for m in VERSION_MARKER.finditer(text)})


def docx_revision_signals(path):
    """Word files store strikethrough and tracked changes explicitly in XML,
    so detection is exact and free. No parsing of the page image needed."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    on = r'(?:\s+w:val="(?!false|0|off)[^"]*")?\s*/>'
    return {
        "strikethrough_runs": len(re.findall(r"<w:d?strike" + on, xml)),
        "tracked_deletions": len(re.findall(r"<w:del\b", xml)),
        "tracked_insertions": len(re.findall(r"<w:ins\b", xml)),
    }


# ---- tables ------------------------------------------------------------------

CURRENCY = re.compile(r"\b(EUR|USD|GBP|CHF)\b", re.I)
YEAR = re.compile(r"(19|20)\d{2}")


def to_number(cell):
    """Parse '1,234.56', '1.234,56', '(120)', '€ 25 000', '12%'. None if not a number."""
    if cell is None:
        return None
    s = CURRENCY.sub("", str(cell)).strip()
    if not s or s in {"-", "–", "—"} or re.search(r"[^\W\d_]", s):
        return None  # empty, a dash, or contains letters
    negative = s.startswith("(") and s.endswith(")") or s.startswith("-")
    s = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:  # whichever separator comes last is the decimal one
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if len(s.split(",")[-1]) == 3 else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if negative else v


def close(a, b):
    return abs(a - b) <= max(1.0, 0.01 * abs(b))


def check_table(rows):
    """Structural checks on one table given as a list of rows (header first)."""
    rows = [["" if c is None else str(c).strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]
    header, body = rows[0], rows[1:]

    # Years ('2023 | 2024') are normal column headers in financial tables.
    header_ok = sum(1 for c in header if c and (to_number(c) is None or YEAR.fullmatch(c))) \
        >= max(1, n_cols // 2)

    numeric_cols, breaks, numeric_cells = [], 0, 0
    for j in range(n_cols):
        filled = [r[j] for r in body if r[j]]
        parsed = [to_number(c) for c in filled]
        if filled and sum(v is not None for v in parsed) / len(filled) >= 0.6:
            numeric_cols.append(j)
            numeric_cells += len(filled)
            breaks += sum(v is None for v in parsed)

    # Totals: compare each 'total' row with the rows since the previous total,
    # and also with all non-total rows (so grand totals after subtotals pass).
    checked = failed = 0
    start = 0
    for i, r in enumerate(body):
        if "total" not in r[0].lower():
            continue
        for j in numeric_cols:
            if j == 0:
                continue
            total = to_number(r[j])
            recent = [to_number(b[j]) for b in body[start:i]]
            recent = [v for v in recent if v is not None]
            everything = [to_number(b[j]) for b in body[:i] if "total" not in b[0].lower()]
            everything = [v for v in everything if v is not None]
            if total is None or len(recent) < 2:
                continue
            checked += 1
            if not (close(sum(recent), total) or close(sum(everything), total)):
                failed += 1
        start = i + 1

    return {"header_ok": header_ok, "type_breaks": breaks, "numeric_cells": numeric_cells,
            "totals_checked": checked, "totals_failed": failed}


def table_signals(tables):
    results = [r for r in (check_table(t) for t in tables) if r]
    if not results:
        return {"n_tables": 0, "tables_without_header": None, "table_type_break_share": None,
                "totals_checked": None, "totals_failed": None}
    cells = sum(r["numeric_cells"] for r in results)
    return {
        "n_tables": len(results),
        "tables_without_header": sum(not r["header_ok"] for r in results),
        "table_type_break_share": round(sum(r["type_breaks"] for r in results) / cells, 3) if cells else 0.0,
        "totals_checked": sum(r["totals_checked"] for r in results),
        "totals_failed": sum(r["totals_failed"] for r in results),
    }


def numeric_row_share(ocr_words):
    """Share of visual text rows with 3+ numbers in them, i.e. rows that look like
    a table. OCR splits table columns into separate blocks, so rows are rebuilt
    from word positions rather than taken from Tesseract's own lines."""
    if not ocr_words:
        return None
    heights = sorted(b[3] - b[1] for _, b in ocr_words)
    tol = 0.5 * heights[len(heights) // 2]
    rows, current, y = [], [], None
    for word, b in sorted(ocr_words, key=lambda wb: (wb[1][1] + wb[1][3]) / 2):
        yc = (b[1] + b[3]) / 2
        if y is not None and yc - y > tol:
            rows.append(current)
            current = []
        current.append(word)
        y = yc
    rows.append(current)
    real = [r for r in rows if len(r) >= 3]
    numeric = [r for r in real if sum(to_number(w) is not None for w in r) >= 3]
    return round(len(numeric) / len(real), 3) if real else None


def tables_as_html(tables):
    return "\n\n".join(
        "<table>" + "".join("<tr>" + "".join(f"<td>{html.escape('' if c is None else str(c))}</td>"
                                             for c in row) + "</tr>" for row in t) + "</table>"
        for t in tables)


# =============================================================== ground truth

GT_TEXT_TYPES = {
    "gt_words_score": {"missing_specific_word", "missing_specific_sentence",
                       "missing_word_percent", "missing_sentence_percent"},
    "gt_digits_score": {"bag_of_digit_percent"},
    "gt_order_score": {"order"},
    "gt_header_footer_score": {"is_header", "is_footer"},
    "gt_extra_score": {"unexpected_word_percent", "unexpected_sentence_percent",
                       "too_many_word_occurence_percent", "too_many_sentence_occurence_percent"},
}
GT_COLUMNS = ["gt_text_score", *GT_TEXT_TYPES, "gt_text_score_other_parser",
              "gt_format_score", "gt_table_grits", "gt_chart_score", "gt_label"]


def rule_dicts(sub):
    out = []
    for r in sub.itertuples():
        d = json.loads(r.rule) if isinstance(r.rule, str) and r.rule else {}
        d.update(type=r.type, id=r.id)
        if pd.notna(r.page):
            d["page"] = int(r.page)
        out.append(d)
    return out


def score_rules(sub, text):
    """Run ParseBench's own rule scorer. Returns the overall score and every rule's result."""
    from parse_bench.evaluation.metrics.parse.rule_based_metric import RuleBasedMetric
    with contextlib.redirect_stdout(io.StringIO()):  # it prints progress per rule
        m = RuleBasedMetric().compute(rule_dicts(sub), text)
    return m.value, m.metadata.get("rule_results", [])


def score_tables(expected_md, tables):
    """GriTS (grid table similarity), the table metric ParseBench reports."""
    from parse_bench.evaluation.metrics.parse.grits_metric import GriTSMetric
    from parse_bench.evaluation.metrics.parse.table_extraction import extract_table_pairs
    expected, actual, _ = extract_table_pairs(expected_md, tables_as_html(tables))
    if not expected:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = GriTSMetric().compute(expected, actual)
    for m in metrics:
        if m.metric_name == "grits_con":
            return round(m.value, 3)
    return None


def ground_truth(pdf, rules, primary_text, other_text, tables):
    gt = dict.fromkeys(GT_COLUMNS)
    if not rules:
        return gt

    def rules_for(name):
        df = rules.get(name)
        return df[df["pdf"] == pdf] if df is not None else pd.DataFrame()

    sub = rules_for("text_content")
    if not sub.empty:
        value, results = score_rules(sub, primary_text)
        gt["gt_text_score"] = round(value, 3)
        for col, types in GT_TEXT_TYPES.items():
            scores = [r["score"] for r in results if r["type"] in types]
            gt[col] = round(sum(scores) / len(scores), 3) if scores else None
        if other_text:
            gt["gt_text_score_other_parser"] = round(score_rules(sub, other_text)[0], 3)
    sub = rules_for("text_formatting")
    if not sub.empty:
        gt["gt_format_score"] = round(score_rules(sub, primary_text)[0], 3)
    sub = rules_for("chart")
    if not sub.empty:
        gt["gt_chart_score"] = round(score_rules(sub, primary_text)[0], 3)
    sub = rules_for("table")
    if not sub.empty:
        gt["gt_table_grits"] = score_tables("\n\n".join(sub["expected_markdown"].dropna()), tables)

    scored = {k: gt[k] for k in GT_THRESHOLDS if gt[k] is not None}
    if scored:
        gt["gt_label"] = "bad" if any(v < GT_THRESHOLDS[k] for k, v in scored.items()) else "ok"
    return gt


# =============================================================== flags and gate

def below(value, limit):
    return value is not None and value < limit


def above(value, limit):
    return value is not None and value > limit


def add_flags(row):
    t = THRESHOLDS
    scan = row["text_layer_kind"] != "native"
    row["flag_empty_output"] = row["primary_chars"] < 50
    # Script only matters where OCR text is what gets indexed; OSD guesses
    # wrongly on sparse pages like tables, so don't apply it to native layers.
    row["flag_wrong_script"] = scan and row["ocr_script"] is not None and not row["script_matches_lang"]
    row["flag_rotated"] = bool(row["ocr_rotate"]) and not below(row["osd_orientation_conf"],
                                                                t["osd_orientation_conf_min"])
    # A born-digital text layer is normally the truth. It is suspect only when
    # OCR confidently reads words the layer doesn't have (broken font encoding,
    # text inside images). OCR missing words the layer has says nothing about it.
    row["flag_text_layer_suspect"] = (row["text_layer_kind"] == "native"
                                      and below(row["layer_recall_of_ocr"], t["layer_recall_min"])
                                      and not below(row["ocr_mean_conf"], t["ocr_mean_conf_min"]))
    row["flag_possible_omission"] = above(row["uncovered_ink_share"], t["uncovered_ink_max"])
    row["flag_table_missing"] = row["n_tables"] == 0 and above(row["numeric_row_share"],
                                                               t["numeric_row_share_max"])

    row["flag_low_ocr_conf"] = scan and below(row["ocr_mean_conf"], t["ocr_mean_conf_min"])
    row["flag_garbled"] = above(row["garble_share"], t["garble_share_max"]) or above(row["odd_char_share"], t["odd_char_share_max"])
    # Digit disagreement only means something when neither reading is authoritative:
    # an old invisible OCR layer vs our OCR. Against a native layer it measures our OCR.
    row["flag_digit_disagreement"] = (row["text_layer_kind"] == "invisible_ocr"
                                      and below(row["digit_agreement"], t["digit_agreement_min"]))
    row["flag_table_issues"] = bool(row["tables_without_header"] or row["totals_failed"]
                                    or above(row["table_type_break_share"], t["table_type_break_max"]))

    row["flag_multicolumn"] = row["columns_estimate"] >= 2
    row["flag_header_footer"] = bool(row["header_footer"])
    row["flag_strikethrough"] = bool(row["struck_text"])
    row["flag_figure"] = above(row["figure_area_share"], t["figure_area_max"])
    row["flag_version_marker"] = bool(row["version_markers"])

    if any(row[f] for f in REPARSE):
        row["suggested_gate"] = "re-parse"
    elif any(row[f] for f in QUARANTINE):
        row["suggested_gate"] = "quarantine"
    elif any(row[f] for f in DEGRADING):
        row["suggested_gate"] = "accept with flags"
    else:
        row["suggested_gate"] = "accept"
    return row


# =============================================================== evaluation

SWEEP = [  # continuous signals worth a threshold, and which side is bad
    ("ocr_mean_conf", "low"), ("garble_share", "high"), ("layer_recall_of_ocr", "low"),
    ("digit_agreement", "low"), ("uncovered_ink_share", "high"), ("numeric_row_share", "high"),
]


def evaluate(csv_path):
    """Compare each detector with ParseBench's answer key, or with your own labels."""
    df = pd.read_csv(csv_path, sep=None, engine="python")  # copes with Excel's ';' too
    mine = df["your_label"].fillna("").astype(str).str.strip().str.lower()
    if mine.isin(["ok", "bad"]).any():
        source, label = "your_label", mine
    else:
        source, label = "gt_label (ParseBench rules)", df["gt_label"].fillna("").astype(str)
    all_df = df
    df = df[label.isin(["ok", "bad"])].copy()
    truth = label[df.index].eq("bad")
    if df.empty:
        print("No labelled rows: no ParseBench rules for these pages and your_label is empty.")
        return
    df["gate_blocks"] = df["suggested_gate"].isin(["re-parse", "quarantine"])

    def as_bool(s):
        return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    flags = REPARSE + QUARANTINE + DEGRADING
    print(f"Truth from {source}: {len(df)} labelled pages, {truth.sum()} bad.\n")
    print(f"{'detector':28} {'flagged':>7} {'caught':>7} {'false':>7} {'missed':>7} {'precision':>10} {'recall':>7}")
    for col in flags + ["gate_blocks"]:
        pred = as_bool(df[col])
        tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
        precision = f"{tp / (tp + fp):.2f}" if tp + fp else "-"
        recall = f"{tp / (tp + fn):.2f}" if tp + fn else "-"
        print(f"{col:28} {int(pred.sum()):>7} {tp:>7} {fp:>7} {fn:>7} {precision:>10} {recall:>7}")

    # Does a flag pick out pages that really score worse on the matching rules?
    # Uses every page with a score, including layout/chart pages without a label.
    gt_cols = [c for c in ["gt_words_score", "gt_digits_score", "gt_order_score",
                           "gt_header_footer_score", "gt_format_score", "gt_table_grits",
                           "gt_chart_score"] if all_df[c].notna().any()]
    print("\nMean ParseBench score, flagged / not flagged (n/a = no page in that group has rules):")
    print(f"{'detector':28}" + "".join(f"{c[3:]:>22}" for c in gt_cols))
    for col in flags:
        pred = as_bool(all_df[col])
        if not pred.any():
            continue
        cells = []
        for c in gt_cols:
            a, b = all_df.loc[pred, c].mean(), all_df.loc[~pred, c].mean()
            cells.append(f"{'n/a' if pd.isna(a) else f'{a:.2f}':>10} / {'n/a' if pd.isna(b) else f'{b:.2f}':<9}")
        print(f"{col:28}" + "".join(f"{c:>22}" for c in cells))

    print("\nThreshold sweep against the labels (pick the row you'd defend):")
    for signal, bad_side in SWEEP:
        values = df[signal].dropna()
        if values.nunique() < 2:
            continue
        print(f"  {signal} (bad when {bad_side})")
        for limit in sorted(values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).unique()):
            pred = (df[signal] < limit) if bad_side == "low" else (df[signal] > limit)
            tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
            fn = int((~pred & truth).sum())
            p = f"{tp / (tp + fp):.2f}" if tp + fp else "-"
            r = f"{tp / (tp + fn):.2f}" if tp + fn else "-"
            print(f"    {'<' if bad_side == 'low' else '>'} {limit:<8} flags {int(pred.sum()):>3}  precision {p:>4}  recall {r:>4}")

    missed = df[truth & ~df["gate_blocks"]]
    if not missed.empty:
        print("\nBad pages the gate would have accepted (the most interesting rows):")
        for _, r in missed.iterrows():
            print(f"  {r['id']}: words {r['gt_words_score']}, digits {r['gt_digits_score']}, "
                  f"tables {r['gt_table_grits']}  {r.get('your_notes', '') if pd.notna(r.get('your_notes')) else ''}")
    print("\nprecision = of the pages flagged, share that are bad (low = noisy review queue)")
    print("recall    = of the bad pages, share flagged (low = failures slip through)")


# =============================================================== report

def html_report(rows, out_dir):
    css = """
    :root { --bg:#fff; --fg:#1d1d1f; --muted:#666; --box:#f4f4f6; --line:#ddd;
            --red:#b42318; --amber:#9a6700; --green:#1a7f37; }
    @media (prefers-color-scheme: dark) {
      :root { --bg:#161618; --fg:#eee; --muted:#aaa; --box:#222226; --line:#333;
              --red:#ff7b72; --amber:#e3b341; --green:#56d364; } }
    body { font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg);
           margin:0 auto; max-width:1500px; padding:24px; }
    section { border-top:1px solid var(--line); padding:20px 0; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    img { max-width:100%; border:1px solid var(--line); }
    pre { background:var(--box); padding:10px; max-height:480px; overflow:auto;
          white-space:pre-wrap; font-size:12px; }
    .sig { color:var(--muted); font-size:13px; }
    .tag { display:inline-block; border:1px solid currentColor; border-radius:4px;
           padding:1px 6px; margin:2px; font-size:12px; }
    .block { color:var(--red); } .degrade { color:var(--amber); } .ok { color:var(--green); }
    """
    signal_keys = ["text_layer_kind", "primary_parser", "primary_chars", "ocr_mean_conf",
                   "ocr_script", "text_lang", "garble_share", "odd_char_share", "word_agreement",
                   "digit_agreement", "layer_recall_of_ocr", "uncovered_ink_share", "columns_estimate", "n_tables",
                   "numeric_row_share", "figure_area_share", "totals_checked", "totals_failed"]
    parts = [f"<!doctype html><meta charset='utf-8'><title>Parser review</title>"
             f"<style>{css}</style><h1>Parser review ({len(rows)} pages)</h1>"]
    for r in rows:
        gate_cls = {"accept": "ok", "accept with flags": "degrade"}.get(r["suggested_gate"], "block")
        tags = [f"<span class='tag {gate_cls}'>gate: {r['suggested_gate']}</span>"]
        tags += [f"<span class='tag block'>{f[5:]}</span>" for f in REPARSE + QUARANTINE if r[f]]
        tags += [f"<span class='tag degrade'>{f[5:]}</span>" for f in DEGRADING if r[f]]
        if r["gt_label"]:
            tags.append(f"<span class='tag {'block' if r['gt_label'] == 'bad' else 'ok'}'>"
                        f"ParseBench: {r['gt_label']}</span>")
        sig = " · ".join(f"{k}: {r[k]}" for k in signal_keys)
        gt = " · ".join(f"{k[3:]}: {r[k]}" for k in GT_COLUMNS[:-1] if r[k] is not None)
        extra = f"<p class='sig'>Ground truth: {gt}</p>" if gt else ""
        if r["struck_text"]:
            extra += f"<p class='sig'>Struck through: {html.escape(r['struck_text'])}</p>"
        if r["version_markers"]:
            extra += f"<p class='sig'>Version markers: {html.escape(r['version_markers'])}</p>"
        if r["header_footer"]:
            extra += f"<p class='sig'>Header/footer lines: {html.escape(r['header_footer'])}</p>"
        outputs = "".join(f"<h3>{name}</h3><pre>{html.escape(text or '(empty)')}</pre>"
                          for name, text in r["_out"].items())
        parts.append(
            f"<section><h2>{r['id']} [{r['group']}] {html.escape(r['source'])}</h2>"
            f"<div>{''.join(tags)}</div><p class='sig'>{sig}</p>{extra}"
            f"<div class='grid'><img src='pages/{r['id']}.png'><div>{outputs}</div></div></section>")
    (out_dir / "review.html").write_text("".join(parts), encoding="utf-8")


def tables_as_text(tables):
    return "\n\n".join("\n".join(" | ".join("" if c is None else str(c) for c in row) for row in t)
                       for t in tables)


# =============================================================== main

def inspect_page(group, source, path, page_id, out, args, rules):
    doc = pymupdf.open(path)  # also opens JPG/PNG as a one-page document
    page = doc[0]             # ParseBench files are (almost always) single pages
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    pix.save(out / "pages" / f"{page_id}.png")
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    layer_text, layer_boxes, mu_tables = parse_pymupdf(page)
    ocr_text, confs, ocr_words, ocr_lines = parse_tesseract(image, args.lang)
    ocr_boxes = [b for _, b in ocr_words]
    kind = text_layer_kind(page)
    has_layer = kind != "none"
    script, rotate, orientation_conf = detect_script_and_rotation(image)
    expected_scripts = {LANG_SCRIPT.get(l) for l in args.lang.split("+")}

    outputs = {"PyMuPDF (text layer)": layer_text, "Tesseract (OCR)": ocr_text}
    if mu_tables:
        outputs["PyMuPDF tables"] = tables_as_text(mu_tables)
    tables, table_source = mu_tables, "pymupdf" if mu_tables else None
    if args.docling:
        try:
            md, dl_tables = parse_docling(path)
            outputs["Docling"] = md
            if dl_tables:
                tables, table_source = dl_tables, "docling"
        except Exception as e:
            outputs["Docling"] = f"(Docling failed: {e})"

    for name, text in outputs.items():
        fname = re.sub(r"\W+", "_", name.lower()).strip("_") + ".txt"
        (out / "text" / f"{page_id}_{fname}").write_text(text, encoding="utf-8")

    # The 'primary' output is what a cheap production pipeline would index:
    # the text layer when there is one, otherwise OCR.
    primary = "pymupdf" if has_layer else "tesseract"
    primary_text = layer_text if has_layer else ocr_text
    other_text = ocr_text if has_layer else None
    garble, text_lang, odd = garble_signals(primary_text)
    struck = detect_strikethrough(page) if kind == "native" else []
    ink = (uncovered_ink_share(image, layer_boxes, pdf_ignore_boxes(page, ocr_boxes)) if has_layer
           else uncovered_ink_share(image, ocr_boxes))

    header_footer = header_footer_lines(doc, ocr_lines, pix.height)
    title_area = [l["text"] for l in ocr_lines if l["box"][3] < 0.2 * pix.height]

    row = {
        "id": page_id, "group": group, "source": source, "n_pages": len(doc),
        "text_layer_kind": kind,
        "primary_parser": primary,
        "text_layer_chars": len(layer_text.strip()),
        "primary_chars": len(primary_text.strip()),
        "ocr_words": len(confs),
        "ocr_mean_conf": round(sum(confs) / len(confs), 1) if confs else None,
        "ocr_script": script,
        "script_matches_lang": script in expected_scripts if script else None,
        "ocr_rotate": rotate,
        "osd_orientation_conf": orientation_conf,
        "text_lang": text_lang,
        "garble_share": garble,
        "odd_char_share": odd,
        # Two independent readings of the same page; needs a text layer to compare with.
        "word_agreement": overlap(words(layer_text), words(ocr_text)) if has_layer else None,
        "digit_agreement": overlap(digits(layer_text), digits(ocr_text)) if has_layer else None,
        "layer_recall_of_ocr": recall(words(layer_text), words(ocr_text)) if has_layer else None,
        "uncovered_ink_share": ink,
        "columns_estimate": estimate_columns(ocr_lines, pix.width),
        "numeric_row_share": numeric_row_share(ocr_words),
        "figure_area_share": figure_area_share(page, layer_boxes) if kind == "native" else None,
        "table_source": table_source,
        **table_signals(tables),
        "struck_text": " | ".join(struck),
        "header_footer": " | ".join(header_footer),
        "version_markers": " | ".join(version_markers("\n".join(header_footer + title_area))),
    }
    row = add_flags(row)
    row.update(ground_truth(source, rules, primary_text, other_text, tables))
    row.update({"your_label": "", "your_notes": "", "_out": outputs})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=4, help="pages per group (default 4)")
    ap.add_argument("--seed", type=int, default=42, help="change for a different sample")
    ap.add_argument("--out", default="inspection")
    ap.add_argument("--lang", default="eng", help="Tesseract language(s), e.g. eng+fra")
    ap.add_argument("--docling", action="store_true", help="also run Docling (slower)")
    ap.add_argument("--rules-dir", help="local ParseBench copy (jsonl files + docs/), e.g. data/test")
    ap.add_argument("--local-dir", help="inspect your own PDFs/images instead of ParseBench")
    ap.add_argument("--evaluate", metavar="CSV", help="score detectors against ground truth / your labels")
    ap.add_argument("--check-docx", nargs="+", metavar="DOCX", help="Word strikethrough/tracked changes")
    args = ap.parse_args()

    if args.evaluate:
        return evaluate(args.evaluate)
    if args.check_docx:
        for f in args.check_docx:
            print(f, docx_revision_signals(f))
        return
    if args.docling:
        try:
            import docling  # noqa: F401
        except ImportError:
            ap.error("--docling needs Docling installed: uv add docling")

    logging.getLogger("parse_bench").setLevel(logging.WARNING)
    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "text").mkdir(exist_ok=True)

    rules = {}
    if args.local_dir:
        files = sorted(p for p in Path(args.local_dir).iterdir()
                       if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})
        picks = [("local", str(p)) for p in files]
    else:
        print("Loading ParseBench rules (first run downloads them)...")
        rules = load_rules(args.rules_dir)
        picks = pick_parsebench_pages(rules, args.per_group, args.seed)

    rows = []
    for i, (group, source) in enumerate(picks, 1):
        page_id = f"{i:02d}_{group}"
        print(f"[{i}/{len(picks)}] {group}: {source}")
        if args.local_dir:
            path = Path(source)
        elif args.rules_dir:
            path = Path(args.rules_dir) / source
        else:
            path = download(source)
        try:
            rows.append(inspect_page(group, source, path, page_id, out, args, rules))
        except Exception as e:
            print(f"   skipped ({type(e).__name__}: {e})")

    pd.DataFrame([{k: v for k, v in r.items() if k != "_out"} for r in rows]) \
        .to_csv(out / "signals.csv", index=False)
    html_report(rows, out)
    print(f"\nDone. Open {out / 'review.html'} in a browser, then run:")
    print(f"   python ParseTest.py --evaluate {out / 'signals.csv'}")
    print("Fill your_label (ok / bad) in the CSV to override ParseBench's verdict on a page.")


if __name__ == "__main__":
    main()
