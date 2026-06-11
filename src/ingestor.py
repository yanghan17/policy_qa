"""
ingestor.py
-----------
Loads PDF policy documents and splits them into section-aware chunks.

Each chunk carries metadata:
  {
    "doc_name"     : "qbe_home_policy.pdf",
    "page"         : 9,
    "section"      : "Section 1 - Home Contents",
    "heading_path" : "Section 1 - Home Contents > Exclusions",
    "clause_ref"   : "1(c)",
    "text"         : "...",
    "chunk_id"     : "qbe_home_policy.pdf::42"
  }

Chunking strategy
-----------------
* pdfplumber extracts text page-by-page with better whitespace
  preservation than pypdf — important for detecting headings.
* Heading detection uses three heuristics: all-caps short lines,
  lines matching insurance keywords, and numbered section patterns.
* A heading breadcrumb stack tracks the hierarchy so every chunk
  knows exactly where it lives in the document (used for citations).
* Chunks flush at ~TARGET_TOKENS tokens, but NEVER straddle a section
  boundary — a heading always starts a fresh chunk.
* The last OVERLAP_TOKENS tokens of each chunk are prepended to the
  next one so clauses spanning a boundary appear fully in both.
* Token count is approximated as len(text)//4 (no tokeniser needed).
"""

import re
import os
from dataclasses import dataclass
from typing import List

import pdfplumber


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_TOKENS         = 600   # target chunk size in tokens
OVERLAP_TOKENS        = 80    # overlap between consecutive chunks
APPROX_CHARS_PER_TOK  = 4     # English prose approximation


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:     str    # unique: "<doc_name>::<index>"
    doc_name:     str    # filename only, e.g. "policy.pdf"
    page:         int    # page where chunk starts (1-based)
    section:      str    # top-level section heading
    heading_path: str    # breadcrumb: "Section 1 > Exclusions > ..."
    clause_ref:   str    # best-effort clause number, e.g. "1(c)(ii)"
    text:         str    # chunk body text

    def to_dict(self) -> dict:
        return {
            "chunk_id":     self.chunk_id,
            "doc_name":     self.doc_name,
            "page":         self.page,
            "section":      self.section,
            "heading_path": self.heading_path,
            "clause_ref":   self.clause_ref,
            "text":         self.text,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


# ---------------------------------------------------------------------------
# Heading / clause detection
# ---------------------------------------------------------------------------

HEADING_KEYWORDS = re.compile(
    r"\b(section|clause|part|schedule|definition|exclusion|condition|"
    r"cover|benefit|premium|endorsement|general|specific|special|appendix|"
    r"claim|settlement|guarantee|cancell|excess|liability|procedure|"
    r"sanction|unoccupan|discount|subrogat|alterati)\b",
    re.IGNORECASE,
)

CLAUSE_RE = re.compile(r"^[\s]*(\d[\.\d]*\.?|[a-zA-Z][\.\)]\s|[\(\d]+[\)\.])\s")

# Normalise decorative fonts: "SECtIon 1 - HomE ContEntS" → "Section 1 - Home Contents"
# pdfplumber sometimes reads stylised all-small-caps fonts as mixed case.
# We detect this by checking if the string is neither all-upper nor normal title/sentence case.
def _normalise_heading(s: str) -> str:
    """
    If a line looks like it came from a small-caps decorative font
    (alternating or inconsistent casing that isn't normal prose), normalise it
    to title case so heading detection and metadata are readable.
    """
    # If already sensible, leave it alone
    if s == s.upper() or s == s.lower() or s == s.title():
        return s
    # Heuristic: if more than 40% of alpha chars are uppercase, treat as all-caps heading
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return s
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if upper_ratio > 0.4:
        return s.title()
    return s


def _approx_tokens(text: str) -> int:
    return len(text) // APPROX_CHARS_PER_TOK


def _is_heading(line: str) -> bool:
    """Return True if line looks like a section / clause heading."""
    s = _normalise_heading(line.strip())
    if not s or len(s) > 120:
        return False

    # Reject lines that are clearly body sentences or list items:
    # - ends with . ; or ,
    # - starts with a roman numeral list prefix (i. ii. iii. iv.)
    # - starts with lowercase AND is long
    # - starts with common body-text opener words
    # - contains CJK characters (non-English headers, addresses, etc.)
    if s.endswith(".") or s.endswith(";") or s.endswith(","):
        return False
    if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", s):
        return False
    if re.match(r"^(https?://|www\.)", s):
        return False
    if re.match(r"^(i{1,3}v?|vi{0,3}|ix|iv)\.\s", s, re.IGNORECASE):
        return False
    if s[0].islower() and len(s) > 55:
        return False
    if re.match(r"^(However|If |We will|You |Your |provided|subject|unless)", s):
        return False

    # All-caps short line (e.g. "EXCLUSIONS", "WHAT WE COVER")
    if s.isupper() and len(s) > 3:
        return True
    # Keyword match within a short line — must start with uppercase or digit
    # to avoid matching body fragments like "condition substantially the same..."
    if HEADING_KEYWORDS.search(s) and len(s) < 80 and (s[0].isupper() or s[0].isdigit()):
        return True
    # Explicit Section N pattern
    if re.match(r"^(Section\s+\d+|SECTION\s+\d+|\d+\.\s+[A-Z])", s):
        return True
    # Short lines (≤ 55 chars) that are styled as headings:
    # - starts with uppercase and has no full stop (already checked above)
    # - OR all-lowercase short phrase (deliberate heading style in some PDFs)
    # We exclude lines starting with "(", digits, or bullet chars to avoid
    # treating sub-clause labels like "(h) caused by..." as headings.
    if len(s) <= 55 and not re.match(r"^[\(\d•\-]", s):
        if s[0].isupper():   # first word capitalised, rest can be any case
            return True
        if s == s.lower() and len(s.split()) <= 4:   # e.g. "money back guarantee"
            return True

    return False


def _extract_clause_ref(line: str) -> str:
    m = CLAUSE_RE.match(line)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Page-level text extraction
# ---------------------------------------------------------------------------

def _is_two_column(page) -> bool:
    """
    Detect two-column layout by checking if significant text exists
    in both the left and right halves of the page.
    """
    words    = page.extract_words()
    if not words:
        return False
    mid      = page.width / 2
    n_left   = sum(1 for w in words if w["x0"] < mid)
    n_right  = sum(1 for w in words if w["x0"] >= mid)
    # Both halves need at least 20 words, and neither dominates by 4:1
    if n_left < 20 or n_right < 20:
        return False
    ratio = n_right / n_left
    return 0.25 < ratio < 4.0


def _extract_page_text(page) -> str:
    """
    Extract text from a single pdfplumber page.
    For two-column layouts, extract each column separately and
    concatenate left then right — preventing cross-column line merging.
    """
    if _is_two_column(page):
        w, h  = page.width, page.height
        left  = page.within_bbox((0,   0, w / 2, h)).extract_text() or ""
        right = page.within_bbox((w / 2, 0, w,   h)).extract_text() or ""
        return left + "\n" + right
    return page.extract_text(x_tolerance=2, y_tolerance=3) or ""


def _extract_pages(pdf_path: str) -> List[dict]:
    """Return [{page_num, lines}] for every page in the PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw   = _extract_page_text(page)
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            pages.append({"page_num": i, "lines": lines})
    return pages


# Strict pattern for top-level section resets — must start with "Section N"
# followed by a dash or space and a meaningful title word.
# This prevents body text like "Section 1 or Section 2 We will pay:" from
# being treated as a new top-level section.
TOP_LEVEL_SECTION_RE = re.compile(
    r"^Section\s+\d+\s*[-–—]\s*\S",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Section-aware chunker
# ---------------------------------------------------------------------------

def _build_chunks(pages: List[dict], doc_name: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx            = 0
    heading_stack: List[str] = []
    current_section          = ""
    buf_lines:    List[str]  = []
    buf_page                 = 1
    buf_clause               = ""
    overlap_text             = ""

    # Track how many times each top-level section title has been seen.
    # In the home policy, p.1 is a summary that lists all sections once.
    # The real content pages repeat the same titles — we want those, not p.1.
    # Simplest fix: don't update current_section on page 1 of any document,
    # since page 1 is always the cover/summary in these QBE PDFs.
    SKIP_SECTION_RESET_PAGE = 1

    def _flush():
        nonlocal idx, overlap_text, buf_lines, buf_page, buf_clause
        if not buf_lines:
            return
        text = (overlap_text + " " + " ".join(buf_lines)).strip()
        if not text:
            return
        heading_path = " > ".join(heading_stack) if heading_stack else current_section
        chunks.append(Chunk(
            chunk_id     = f"{doc_name}::{idx}",
            doc_name     = doc_name,
            page         = buf_page,
            section      = current_section,
            heading_path = heading_path,
            clause_ref   = buf_clause,
            text         = text,
        ))
        idx += 1
        tail = OVERLAP_TOKENS * APPROX_CHARS_PER_TOK
        overlap_text = text[-tail:] if len(text) > tail else text
        buf_lines  = []
        buf_clause = ""

    for page_info in pages:
        page_num = page_info["page_num"]
        for line in page_info["lines"]:
            s = line.strip()

            if _is_heading(s):
                # Flush accumulated content before starting new section
                if _approx_tokens(" ".join(buf_lines)) > 80:
                    _flush()

                # Normalise the heading text before storing in metadata
                s_norm = _normalise_heading(s)

                # Update heading hierarchy
                if TOP_LEVEL_SECTION_RE.match(s_norm):
                    # Skip section resets on the cover/summary page (p.1)
                    # to prevent summary mentions from poisoning the heading stack.
                    if page_num > SKIP_SECTION_RESET_PAGE:
                        current_section = s_norm
                        heading_stack   = [s_norm]
                elif not current_section:
                    # No section set yet — use first heading found as anchor
                    current_section = s_norm
                    heading_stack   = [s_norm]
                else:
                    heading_stack = heading_stack[:2] + [s_norm]

                buf_lines.append(s_norm)
                if not buf_page:
                    buf_page = page_num
                continue

            if not buf_lines:
                buf_page = page_num
            if not buf_clause:
                buf_clause = _extract_clause_ref(s)

            buf_lines.append(s)

            if _approx_tokens(" ".join(buf_lines)) >= TARGET_TOKENS:
                _flush()

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str) -> List[Chunk]:
    """Load one PDF and return its chunks."""
    doc_name = os.path.basename(pdf_path)
    return _build_chunks(_extract_pages(pdf_path), doc_name)


def ingest_folder(folder: str) -> List[Chunk]:
    """Ingest every PDF in folder, return combined chunk list."""
    all_chunks: List[Chunk] = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(folder, fname)
            print(f"  Ingesting: {fname}")
            doc_chunks = ingest_pdf(path)
            all_chunks.extend(doc_chunks)
            print(f"    → {len(doc_chunks)} chunks")
    return all_chunks


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    chunks = ingest_folder(folder)
    print(f"\nTotal chunks: {len(chunks)}")
    c = chunks[10]
    print(f"\n--- Sample chunk {c.chunk_id} ---")
    print(f"  Doc     : {c.doc_name}")
    print(f"  Page    : {c.page}")
    print(f"  Section : {c.section}")
    print(f"  Path    : {c.heading_path}")
    print(f"  Clause  : {c.clause_ref}")
    print(f"  Text    : {c.text[:300]}…")
