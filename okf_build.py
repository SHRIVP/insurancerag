"""
Build an Open Knowledge Format (OKF) bundle from the source PDFs.

OKF (Google Cloud, v0.1) packages knowledge as a directory of markdown files
with YAML frontmatter, cross-linked with ordinary markdown links so agents can
navigate it as a graph. `type` is the only required frontmatter field;
`index.md` files act as entry points for progressive disclosure.

Bundle layout produced here:

    okf/
      index.md                                  ← root entry point
      naics/
        us/index.md                             ← sectors (NAICS USA 2022)
        us/<code>.md                            ← one page per code (2–6 digits)
        ca/index.md, ca/<code>.md               ← NAICS Canada 2017
      commercial-property/index.md
      commercial-property/<source-stem>/index.md
      commercial-property/<source-stem>/p<N>.md ← one page per PDF page
      commercial-general-liability/...
      personal-auto/...

NAICS pages are linked to their parent and children and to any codes named in
their cross-references, which makes "what code is X / what does code N cover"
answerable by direct lookup and link traversal instead of vector search.

Usage:
    python okf_build.py            # writes ./okf/ from ./data/*.pdf
"""

from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import yaml

from download_data import SOURCES

DATA_DIR = Path("data")
OKF_DIR = Path("okf")

LOB_MAP = {
    "commercial_property": "Commercial Property",
    "cgl": "Commercial General Liability",
    "personal_auto": "Personal Auto",
    "naics": "NAICS Classification",
}

# Human-readable document titles (PDF metadata titles are often junk like "aut-pp.qxp").
DOC_TITLES = {s["filename"]: s["description"] for s in SOURCES}

# The 20 NAICS sectors (identical in the USA 2022 and Canada 2017 editions).
NAICS_SECTORS = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management and Remediation Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}
_SECTOR_OF_PREFIX = {
    p: key
    for key in NAICS_SECTORS
    for p in (
        [str(n) for n in range(int(key[:2]), int(key[-2:]) + 1)] if "-" in key else [key]
    )
}

NAICS_EDITIONS = {
    "naics_usa": ("us", "NAICS USA 2022"),
    "naics_canada": ("ca", "NAICS Canada 2017"),
}

_LEVEL_TYPE = {
    2: "NAICS Sector",
    3: "NAICS Subsector",
    4: "NAICS Industry Group",
    5: "NAICS Industry",
    6: "NAICS National Industry",
}

# Running headers / footers that should never become part of a page body.
_NOISE = re.compile(
    r"^(census\.gov/naics|NAICS 2017 Version 1\.0|Statistics Canada - catalogue no\..*|"
    r"T—Canadian, Mexican, and United States industries are comparable\.)$"
)
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
# "325510 Paint and Coating Manufacturing", "446110  Pharmacies and drug stores  US"
_INLINE_HEADING = re.compile(r"^(\d{2,6})\s{1,3}([A-Z][^\n]{2,150}?)\s*(?:T|US|CAN|MEX)?$")
_CODE_ONLY = re.compile(r"^(\d{2,6})$")
_CODE_REF = re.compile(r"(?<!\d)(\d{2,6})(?!\d)")


def lob_of(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for prefix, label in LOB_MAP.items():
        if stem.startswith(prefix):
            return label
    return "Unknown"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def sector_key(code: str) -> str | None:
    return _SECTOR_OF_PREFIX.get(code[:2])


def parent_code(code: str) -> str | None:
    if len(code) <= 2:
        return None
    if len(code) == 3:
        return sector_key(code)
    return code[:-1]


def _clean_lines(text: str) -> list[str]:
    """Drop blank lines, running headers/footers, and the page number (a bare
    number within the first three lines — later bare numbers are NAICS codes)."""
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [
        line
        for idx, line in enumerate(raw_lines)
        if not _NOISE.match(line) and not (idx < 3 and _PAGE_NUMBER.match(line))
    ]


def _join_bullets(lines: list[str]) -> str:
    """Join PyMuPDF's split bullets ('•' on its own line) and soft-wrapped lines."""
    out: list[str] = []
    for line in lines:
        if out and out[-1] == "-":
            out[-1] = f"- {line}"
        elif line == "•":
            out.append("-")
        elif line.startswith("• "):
            out.append(f"- {line[2:]}")
        elif out and not out[-1].endswith((".", ":", ";")) and line[:1].islower():
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(l for l in out if l != "-")


# ── NAICS parsing ────────────────────────────────────────────────────────────

@dataclass
class NaicsEntry:
    code: str
    title: str
    page: int
    body_lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return _join_bullets(self.body_lines)


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    title = re.sub(r"\s*\.{3,}.*$", "", title)          # table-of-contents leaders
    title = re.sub(r"(?<=[a-z\)])(T|US|CAN|MEX)$", "", title)  # trailing comparability flag
    return title.strip()


def parse_naics_lines(pages: list[tuple[int, str]]) -> dict[str, NaicsEntry]:
    """
    Parse (page_number, page_text) pairs into code → entry.

    Recognises a heading either as "<code> <Title>" on one line or as a bare
    code line followed by its title line. A code is only accepted if its first
    two digits are a real NAICS sector (filters page numbers like "199").
    When a code appears more than once (contents pages, then the body), the
    occurrence with the longest description wins.
    """
    found: list[NaicsEntry] = []
    current: NaicsEntry | None = None

    for page_no, text in pages:
        lines = _clean_lines(text)
        i = 0
        while i < len(lines):
            line = lines[i]
            heading: tuple[str, str] | None = None
            m = _INLINE_HEADING.match(line)
            if m and sector_key(m.group(1)):
                heading = (m.group(1), m.group(2))
            else:
                m = _CODE_ONLY.match(line)
                if (
                    m
                    and sector_key(m.group(1))
                    and i + 1 < len(lines)
                    and lines[i + 1][:1].isupper()
                ):
                    heading = (m.group(1), lines[i + 1])
                    i += 1
            if heading:
                current = NaicsEntry(heading[0], _clean_title(heading[1]), page_no)
                found.append(current)
            elif current is not None:
                current.body_lines.append(line)
            i += 1

    entries: dict[str, NaicsEntry] = {}
    for e in found:
        if not e.title:
            continue
        prev = entries.get(e.code)
        if prev is None or len(e.body) > len(prev.body):
            entries[e.code] = e
    return entries


# ── Writing documents ────────────────────────────────────────────────────────

def write_doc(path: Path, frontmatter: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    path.write_text(f"---\n{fm}\n---\n\n{body.strip()}\n", encoding="utf-8")


def _first_sentence(text: str, limit: int = 220) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"(.{20,}?[.!?])(\s|$)", flat)
    s = m.group(1) if m else flat
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def build_naics(pdf_path: Path, out_root: Path) -> int:
    edition_key = next(k for k in NAICS_EDITIONS if pdf_path.stem.startswith(k))
    folder, edition = NAICS_EDITIONS[edition_key]
    base = out_root / "naics" / folder
    link_base = f"/naics/{folder}"

    doc = fitz.open(str(pdf_path))
    pages = [(i + 1, doc[i].get_text()) for i in range(len(doc))]
    doc.close()
    entries = parse_naics_lines(pages)

    # Sector pages are synthesised from the canonical sector list: bare 2-digit
    # numbers in running text produce unreliable sector "headings", so any
    # parsed sector entries are discarded and the page lists its subsectors.
    for key, name in NAICS_SECTORS.items():
        entries[key] = NaicsEntry(key, name, 0, [])
    for key in NAICS_SECTORS:
        if "-" in key:
            for p in range(int(key[:2]), int(key[-2:]) + 1):
                entries.pop(str(p), None)

    # Drop orphans (codes whose parent level never appears). In practice these
    # are retired codes quoted in a manual's concordance tables, not live codes.
    for level in (3, 4, 5, 6):
        for code in [c for c in entries if len(c) == level and c not in NAICS_SECTORS]:
            if parent_code(code) not in entries:
                del entries[code]

    children: dict[str, list[str]] = {}
    for code in entries:
        par = parent_code(code)
        if par and par in entries:
            children.setdefault(par, []).append(code)

    def link(code: str) -> str:
        return f"[{code} {entries[code].title}]({link_base}/{code}.md)"

    for code, e in entries.items():
        level = 2 if code in NAICS_SECTORS else len(code)
        par = parent_code(code)
        body_text = e.body
        refs = sorted(
            {c for c in _CODE_REF.findall(body_text) if c in entries and c != code}
        )

        parts = [f"# {code} — {e.title}", ""]
        if par and par in entries:
            parts += [f"Part of {link(par)}.", ""]
        if body_text:
            parts += ["## Description", "", body_text, ""]
        kids = sorted(children.get(code, []))
        if kids:
            parts += ["## Contains", ""] + [f"- {link(k)}" for k in kids] + [""]
        if refs:
            parts += ["## Related codes", ""] + [f"- {link(r)}" for r in refs] + [""]

        write_doc(
            base / f"{code}.md",
            {
                "type": _LEVEL_TYPE[level],
                "title": f"{code} {e.title}",
                "description": _first_sentence(body_text) if body_text else f"{edition} {_LEVEL_TYPE[level].lower()} {code}.",
                "resource": f"{pdf_path.name}#page={e.page}" if e.page else pdf_path.name,
                "tags": ["naics", folder, f"sector-{sector_key(code) or code}"],
                "lob": "NAICS Classification",
                "edition": edition,
                "code": code,
                "source": pdf_path.name,
                "page": e.page,
            },
            "\n".join(parts),
        )

    write_doc(
        base / "index.md",
        {
            "type": "Index",
            "title": f"{edition} — sectors",
            "description": f"Entry point for the {edition} classification hierarchy.",
            "tags": ["naics", folder],
            "lob": "NAICS Classification",
            "source": pdf_path.name,
        },
        "\n".join([f"# {edition}", ""] + [f"- {link(k)}" for k in NAICS_SECTORS]),
    )
    return len(entries)


def build_guide(pdf_path: Path, out_root: Path) -> int:
    lob = lob_of(pdf_path.name)
    stem = pdf_path.stem
    base = out_root / slug(lob) / slug(stem)
    link_base = f"/{slug(lob)}/{slug(stem)}"

    doc = fitz.open(str(pdf_path))
    toc = doc.get_toc()  # [level, title, page]
    doc_title = DOC_TITLES.get(pdf_path.name) or stem.replace("_", " ").title()
    written: list[tuple[int, str]] = []

    for i in range(len(doc)):
        page_no = i + 1
        lines = _clean_lines(doc[i].get_text())
        if not lines:
            continue
        body = _join_bullets(lines)
        if len(body) < 80:
            continue
        sections = [t[1].strip() for t in toc if t[2] == page_no and t[1].strip()]
        title = sections[0] if sections else _first_sentence(lines[0], 80)
        nav = [f"Part of [{doc_title}]({link_base}/index.md)."]
        if written:
            nav.append(f"Previous: [page {written[-1][0]}]({link_base}/p{written[-1][0]}.md).")
        write_doc(
            base / f"p{page_no}.md",
            {
                "type": "Guideline Page",
                "title": f"{doc_title} — p.{page_no}: {title}",
                "description": _first_sentence(body),
                "resource": f"{pdf_path.name}#page={page_no}",
                "tags": [slug(lob), slug(stem)] + [slug(s) for s in sections[:5]],
                "lob": lob,
                "source": pdf_path.name,
                "page": page_no,
            },
            "\n".join([f"# {title}", "", " ".join(nav), "", body]),
        )
        written.append((page_no, title))
    doc.close()

    write_doc(
        base / "index.md",
        {
            "type": "Index",
            "title": doc_title,
            "description": f"{lob} source document {pdf_path.name}, one OKF page per PDF page.",
            "resource": pdf_path.name,
            "tags": [slug(lob), slug(stem)],
            "lob": lob,
            "source": pdf_path.name,
        },
        "\n".join(
            [f"# {doc_title}", ""]
            + [f"- [p.{p}: {t}]({link_base}/p{p}.md)" for p, t in written]
        ),
    )
    return len(written)


def build_bundle(data_dir: Path = DATA_DIR, out_root: Path = OKF_DIR) -> dict[str, int]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"ERROR: No PDFs found in ./{data_dir}. Run download_data.py first.")
    if out_root.exists():
        shutil.rmtree(out_root)

    counts: dict[str, int] = {}
    by_lob: dict[str, list[str]] = {}
    for pdf in pdfs:
        if pdf.stem.startswith("naics"):
            counts[pdf.name] = build_naics(pdf, out_root)
        else:
            counts[pdf.name] = build_guide(pdf, out_root)
            by_lob.setdefault(lob_of(pdf.name), []).append(pdf.stem)
        print(f"  {pdf.name}: {counts[pdf.name]} docs")

    for lob, stems in by_lob.items():
        write_doc(
            out_root / slug(lob) / "index.md",
            {"type": "Index", "title": lob, "description": f"{lob} guideline sources.",
             "tags": [slug(lob)], "lob": lob},
            "\n".join([f"# {lob}", ""] + [f"- [{s}](/{slug(lob)}/{slug(s)}/index.md)" for s in stems]),
        )

    root = ["# P&C Underwriting Knowledge", ""]
    root += [f"- [{lob}](/{slug(lob)}/index.md)" for lob in by_lob]
    for key, (folder, edition) in NAICS_EDITIONS.items():
        if any(p.stem.startswith(key) for p in pdfs):
            root.append(f"- [{edition}](/naics/{folder}/index.md)")
    write_doc(
        out_root / "index.md",
        {"type": "Index", "title": "P&C Underwriting Knowledge",
         "description": "Root of the OKF bundle built from the underwriting corpus.",
         "tags": ["root"]},
        "\n".join(root),
    )
    return counts


if __name__ == "__main__":
    print(f"Building OKF bundle → ./{OKF_DIR}/")
    totals = build_bundle()
    print(f"Done: {sum(totals.values())} documents.")
