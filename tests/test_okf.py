from pathlib import Path

from okf_build import parse_naics_lines, parent_code, sector_key, write_doc
from okf_retrieve import OkfIndex, parse_doc

US_PAGE = """MANUFACTURING
199
T—Canadian, Mexican, and United States industries are comparable.
census.gov/naics
3255
Paint, Coating, and Adhesive ManufacturingT
This industry group comprises establishments primarily engaged in mixing pigments.
325510 Paint and Coating Manufacturing
This industry comprises establishments primarily engaged in mixing pigments, solvents, and binders.
Manufacturing caulking compounds--are classified in Industry 325520, Adhesive Manufacturing.
325520 Adhesive Manufacturing
This industry comprises establishments primarily engaged in manufacturing adhesives.
"""


def test_parse_naics_headings_and_page_numbers():
    entries = parse_naics_lines([(201, US_PAGE)])
    assert set(entries) == {"3255", "325510", "325520"}  # "199" page number ignored
    assert entries["3255"].title == "Paint, Coating, and Adhesive Manufacturing"
    assert "325520" in entries["325510"].body


def test_hierarchy_helpers():
    assert sector_key("325510") == "31-33"
    assert parent_code("325") == "31-33"
    assert parent_code("325510") == "32551"
    assert sector_key("199") is None


def _bundle(tmp_path: Path) -> OkfIndex:
    write_doc(tmp_path / "naics/us/524126.md",
              {"type": "NAICS National Industry", "title": "524126 Direct Property and Casualty Insurance Carriers",
               "lob": "NAICS Classification", "code": "524126", "source": "n.pdf", "page": 1},
              "Part of [52412 Direct Insurance](/naics/us/52412.md).\n\nUnderwriting property and casualty insurance.")
    write_doc(tmp_path / "naics/us/52412.md",
              {"type": "NAICS Industry", "title": "52412 Direct Insurance (except Life) Carriers",
               "lob": "NAICS Classification", "code": "52412"}, "Direct carriers.")
    write_doc(tmp_path / "personal-auto/g/p1.md",
              {"type": "Guideline Page", "title": "Uninsured motorist coverage", "lob": "Personal Auto",
               "source": "a.pdf", "page": 1}, "Uninsured motorist coverage pays when the other driver has no insurance.")
    write_doc(tmp_path / "index.md", {"type": "Index", "title": "root"}, "uninsured motorist")
    return OkfIndex.load(tmp_path)


def test_frontmatter_roundtrip(tmp_path):
    ix = _bundle(tmp_path)
    d = parse_doc(tmp_path / "naics/us/524126.md", tmp_path)
    assert d.meta["code"] == "524126" and d.link == "/naics/us/524126.md"
    assert len(ix.docs) == 4


def test_exact_code_lookup_and_link_expansion(tmp_path):
    hits = _bundle(tmp_path).search("What does 524126 cover?")
    links = [d.link for _, d in hits]
    assert links[0] == "/naics/us/524126.md" and hits[0][0] == 1.0
    assert "/naics/us/52412.md" in links  # parent pulled in via its markdown link
    assert "/personal-auto/g/p1.md" not in links  # no keyword filler around an exact hit


def test_bm25_respects_lob_filter_and_skips_index(tmp_path):
    ix = _bundle(tmp_path)
    hits = ix.search("uninsured motorist", top_k=3)
    assert hits[0][1].link == "/personal-auto/g/p1.md"
    assert all(d.meta["type"] != "Index" for _, d in hits)
    assert ix.search("uninsured motorist", lob_filter="NAICS Classification") == []
