"""
Download publicly available P&C insurance documents into ./data/

Sources:
  Commercial Property — CA DOI Commercial Insurance Guide, IL DOI Guide, Oregon DFR Guide
  CGL               — NY OGS CGL Policy Form (CG 00 01), Wisconsin DOI CGL Consumer Guide
  Personal Auto     — NAIC Consumer Guide, Texas DOI PA Statistical Plan, Florida DFS Toolkit
  NAICS             — US Census NAICS 2022 Manual, Statistics Canada NAICS 2017 Manual

Run once before ingestion:
    python download_data.py
"""

import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SOURCES = [
    # ── Commercial Property ──────────────────────────────────────────────────
    {
        "lob": "commercial_property",
        "filename": "commercial_property_il_doi_guide.pdf",
        "url": "https://insurance.illinois.gov/Docs/ConsumerTopics/PropertyAndCasualty/CommercialPropertyInsurance.pdf",
        "description": "Illinois DOI — Commercial Property Insurance Consumer Guide",
    },
    {
        "lob": "commercial_property",
        "filename": "commercial_property_ca_doi_guide.pdf",
        "url": "https://www.insurance.ca.gov/01-consumers/105-type/95-guides/09-comm/upload/Commercial-Insurance-Updated-090623.pdf",
        "description": "California DOI Commercial Insurance Consumer Guide (2023)",
    },
    {
        "lob": "commercial_property",
        "filename": "commercial_property_oregon_pc_guide.pdf",
        "url": "https://dfr.oregon.gov/help/Documents/commercial-insurance-property-casualty.pdf",
        "description": "Oregon DFR Guide to Commercial Property & Casualty Insurance",
    },
    # ── CGL ─────────────────────────────────────────────────────────────────
    {
        "lob": "cgl",
        "filename": "cgl_ny_policy_form.pdf",
        "url": "https://ogs.ny.gov/system/files/documents/2021/09/cg-00-01-01-96.pdf",
        "description": "ISO CGL Policy Form CG 00 01 (New York OGS filing)",
    },
    {
        "lob": "cgl",
        "filename": "cgl_wi_consumer_guide.pdf",
        "url": "https://oci.wi.gov/Documents/Consumers/PI-045.pdf",
        "description": "Wisconsin OCI Consumer Guide to Commercial Liability Insurance (PI-045)",
    },
    # ── Personal Auto ────────────────────────────────────────────────────────
    {
        "lob": "personal_auto",
        "filename": "personal_auto_naic_guide.pdf",
        "url": "https://content.naic.org/sites/default/files/publication-aut-pp-consumer-auto.pdf",
        "description": "NAIC Consumer's Guide to Auto Insurance",
    },
    {
        "lob": "personal_auto",
        "filename": "personal_auto_tx_statistical_plan.pdf",
        "url": "https://www.tdi.texas.gov/company/documents/ta_ppasp.pdf",
        "description": "Texas DOI Private Passenger Auto Statistical Plan & Reporting Manual",
    },
    {
        "lob": "personal_auto",
        "filename": "personal_auto_fl_toolkit.pdf",
        "url": "https://www.myfloridacfo.com/docs-sf/consumer-services-libraries/consumerservices-documents/understanding-coverage/consumer-guides/english---automobile-insurance-toolkit.pdf",
        "description": "Florida DFS Automobile Insurance Toolkit",
    },
    # ── NAICS Classification ─────────────────────────────────────────────────
    {
        "lob": "naics",
        "filename": "naics_usa_2022_manual.pdf",
        "url": "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf",
        "description": "US Census Bureau — NAICS 2022 Manual (North American Industry Classification System)",
    },
    {
        "lob": "naics",
        "filename": "naics_canada_2017_manual.pdf",
        "url": "https://www150.statcan.gc.ca/n1/pub/12-501-x/12-501-x2016001-eng.pdf",
        "description": "Statistics Canada — NAICS Canada 2017 Classification Manual",
    },
]


def _progress(downloaded: int, chunk: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {downloaded // 1024} KB downloaded")
    else:
        pct = min(100, downloaded * 100 // total)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({downloaded // 1024} / {total // 1024} KB)")
    sys.stdout.flush()


def download(source: dict) -> bool:
    dest = DATA_DIR / source["filename"]
    if dest.exists():
        print(f"  [skip] {source['filename']} already exists")
        return True

    print(f"\n{source['description']}")
    print(f"  -> {source['url']}")
    try:
        req = urllib.request.Request(
            source["url"],
            headers={"User-Agent": "Mozilla/5.0 (compatible; InsuranceRAG/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _progress(downloaded, len(chunk), total)
        print(f"\n  Saved: {dest} ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    print("=== Downloading P&C Insurance Guideline PDFs ===\n")
    ok, fail = 0, 0
    for src in SOURCES:
        if download(src):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*48}")
    print(f"Downloaded: {ok}  |  Failed: {fail}")
    pdfs = list(DATA_DIR.glob("*.pdf"))
    print(f"PDFs in ./data/: {len(pdfs)}")
    for p in sorted(pdfs):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
