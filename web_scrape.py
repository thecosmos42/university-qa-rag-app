"""
VU Amsterdam Masters PDF Scraper — Optimised Final
====================================================
Phase 1 : GET /api/studyguide/en/Master/2025-2026/studies → all 89 programs instantly
          No browser needed. API returns name, slug, code (=studyCode) for every program.

Phase 2 : One shared browser session to:
            a) Grab CSRF token + session cookies (single page load, reused for all POSTs)
            b) Find TER/OER PDF links (one page visit per program, tab iteration)

Phase 3 : 
            - Studiegids: POST /nl/pdf/Study per program (parallel, shared CSRF session)
            - TER:        GET CDN link per program (parallel, deduplicated by URL)

Naming:   Studiegids_Masters_[CourseName].pdf
          TER_Masters_[CourseName].pdf

Requirements:
    pip install playwright httpx tqdm
    playwright install chromium
"""

import asyncio
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import httpx
from playwright.async_api import async_playwright, BrowserContext

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_URL        = "https://studiegids.vu.nl"
API_URL         = f"{BASE_URL}/api/studyguide/en/Master/2025-2026/studies"
PDF_POST_URL    = f"{BASE_URL}/nl/pdf/Study"
OUTPUT_DIR      = Path("university_docs")
CHECKPOINT_FILE = Path("progress.json")
RESULTS_FILE    = Path("scraped_results.json")

CONCURRENT_PAGES     = 3      # parallel Playwright pages for TER scraping
CONCURRENT_DOWNLOADS = 5      # parallel POST + GET downloads
PAGE_DELAY_S         = 1.5    # polite gap between page visits
PAGE_TIMEOUT_MS      = 30_000
RENDER_WAIT_MS       = 2_500  # settle time after navigation

HEADLESS = True  # set False to watch the browser


# ── Helpers ────────────────────────────────────────────────────────────────────

def course_name_from_slug(slug: str) -> str:
    """'science-business-and-innovation' → 'ScienceBusinessAndInnovation'"""
    return "".join(w.capitalize() for w in slug.replace("-", " ").split())


def make_filename(doc_type: str, course_name: str) -> str:
    """[DocType]_Masters_[CourseName].pdf"""
    return f"{doc_type}_Masters_{course_name}.pdf"


def program_url(slug: str) -> str:
    return f"{BASE_URL}/en/Master/2025-2026/{slug}"


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"completed": [], "failed": []}


def save_checkpoint(cp: dict):
    CHECKPOINT_FILE.write_text(json.dumps(cp, indent=2))


def pw_cookies_to_dict(cookies: list[dict]) -> dict:
    return {c["name"]: c["value"] for c in cookies}


def write_sidecar(filepath: Path, metadata: dict, size_bytes: int):
    sidecar = {
        **metadata,
        "file_size_bytes": size_bytes,
        "scrape_date":     datetime.now().isoformat(),
        "filename":        filepath.name,
    }
    filepath.with_suffix(".pdf.json").write_text(json.dumps(sidecar, indent=2))


# ── Phase 1: Fetch all programs from API ──────────────────────────────────────

async def phase1_fetch_programs() -> list[dict]:
    """
    Single API call — returns all 89 programs with name, slug, code, faculty.
    No browser required.
    """
    print("\n📋 Phase 1 — Fetching all programs from API...")
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
        follow_redirects=True,
    ) as client:
        r = await client.get(API_URL)
        r.raise_for_status()
        programs = r.json()

    # Normalise into the shape the rest of the script expects
    normalised = [
        {
            "name":        p["name"],
            "slug":        p["slug"],
            "code":        p["code"],          # = studyCode for POST
            "degree":      p["degree"],        # = studyType for POST  (always "Master")
            "faculty":     p["faculty"]["name"],
            "url":         program_url(p["slug"]),
            "course_name": course_name_from_slug(p["slug"]),
        }
        for p in programs
        if p.get("studyLanguage") == ["EN"] and p.get("faculty")['name'] == "Faculty of Science"
    ]

    print(f"   ✅ {len(normalised)} programs retrieved")
    return normalised


# ── Phase 2a: Grab shared CSRF token + cookies ────────────────────────────────

async def phase2a_get_csrf_session(context: BrowserContext) -> tuple[str, list[dict]]:
    """
    Load ONE program page to harvest the CSRF token and session cookies.
    These are reused for every Studiegids POST — the server only checks
    that the token matches the session, not that it came from a specific page.
    """
    print("\n🔑 Phase 2a — Acquiring shared CSRF token...")
    page = await context.new_page()

    # Use the first program page — any will do
    sample_url = f"{BASE_URL}/en/Master/2025-2026/accounting-and-control-ft"
    await page.goto(sample_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
    await page.wait_for_timeout(RENDER_WAIT_MS)

    csrf_token = await page.evaluate("""
        () => document.querySelector('[name="__RequestVerificationToken"]')?.value ?? null
    """)

    if not csrf_token:
        raise RuntimeError("Could not find CSRF token on page — check if site structure changed")

    cookies = await context.cookies(urls=[BASE_URL])
    await page.close()

    print(f"   ✅ CSRF token acquired ({csrf_token[:20]}...)")
    return csrf_token, cookies


# ── Phase 2b: Scrape TER link for one program ─────────────────────────────────

async def phase2b_scrape_ter(
    context: BrowserContext,
    program: dict,
    sem: asyncio.Semaphore,
) -> str | None:
    """
    Visit each tab of a program page looking for the TER/OER PDF link.
    Primary signal: href contains 'kc-usercontent.com' (Kentico CDN).
    """
    async with sem:
        await asyncio.sleep(PAGE_DELAY_S)
        page = await context.new_page()
        ter_url = None

        try:
            for tab in [1, 2, 3, 4]:
                tab_url = f"{program['url']}#/tab={tab}"
                await page.goto(tab_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                await page.wait_for_timeout(RENDER_WAIT_MS)

                ter_url = await page.evaluate("""
                    () => {
                        // Primary: Kentico CDN — always OER/TER
                        const cdn = document.querySelector("a[href*='kc-usercontent.com']");
                        if (cdn) return cdn.href;

                        // Fallback: PDF link mentioning OER/TER in text or href
                        const all = [...document.querySelectorAll("a[href$='.pdf']")];
                        const match = all.find(a =>
                            /\\boer\\b/i.test(a.href + a.textContent) ||
                            /\\bter\\b/i.test(a.textContent) ||
                            /teaching.and.examination/i.test(a.textContent)
                        );
                        return match ? match.href : null;
                    }
                """)

                if ter_url:
                    break

        except Exception as e:
            print(f"   ⚠️  TER scrape error [{program['name']}]: {e}")
        finally:
            await page.close()

        status = "✓" if ter_url else "✗"
        print(f"   TER={status}  {program['name'][:60]}")
        return ter_url


# ── Phase 3a: Download Studiegids via POST ────────────────────────────────────

async def phase3a_download_studiegids(
    client: httpx.AsyncClient,
    program: dict,
    csrf_token: str,
    cookies: dict,
    sem: asyncio.Semaphore,
) -> bool:
    """
    POST to /nl/pdf/Study using:
      - studyCode, slug, studyType from the API response (no page scraping needed)
      - CSRF token + session cookies from the shared browser session
    """
    async with sem:
        course   = program["course_name"]
        filepath = OUTPUT_DIR / make_filename("Studiegids", course)

        payload = {
            "studyCode":                  program["code"],
            "studyYear":                  "2025-2026",
            "slug":                       program["slug"],
            "studyType":                  program["degree"],
            "moduleGroupCode":            "",
            "__RequestVerificationToken": csrf_token,
        }

        try:
            r = await client.post(
                PDF_POST_URL,
                data=payload,
                cookies=cookies,
                headers={
                    "Referer":      program["url"],
                    "Origin":       BASE_URL,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                timeout=60,
                follow_redirects=True,
            )
            r.raise_for_status()

            content_type = r.headers.get("content-type", "")
            if "pdf" not in content_type:
                print(f"   ❌ Studiegids POST returned non-PDF [{program['name']}]: "
                      f"{content_type} — {r.text[:150]}")
                return False

            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(r.content)
            write_sidecar(filepath, {
                "doc_type":     "Studiegids",
                "program_type": "Masters",
                "program_name": program["name"],
                "course_name":  course,
                "faculty":      program["faculty"],
                "program_url":  program["url"],
                "post_url":     PDF_POST_URL,
                "study_code":   program["code"],
                "academic_year": "2025-2026",
            }, len(r.content))

            print(f"   ⬇️  Studiegids: {filepath.name}  ({len(r.content)//1024}KB)")
            return True

        except Exception as e:
            print(f"   ❌ Studiegids failed [{program['name']}]: {e}")
            return False


# ── Phase 3b: Download TER via GET ───────────────────────────────────────────

async def phase3b_download_ter(
    client: httpx.AsyncClient,
    program: dict,
    ter_url: str,
    filepath: Path,
    sem: asyncio.Semaphore,
) -> bool:
    async with sem:
        try:
            r = await client.get(ter_url, timeout=60, follow_redirects=True)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "")
            if "pdf" not in content_type:
                print(f"   ❌ TER returned non-PDF [{program['name']}]: {content_type}")
                return False

            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(r.content)
            write_sidecar(filepath, {
                "doc_type":     "TER",
                "program_type": "Masters",
                "program_name": program["name"],
                "course_name":  program["course_name"],
                "faculty":      program["faculty"],
                "program_url":  program["url"],
                "pdf_url":      ter_url,
                "academic_year": "2025-2026",
            }, len(r.content))

            print(f"   ⬇️  TER:         {filepath.name}  ({len(r.content)//1024}KB)")
            return True

        except Exception as e:
            print(f"   ❌ TER failed [{program['name']}]: {e}")
            return False


# ── Orchestrator ───────────────────────────────────────────────────────────────

async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cp = load_checkpoint()
    completed_slugs = set(cp["completed"])

    # ── Phase 1: API call — all 89 programs, no browser ──────────────────────
    all_programs = await phase1_fetch_programs()
    remaining    = [p for p in all_programs if p["slug"] not in completed_slugs]
    skipped      = len(all_programs) - len(remaining)
    if skipped:
        print(f"   ⏭️  Skipping {skipped} already completed programs")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )

        # ── Phase 2a: One page load → shared CSRF token + cookies ────────────
        csrf_token, pw_cookies = await phase2a_get_csrf_session(context)
        cookies_dict = pw_cookies_to_dict(pw_cookies)

        # ── Phase 2b: Scrape TER links (parallel, rate-limited) ──────────────
        print(f"\n🔍 Phase 2b — Scraping TER links for {len(remaining)} programs "
              f"({CONCURRENT_PAGES} parallel)...\n")
        sem_pages = asyncio.Semaphore(CONCURRENT_PAGES)
        ter_urls: list[str | None] = await asyncio.gather(
            *[phase2b_scrape_ter(context, p, sem_pages) for p in remaining]
        )

        await browser.close()

    # Attach TER URLs to program dicts
    for prog, ter in zip(remaining, ter_urls):
        prog["ter_pdf"] = ter

    # Persist scrape results
    RESULTS_FILE.write_text(json.dumps(remaining, indent=2))
    found_ter = sum(1 for t in ter_urls if t)
    print(f"\n💾 Results saved → {RESULTS_FILE}")
    print(f"   TER links found: {found_ter}/{len(remaining)}")

    # ── Phase 3: Download everything in parallel ──────────────────────────────
    print(f"\n⬇️  Phase 3 — Downloading PDFs ({CONCURRENT_DOWNLOADS} parallel)...\n")
    sem_dl = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
    dl_tasks = []
    seen_ter: dict[str, Path] = {}   # deduplicate shared TER PDFs by URL

    async with httpx.AsyncClient(follow_redirects=True) as client:

        for prog in remaining:
            course = prog["course_name"]

            # Studiegids — unique POST per program (API gave us all fields we need)
            dl_tasks.append(
                phase3a_download_studiegids(client, prog, csrf_token, cookies_dict, sem_dl)
            )

            # TER — deduplicate by URL (same faculty = same PDF)
            if prog["ter_pdf"]:
                ter_path = OUTPUT_DIR / make_filename("TER", course)
                if prog["ter_pdf"] not in seen_ter:
                    seen_ter[prog["ter_pdf"]] = ter_path
                    dl_tasks.append(
                        phase3b_download_ter(client, prog, prog["ter_pdf"], ter_path, sem_dl)
                    )
                else:
                    # Copy from already-downloaded source
                    src = seen_ter[prog["ter_pdf"]]
                    async def _copy(src=src, dst=ter_path, p=prog):
                        for _ in range(20):
                            if src.exists():
                                break
                            await asyncio.sleep(2)
                        if not src.exists():
                            print(f"   ❌ TER copy source never appeared: {src}")
                            return False
                        shutil.copy2(src, dst)
                        write_sidecar(dst, {
                            "doc_type":     "TER",
                            "program_type": "Masters",
                            "program_name": p["name"],
                            "course_name":  p["course_name"],
                            "faculty":      p["faculty"],
                            "program_url":  p["url"],
                            "pdf_url":      p["ter_pdf"],
                            "academic_year": "2025-2026",
                            "note": "Shared TER — copied from same faculty",
                        }, src.stat().st_size)
                        print(f"   📋 TER copied:  {dst.name}")
                        return True
                    dl_tasks.append(_copy())

        results = await asyncio.gather(*dl_tasks)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    for prog in remaining:
        cp["completed"].append(prog["slug"])
    save_checkpoint(cp)

    # ── Summary ───────────────────────────────────────────────────────────────
    success = sum(1 for r in results if r)
    print(f"\n{'='*55}")
    print(f"✅ Files written : {success}/{len(dl_tasks)}")
    print(f"❌ Failed        : {len(dl_tasks) - success}")
    print(f"📁 Output dir    : {OUTPUT_DIR.resolve()}")

    # Programs where TER wasn't found — print for manual review
    no_ter = [p for p in remaining if not p["ter_pdf"]]
    if no_ter:
        print(f"\n⚠️  TER not found for {len(no_ter)} programs:")
        for p in no_ter:
            print(f"   • {p['name']:50s} {p['url']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run():
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n⛔ Interrupted — re-run to resume from checkpoint.")
    finally:
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    run()