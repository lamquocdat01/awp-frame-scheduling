"""Verify the topic-03 (AwP v3) bibliography against registries and emit refs.bib + refs_verify.csv.

Registries:
  * Crossref REST API  (https://api.crossref.org/works/<doi>)          ~1 req/s
  * arXiv export API   (http://export.arxiv.org/api/query?id_list=<id>) >= 3 s between calls
  * Plain HTTP GET for policy pages (URL-ONLY entries)

Every raw response is cached under $THS_DATASETS/_derived/43/cache/lit/ and re-used;
nothing is re-queried when a cache file exists.

Usage:
  python v3_verify_refs.py                   # verify all entries, write Revision_v3/refs.bib + lit/refs_verify_v3.csv
  (adapted from 43_RefScreen_Scientometrics/code/verify_refs.py; entries in v3_refs_spec.py)
  python verify_refs.py --limit 5            # only first 5 entries (no files written unless --write)
  python verify_refs.py --search "query" [--container "Scientometrics"] [--from 2023] [--rows 10]
  python verify_refs.py --arxiv-search "ti:hallucinator"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from v3_refs_spec import ACR, ENTRIES

MAILTO = "noreply@example.org"
UA = f"awp-03-v3/0.1 (mailto:{MAILTO})"
DATASETS = Path(os.environ.get("THS_DATASETS", r"D:\Datasets"))
CACHE = DATASETS / "_derived" / "03" / "cache" / "lit"
TOPIC = Path(__file__).resolve().parents[1]
LIT = TOPIC / "lit"
TODAY = dt.date.today().isoformat()

_last_call: dict[str, float] = {}
_MIN_GAP = {"crossref": 1.1, "arxiv": 3.2, "web": 1.0}


# --------------------------------------------------------------------------- HTTP + cache
def _cache_path(kind: str, url: str, ext: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])[:80]
    return CACHE / kind / f"{slug}__{h}.{ext}"


def _throttle(kind: str) -> None:
    gap = _MIN_GAP[kind]
    wait = _last_call.get(kind, 0.0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[kind] = time.time()


def cached_get(kind: str, url: str, ext: str = "json") -> dict:
    """Return {'status': int, 'text': str, 'final_url': str} from cache or network."""
    path = _cache_path(kind, url, ext + ".cache.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        _throttle(kind)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=40, allow_redirects=True)
            rec = {"url": url, "status": r.status_code, "final_url": r.url,
                   "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
                   "text": r.text if (kind != "web" or r.status_code != 200) else r.text[:20000]}
        except requests.RequestException as exc:
            rec = {"url": url, "status": -1, "final_url": url, "text": str(exc)}
        if rec["status"] not in (-1, 429, 500, 502, 503, 504):
            break
        if kind == "web":
            return rec  # do not hammer third-party sites
        time.sleep(10 * (attempt + 1))  # back off on transient errors
    if rec["status"] in (-1, 429, 500, 502, 503, 504):  # transient: do not cache
        print(f"  transient HTTP {rec['status']} for {url}", file=sys.stderr)
        return rec
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def crossref_work(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}?mailto={MAILTO}"
    rec = cached_get("crossref", url)
    if rec["status"] != 200:
        return None
    return json.loads(rec["text"])["message"]


def crossref_search(query: str, container: str | None = None, from_year: str | None = None,
                    rows: int = 10, until: str | None = None) -> list[dict]:
    params = {"query.bibliographic": query, "rows": str(rows), "mailto": MAILTO,
              "select": "DOI,title,issued,container-title,author,type"}
    filters = []
    if from_year:
        filters.append(f"from-pub-date:{from_year}")
    if until:
        filters.append(f"until-pub-date:{until}")
    if filters:
        params["filter"] = ",".join(filters)
    if container:
        params["query.container-title"] = container
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    rec = cached_get("crossref", url)
    if rec["status"] != 200:
        print("crossref search failed", rec["status"], file=sys.stderr)
        return []
    return json.loads(rec["text"])["message"]["items"]


_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _parse_arxiv(text: str) -> list[dict]:
    root = ET.fromstring(text)
    out = []
    for e in root.findall("a:entry", _ATOM):
        eid = (e.findtext("a:id", "", _ATOM) or "").strip()
        title = " ".join((e.findtext("a:title", "", _ATOM) or "").split())
        if not eid or "api/errors" in eid or not title or title == "Error":
            continue
        out.append({
            "id": eid.rsplit("/abs/", 1)[-1],
            "title": title,
            "published": e.findtext("a:published", "", _ATOM),
            "authors": [a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)],
            "journal_ref": e.findtext("arxiv:journal_ref", "", _ATOM),
            "doi": e.findtext("arxiv:doi", "", _ATOM),
            "primary": (e.find("arxiv:primary_category", _ATOM).attrib.get("term", "")
                        if e.find("arxiv:primary_category", _ATOM) is not None else ""),
        })
    return out


def arxiv_record(aid: str) -> dict | None:
    rec = cached_get("arxiv", f"http://export.arxiv.org/api/query?id_list={aid}", "xml")
    if rec["status"] != 200:
        return None
    items = _parse_arxiv(rec["text"])
    return items[0] if items else None


def arxiv_search(q: str, n: int = 10) -> list[dict]:
    url = ("http://export.arxiv.org/api/query?" +
           urllib.parse.urlencode({"search_query": q, "max_results": n}))
    rec = cached_get("arxiv", url, "xml")
    return _parse_arxiv(rec["text"]) if rec["status"] == 200 else []


def url_check(url: str) -> tuple[int, str]:
    rec = cached_get("web", url, "html")
    return rec["status"], rec.get("final_url", url)


# --------------------------------------------------------------------------- matching
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"<[^>]+>", " ", s)          # JATS tags in Crossref titles
    s = s.replace("{", "").replace("}", "").replace("\\&", "&")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def cr_years(msg: dict) -> set[int]:
    ys = set()
    for k in ("issued", "published-print", "published-online", "published"):
        dp = (msg.get(k) or {}).get("date-parts") or [[None]]
        if dp[0] and dp[0][0]:
            ys.add(int(dp[0][0]))
    return ys


def cr_title(msg: dict) -> str:
    t = (msg.get("title") or [""])[0]
    sub = (msg.get("subtitle") or [""])
    if sub and sub[0] and norm(sub[0]) not in norm(t):
        t = f"{t}: {sub[0]}"
    return " ".join(t.split())


def find_published_version(title: str, authors: list[str]) -> str:
    """Crossref query.bibliographic for a journal/proceedings version of a preprint."""
    q = title + " " + " ".join(a.split()[-1] for a in authors[:2])
    for it in crossref_search(q, rows=5):
        if it.get("type") in ("posted-content",):
            continue
        t = (it.get("title") or [""])[0]
        if sim(t, title) >= 0.9 and not it["DOI"].lower().startswith("10.48550"):
            venue = (it.get("container-title") or [""])[0]
            return f"{venue} doi:{it['DOI']}"
    return ""


# --------------------------------------------------------------------------- verification
def verify(e: dict) -> dict:
    """Return csv row and fill e['_fields'] (bib fields from the registry)."""
    row = {"key": e["key"], "doi": "", "title": e["title"], "year": e.get("year", ""),
           "venue": e.get("venue", ""), "status": "UNVERIFIED", "checked_at": TODAY}
    kind = e["kind"]
    if kind == "doi":
        msg = crossref_work(e["doi"])
        if msg is None and e.get("doi_alt"):
            msg = crossref_work(e["doi_alt"])
            if msg is not None:
                e["_corrected"] = True
                e["doi"] = e["doi_alt"]
        if msg is None:
            e["_status"] = "UNVERIFIED"
            row["doi"] = e["doi"]
            return row
        reg_title = cr_title(msg)
        s = sim(e["title"], reg_title)
        years = cr_years(msg)
        row["doi"] = msg["DOI"].lower()
        row["venue"] = (msg.get("container-title") or [""])[0] or msg.get("publisher", "")
        y = int(e.get("year") or 0)
        ok_year = y in years
        if s >= 0.9 and ok_year:
            status = "CORRECTED" if e.get("_corrected") else "OK"
        else:
            # trust the registry: correct title/year and mark CORRECTED
            print(f"  [{e['key']}] mismatch sim={s:.2f} years={sorted(years)} spec_year={y}\n"
                  f"     spec: {e['title']}\n     reg : {reg_title}")
            e["title_bib"] = e.get("title_bib") if s >= 0.9 else None
            e["title"] = reg_title
            if not ok_year:
                issued = (msg.get("issued") or {}).get("date-parts", [[None]])[0][0]
                e["year"] = str(issued)
            status = "CORRECTED"
        row["title"] = e["title"]
        row["year"] = e["year"]
        e["_status"] = status
        e["_msg"] = msg
    elif kind == "arxiv":
        rec = arxiv_record(e["arxiv"])
        row["doi"] = f"10.48550/arXiv.{e['arxiv']}"
        if rec is None:
            e["_status"] = "UNVERIFIED"
            return row
        s = sim(e["title"], rec["title"])
        y = rec["published"][:4]
        status = "OK"
        if s < 0.9 or str(e.get("year")) != y:
            print(f"  [{e['key']}] mismatch sim={s:.2f} arxiv_year={y} spec_year={e.get('year')}\n"
                  f"     spec: {e['title']}\n     reg : {rec['title']}")
            if s < 0.9:
                e["title"] = rec["title"]
                e["title_bib"] = None
            e["year"] = y
            status = "CORRECTED"
        row.update(title=e["title"], year=e["year"])
        pub = (rec.get("journal_ref") or "").strip()
        if rec.get("doi"):
            pub = (pub + " doi:" + rec["doi"]).strip()
        if not pub:
            pub = find_published_version(rec["title"], rec["authors"])
        row["venue"] = "arXiv " + rec["primary"] + (f"; published version: {pub}" if pub else "")
        e["_status"] = status
        e["_arxiv"] = rec
    elif kind == "datacite":  # LIPIcs (Dagstuhl) DOIs are registered with DataCite, not Crossref
        url = "https://api.datacite.org/dois/" + urllib.parse.quote(e["doi"], safe="")
        rec = cached_get("crossref", url)
        row["doi"] = e["doi"].lower()
        if rec["status"] != 200:
            e["_status"] = "UNVERIFIED"
            return row
        att = json.loads(rec["text"])["data"]["attributes"]
        reg_title = att["titles"][0]["title"]
        ok = sim(e["title"], reg_title) >= 0.9 and str(att.get("publicationYear")) == str(e["year"])
        row["venue"] = (att.get("container") or {}).get("title", "") + " (DataCite)"
        e["_status"] = "OK" if ok else "CORRECTED"
        if not ok:
            e["title"], e["year"] = reg_title, str(att.get("publicationYear"))
        row.update(title=e["title"], year=e["year"])
        e["_dc"] = att
    elif kind == "url":
        code, final = url_check(e["url"])
        e["_http"] = code
        reachable = code == 200 or (code in (401, 403, 406, 429) and e.get("reachable_note"))
        e["_status"] = "URL-ONLY" if (code == 200 or reachable) else "UNVERIFIED"
        if code == 200:  # v3: official proceedings page must carry the cited title -> OK-URL
            page = norm(cached_get("web", e["url"], "html").get("text", ""))
            if norm(e["title"]) in page:
                e["_status"] = "OK-URL"
        row["venue"] = (e.get("venue", "") + f"; HTTP {code}").strip("; ")
        row["doi"] = ""
    else:  # 'unverified' placeholder (no registry record found)
        e["_status"] = "UNVERIFIED"
    row["status"] = e["_status"]
    return row


# --------------------------------------------------------------------------- BibTeX
def bib_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


def protect(title: str, words: list[str]) -> str:
    t = bib_escape(title)
    for w in sorted(words, key=len, reverse=True):
        t = re.sub(rf"(?<![{{\w]){re.escape(w)}(?![\w}}])", "{" + w + "}", t)
    return t


def cr_authors(msg: dict) -> str:
    out = []
    for a in msg.get("author", []):
        if "family" in a:
            out.append(f"{a['family']}, {a.get('given', '')}".strip(", "))
        elif "name" in a:
            out.append("{" + a["name"] + "}")
    return " and ".join(out)


def arxiv_authors(rec: dict) -> str:
    out = []
    for n in rec["authors"]:
        parts = n.split()
        out.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n)
    return " and ".join(out)


def to_bib(e: dict) -> str:
    st = e["_status"]
    title = e.get("title_bib") or protect(e["title"], ACR + e.get("protect", []))
    f: list[tuple[str, str]] = []
    if e["kind"] == "doi" and "_msg" in e:
        m = e["_msg"]
        typ = {"journal-article": "article", "proceedings-article": "inproceedings",
               "book-chapter": "incollection", "posted-content": "misc"}.get(m.get("type"), "misc")
        f.append(("author", e.get("author") or cr_authors(m)))
        f.append(("title", title))
        cont = (m.get("container-title") or [""])[0]
        if typ == "article":
            f.append(("journal", bib_escape(e.get("journal") or cont)))
        elif typ in ("inproceedings", "incollection"):
            f.append(("booktitle", bib_escape(cont)))
        f.append(("year", str(e["year"])))
        for k_bib, k_cr in (("volume", "volume"), ("number", "issue"), ("pages", "page")):
            if m.get(k_cr):
                f.append((k_bib, m[k_cr].replace("-", "--") if k_bib == "pages" else m[k_cr]))
        if typ == "misc":
            hp = e.get("journal") or (m.get("group-title") or "") or                 ((m.get("institution") or [{}])[0].get("name", ""))
            f.append(("howpublished", bib_escape(hp)))
        if m.get("publisher") and typ != "article":
            f.append(("publisher", bib_escape(m["publisher"])))
        f.append(("doi", m["DOI"].lower()))
    elif e["kind"] == "datacite" and "_dc" in e:
        typ = "inproceedings"
        f += [("author", e.get("author", "")), ("title", title),
              ("booktitle", bib_escape(e.get("booktitle") or (e["_dc"].get("container") or {}).get("title", ""))),
              ("year", str(e["year"])), ("publisher", "Schloss Dagstuhl -- Leibniz-Zentrum f{\\\"u}r Informatik"),
              ("doi", e["doi"].lower())]
    elif e["kind"] == "arxiv" and "_arxiv" in e:
        typ = "misc"
        rec = e["_arxiv"]
        f += [("author", e.get("author") or arxiv_authors(rec)), ("title", title),
              ("year", str(e["year"])), ("eprint", e["arxiv"]), ("archivePrefix", "arXiv"),
              ("primaryClass", rec["primary"]), ("doi", f"10.48550/arXiv.{e['arxiv']}")]
    elif e.get("booktitle"):  # v3: proceedings without DOI -> @inproceedings + official url
        typ = "inproceedings"
        f += [("author", e.get("author", "")), ("title", title), ("booktitle", e["booktitle"]),
              ("year", str(e.get("year", ""))), ("volume", e.get("volume", "")),
              ("series", e.get("series", "")), ("pages", e.get("pages", "")),
              ("url", e.get("url", "")), ("urldate", TODAY)]
    else:
        typ = "misc"
        f += [("author", e.get("author", "")), ("title", title)]
        if e.get("howpublished"):
            f.append(("howpublished", e["howpublished"]))
        f.append(("year", str(e.get("year", ""))))
        if e.get("url"):
            f += [("url", e["url"]), ("urldate", TODAY)]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f if v)
    head = ""
    if st == "UNVERIFIED":
        head = ("% UNVERIFIED -- no registry record found (" + TODAY + "). "
                "Do NOT cite its numbers until a DOI resolves.\n")
    return f"{head}@{typ}{{{e['key']},\n{body}\n}}\n"


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=43, help="unused (kept for pipeline uniformity)")
    ap.add_argument("--limit", type=int, default=0, help="verify only the first N entries")
    ap.add_argument("--write", action="store_true", help="write outputs even with --limit")
    ap.add_argument("--search", help="Crossref query.bibliographic search (discovery)")
    ap.add_argument("--container", help="query.container-title for --search")
    ap.add_argument("--from", dest="from_", help="from-pub-date filter for --search")
    ap.add_argument("--until", help="until-pub-date filter for --search")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--arxiv-search", help="arXiv API search_query (discovery)")
    ap.add_argument("--doi", help="print one Crossref record summary")
    ap.add_argument("--arxiv", help="print one arXiv record summary")
    ap.add_argument("--url", help="HTTP-check one URL")
    a = ap.parse_args()
    random.seed(a.seed)
    CACHE.mkdir(parents=True, exist_ok=True)

    if a.search:
        for it in crossref_search(a.search, a.container, a.from_, a.rows, a.until):
            au = "; ".join(x.get("family", x.get("name", "")) for x in it.get("author", [])[:4])
            y = (it.get("issued") or {}).get("date-parts", [[None]])[0][0]
            print(f"{it['DOI']} | {y} | {(it.get('container-title') or [''])[0]} | "
                  f"{(it.get('title') or [''])[0]} | {au} | {it.get('type')}")
        return
    if a.arxiv_search:
        for r in arxiv_search(a.arxiv_search, a.rows):
            print(f"{r['id']} | {r['published'][:10]} | {r['title']} | {'; '.join(r['authors'][:4])}"
                  f" | jref={r['journal_ref']} doi={r['doi']}")
        return
    if a.doi:
        m = crossref_work(a.doi)
        print(json.dumps(None if m is None else {
            "DOI": m["DOI"], "title": cr_title(m), "years": sorted(cr_years(m)),
            "container": m.get("container-title"), "type": m.get("type"),
            "authors": cr_authors(m), "vol": m.get("volume"), "page": m.get("page")},
            ensure_ascii=False, indent=1))
        return
    if a.arxiv:
        print(json.dumps(arxiv_record(a.arxiv), ensure_ascii=False, indent=1))
        return
    if a.url:
        print(url_check(a.url))
        return

    entries = [dict(e) for e in ENTRIES]
    if a.limit:
        entries = entries[: a.limit]
    rows = []
    for e in entries:
        rows.append(verify(e))
        print(f"{e['_status']:10s} {e['key']}")
    if a.limit and not a.write:
        return
    LIT.mkdir(exist_ok=True)
    header = ("% refs.bib -- topic 03 (AwP v3)\n"
              f"% Generated by code/v3_verify_refs.py on {TODAY}; registry-verified "
              "(Crossref / arXiv API / HTTP). See lit/refs_verify_v3.csv.\n\n")
    (TOPIC / "Revision_v3" / "refs.bib").write_text(header + "\n".join(to_bib(e) for e in entries), encoding="utf-8")
    with open(LIT / "refs_verify_v3.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "doi", "title", "year", "venue", "status", "checked_at"])
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"{len(rows)} entries:", counts)


if __name__ == "__main__":
    main()
