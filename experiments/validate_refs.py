#!/usr/bin/env python3
"""
Mechanical reference validation for the entries ADDED in this revision.

For each new BibTeX entry with a DOI, fetch the canonical metadata from the
Crossref REST API (api.crossref.org/works/{doi}) and assert that the BibTeX
title, first-author family name, year and venue agree after whitespace/accent
normalisation. The arXiv preprint is checked against the arXiv API. Entries
without a resolvable identifier are reported as UNVERIFIED (never silently
passed). Exits non-zero on any hard mismatch.
"""
import os
import re
import sys
import json
import time
import unicodedata
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = os.path.join(HERE, "..", "revised_manuscript", "main.bib")
CACHE = os.path.join(HERE, "results", "ref_cache.json")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

NEW_KEYS = ["petroni2019language", "wang2020language", "garcez2023neurosymbolic",
            "qudus2025fact", "ahmad2026mitigating", "ahmad2026harnessing",
            "ahmad2025enhancing"]

HEADERS = {"User-Agent": "GOFAIGenAI-ref-validator/1.0 (mailto:ecgarrido@comillas.edu)"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[{}\\]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_bib(path):
    text = open(path, encoding="utf-8").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        key = m.group(2).strip()
        if key not in NEW_KEYS:
            continue
        start = m.end()
        depth = 1
        i = m.start(0) + text[m.start(0):].index("{")
        # find matching brace from the entry opening brace
        j = i + 1
        depth = 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        body = text[i + 1:j - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.S):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries[key] = (m.group(1).lower(), fields)
    return entries


def crossref(doi, cache):
    if doi in cache:
        return cache[doi]
    for attempt in range(4):
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=HEADERS, timeout=20)
            if r.status_code == 200:
                cache[doi] = r.json()["message"]
                json.dump(cache, open(CACHE, "w"))
                return cache[doi]
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def arxiv(aid, cache):
    key = "arxiv:" + aid
    if key in cache:
        return cache[key]
    try:
        r = requests.get(f"http://export.arxiv.org/api/query?id_list={aid}", headers=HEADERS, timeout=20)
        if r.status_code == 200:
            cache[key] = r.text
            json.dump(cache, open(CACHE, "w"))
            return r.text
    except Exception:
        return None
    return None


def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    entries = parse_bib(BIB)
    failures = 0
    print(f"{'KEY':26s} {'TITLE':6s} {'AUTHOR':6s} {'YEAR':5s} {'VENUE':6s}  SOURCE")
    print("-" * 78)
    for key in NEW_KEYS:
        if key not in entries:
            print(f"{key:26s}  MISSING FROM BIB")
            failures += 1
            continue
        etype, f = entries[key]
        title = f.get("title", "")
        author = f.get("author", "")
        first_family = norm(author.split(" and ")[0].split(",")[0]) if author else ""
        year = f.get("year", "")
        venue = f.get("journal", "") or f.get("booktitle", "")
        doi = f.get("doi", "")

        if doi:
            msg = crossref(doi, cache)
            time.sleep(0.5)
            if not msg:
                print(f"{key:26s}  DOI did not resolve: {doi}")
                failures += 1
                continue
            ct = norm(" ".join(msg.get("title", [""])))
            ca = ""
            if msg.get("author"):
                ca = norm(msg["author"][0].get("family", ""))
            cy = str((msg.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0])
            cv = norm(" ".join(msg.get("container-title", [""])))
            t_ok = norm(title)[:40] in ct or ct[:40] in norm(title)
            a_ok = (first_family in ca) or (ca in first_family) if ca else False
            y_ok = (year == cy)
            v_ok = (norm(venue)[:12] in cv) or (cv[:12] in norm(venue)) or not cv
            ok = t_ok and a_ok and y_ok and v_ok
            failures += 0 if ok else 1
            print(f"{key:26s} {'OK' if t_ok else 'BAD':6s} {'OK' if a_ok else 'BAD':6s} "
                  f"{'OK' if y_ok else 'BAD':5s} {'OK' if v_ok else 'BAD':6s}  crossref:{doi}"
                  + ("" if ok else f"   <-- got title='{ct[:40]}' author='{ca}' year='{cy}'"))
        elif "arxiv" in (f.get("journal", "").lower()):
            aid = re.search(r"(\d{4}\.\d{4,5})", f.get("journal", ""))
            txt = arxiv(aid.group(1), cache) if aid else None
            t_ok = bool(txt) and norm(title)[:40] in norm(txt)
            failures += 0 if t_ok else 1
            print(f"{key:26s} {'OK' if t_ok else 'BAD':6s} {'--':6s} {'--':5s} {'--':6s}  arxiv:{aid.group(1) if aid else '?'}")
        else:
            print(f"{key:26s} {'?':6s} {'?':6s} {'?':5s} {'?':6s}  UNVERIFIED (no DOI/arXiv id) -- flagged for author confirmation")

    print("-" * 78)
    print(f"{'CLEAN' if failures == 0 else str(failures) + ' ISSUE(S)'} (UNVERIFIED entries are flagged, not failed)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
