#!/usr/bin/env python3
"""Check bibliography entries: DOI metadata against Crossref, URL availability.

For every entry with a DOI, the Crossref record is fetched and its title and year
are compared with the entry. Entries without a DOI are checked by requesting the URL.
Entries with neither are reported as unverifiable.

Usage:
    python tools/check_refs.py references.bib
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "thesis-reference-check/0.1 (mailto:mykyta.kyt@nure.ua)"
TITLE_SIMILARITY = 0.85


def parse_bib(text: str) -> list[dict[str, str]]:
    """Minimal BibTeX parser: @type{key, field = {value} | "value" | number, ...}."""
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start, depth, i = m.end(), 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[start:i - 1]
        entry = {"ENTRYTYPE": m.group(1).lower(), "ID": m.group(2)}
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            j = f.end()
            if j < len(body) and body[j] == "{":
                depth, k = 1, j + 1
                while k < len(body) and depth:
                    depth += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                value = body[j + 1:k - 1]
            elif j < len(body) and body[j] == '"':
                k = body.index('"', j + 1)
                value = body[j + 1:k]
            else:
                value = re.match(r"[^,\s]+", body[j:]).group(0)
            entry[f.group(1).lower()] = " ".join(value.split())
        entries.append(entry)
    return entries


def normalise(title: str) -> str:
    return re.sub(r"[^\w ]", "", title.replace("{", "").replace("}", "")).lower().strip()


def fetch(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def check_doi(entry: dict[str, str]) -> tuple[bool, str]:
    doi = entry["doi"].removeprefix("https://doi.org/")
    status, body = fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    if status != 200:
        return False, f"DOI not found in Crossref (HTTP {status})"
    msg = json.loads(body)["message"]
    problems = []
    remote_title = (msg.get("title") or [""])[0]
    ratio = difflib.SequenceMatcher(None, normalise(entry.get("title", "")), normalise(remote_title)).ratio()
    if ratio < TITLE_SIMILARITY:
        problems.append(f"title mismatch ({ratio:.2f}): Crossref has '{remote_title}'")
    parts = (msg.get("published") or msg.get("issued") or {}).get("date-parts", [[None]])
    remote_year = parts[0][0]
    if entry.get("year") and remote_year and int(entry["year"]) != int(remote_year):
        problems.append(f"year {entry['year']} != Crossref {remote_year}")
    return not problems, "; ".join(problems) or "ok"


def check_url(entry: dict[str, str]) -> tuple[bool, str]:
    status, _ = fetch(entry["url"])
    return status < 400, f"HTTP {status}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bib", nargs="?", default="references.bib")
    args = parser.parse_args()

    with open(args.bib, encoding="utf-8") as f:
        entries = parse_bib(f.read())

    failed = 0
    for e in entries:
        try:
            if "doi" in e:
                ok, note = check_doi(e)
            elif "url" in e:
                ok, note = check_url(e)
            else:
                ok, note = False, "no DOI or URL"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            ok, note = False, f"network error: {exc}"
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {e['ID']}: {note}")

    print(f"\n{len(entries) - failed}/{len(entries)} entries passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
