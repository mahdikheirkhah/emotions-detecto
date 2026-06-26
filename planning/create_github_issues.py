#!/usr/bin/env python3
"""Create GitHub issues from the Markdown files in planning/issues/.

Each issue file starts with a small YAML-style front-matter block:

    ---
    title: "[Phase 0] Project scaffolding & repository structure"
    labels: ["phase-0-foundations", "setup"]
    ---
    ## 1. Description
    ...

This script parses that front-matter, strips it, and POSTs the remaining body to the
GitHub Issues API, in filename order. It uses only the Python standard library.

Usage:
    export GITHUB_TOKEN=ghp_...                      # needs issues:write on the repo
    export GITHUB_REPO=mahdikheirkhah/emotions-detecto
    python planning/create_github_issues.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ISSUES_DIR = Path(__file__).parent / "issues"


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a Markdown file into (front_matter_dict, body).

    The front matter is the block between the first two '---' lines. Only the
    keys 'title' (a quoted string) and 'labels' (a JSON list) are understood.
    Returns the parsed metadata and the body with the front matter removed.
    """
    if not text.startswith("---"):
        raise ValueError("File is missing the leading '---' front matter.")
    _, fm, body = text.split("---", 2)
    meta: dict = {}
    for line in fm.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "labels":
            meta["labels"] = json.loads(value)
        elif key == "title":
            meta["title"] = value.strip('"')
    return meta, body.strip()


def create_issue(repo: str, token: str, title: str, body: str, labels: list[str]) -> int:
    """POST a single issue to the GitHub API and return its number."""
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({"title": title, "body": body, "labels": labels}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "emotions-detecto-issue-creator")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["number"]


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    repo = os.environ.get("GITHUB_REPO", "mahdikheirkhah/emotions-detecto")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not dry_run and not token:
        sys.exit("Set GITHUB_TOKEN (issues:write) or pass --dry-run.")

    files = sorted(
        p
        for p in ISSUES_DIR.glob("*.md")
        if p.name[0].isdigit() and "INDEX" not in p.name.upper()
    )
    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta["title"]
        labels = meta.get("labels", [])
        if dry_run:
            print(f"[dry-run] {path.name}: {title}  labels={labels}")
            continue
        try:
            number = create_issue(repo, token, title, body, labels)
            print(f"created #{number}: {title}")
        except urllib.error.HTTPError as exc:
            print(f"FAILED {path.name}: {exc.code} {exc.read().decode()}", file=sys.stderr)


if __name__ == "__main__":
    main()
