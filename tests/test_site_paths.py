"""Fail on hard-coded site paths and on internal notes anywhere in the repository.

Every data/checkpoint/output location must go through hep4m/paths.py (Python)
or ${HEP4M_*} variables (YAML, shell).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"/global/|/pscratch|/u/kakatin|/ptmp/|/global/homes")
EXCLUDE_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".pixi", "outputs", "data", "work"}
TEXT_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".json", ".toml", ".md", ".tex", ".txt", ".tsv",
                 ".cfg", ".ini", ".example", ""}


def _files():
    for p in REPO.rglob("*"):
        rel = p.relative_to(REPO)
        if p.is_dir() or rel.parts[0] in EXCLUDE_DIRS or "__pycache__" in rel.parts \
                or rel.parts[0].endswith(".egg-info"):
            continue
        if p.suffix in TEXT_SUFFIXES or p.name.startswith(".env"):
            yield p


def test_no_site_paths():
    hits = []
    for p in _files():
        if p == Path(__file__):
            continue
        try:
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if PATTERN.search(line):
                    hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()[:120]}")
        except UnicodeDecodeError:
            continue
    assert not hits, "hard-coded site paths:\n" + "\n".join(hits[:50])


def test_scan_catches_a_planted_path(tmp_path):
    """The pattern must match the literal forms that occur in the dev repo."""
    for s in ("/global/cfs/cdirs/m4958/data/COCOA", "/pscratch/sd/d/x", "/u/kakatin/storage",
              "/ptmp/kakatin/hep4m"):
        assert PATTERN.search(f'x = "{s}/file.root"'), s


# user names, project ids and tool names that must not appear in code or configs
NOTES = re.compile(r"danieltm|kakatin|/nilotpal/|\bm4958\b|overleaf|sprint_T\d|HEP4M-recovery",
                   re.IGNORECASE)
CODE_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".ini"}


def test_no_internal_notes():
    hits = []
    for p in _files():
        if p == Path(__file__) or p.suffix not in CODE_SUFFIXES:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if NOTES.search(line):
                hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()[:120]}")
    assert not hits, "internal names or ids:\n" + "\n".join(hits[:50])
