from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_all_scripts_parse():
    for path in (ROOT / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_public_text_contains_no_machine_specific_paths_or_process_markers():
    patterns = {
        "Windows absolute path": re.compile(r"\b[A-Za-z]:\\"),
        "internal version identifier": re.compile(r"\b[vV]0\d{2}\b"),
        "workflow-state marker": re.compile(
            r"\b(?:" + "dra" + r"ft|revi" + r"sed|lo" + r"cked)\b", re.IGNORECASE
        ),
        "authoring-tool marker": re.compile(r"\b" + "co" + r"dex\b", re.IGNORECASE),
    }
    suffixes = {".py", ".md", ".txt", ".cff", ".yml", ".yaml"}
    failures = []
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(ROOT)}")
    assert not failures, "\n".join(failures)


def test_repository_excludes_research_data_and_generated_outputs():
    disallowed_suffixes = {".tif", ".tiff", ".npz", ".zip"}
    found = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in disallowed_suffixes
    ]
    assert not found, f"Data or generated files are present in the code repository: {found}"
