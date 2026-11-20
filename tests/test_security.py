"""Sécurité opérationnelle du dépôt (T28-T29) : secrets, données, fichiers volumineux."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = {
    "clé AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "jeton GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "clé privée": re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "clé Kaggle": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "mot de passe en dur": re.compile(
        r"(?i)(password|passwd|secret|api_key)\s*=\s*['\"][^'\"]{6,}"
    ),
}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".txt", ".json", ".toml", ".cfg", ".ini", ".env"}
MAX_FILE_MB = 5


def repo_files() -> list[Path]:
    """Fichiers suivis par git si disponible, sinon arborescence hors .gitignore usuels."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        files = [ROOT / f for f in out.splitlines() if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = [
            p
            for p in ROOT.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
        ]
    return [f for f in files if f.is_file()]


def test_no_secrets_in_repository():
    findings = []
    for f in repo_files():
        if f.suffix not in TEXT_SUFFIXES or f.name == "test_security.py":
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{f.relative_to(ROOT)} : {label}")
    assert not findings, "Secrets potentiels :\n" + "\n".join(findings)


def test_secret_patterns_detect_examples():
    """Le détecteur lui-même est testé (sinon un test vert ne prouve rien)."""
    assert SECRET_PATTERNS["clé AWS"].search("AKIA" + "ABCDEFGHIJKLMNOP")
    assert SECRET_PATTERNS["jeton GitHub"].search("ghp_" + "a" * 36)
    assert SECRET_PATTERNS["mot de passe en dur"].search("password = 'hunter22'")


def test_no_dataset_or_large_file_committed():
    offenders = [
        f"{f.relative_to(ROOT)} ({f.stat().st_size / 1e6:.1f} Mo)"
        for f in repo_files()
        if f.stat().st_size > MAX_FILE_MB * 1e6 or f.name == "creditcard.csv"
    ]
    assert not offenders, f"Fichiers à ne pas versionner : {offenders}"


@pytest.mark.parametrize("pattern", ["data/*.csv", "*.joblib", ".env", "kaggle.json", "models/"])
def test_gitignore_protects_sensitive_paths(pattern):
    gitignore = (ROOT / ".gitignore").read_text()
    assert pattern in gitignore.splitlines()
