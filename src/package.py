"""Allowlisted anonymous source export; does not rewrite Git history."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import nbformat

from src.config import PROJECT_ROOT

IDENTITY_PATTERNS = (
    r"(?:/home/|/Users/)[\w.-]+/",  # Real absolute user paths, not generic setup syntax
    r"[A-Za-z]:\\+Users\\+[^\\\s]+\\+",
    r"predict-internet-usage-[a-z]+-secret",  # Identifying legacy project naming pattern
    r"(?:github\.com|gitlab\.com)/[\w.-]+/[\w.-]+",  # Review source repository links manually
)


def anonymity_issues(text):
    return [pattern for pattern in IDENTITY_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]


def sanitized_notebook(notebook):
    """Remove outputs and runtime/personal metadata, retaining source and IDs."""
    notebook = nbformat.from_dict(json.loads(json.dumps(notebook)))
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for cell in notebook.cells:
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)
    return notebook


def sanitize_repository_notebooks(root=PROJECT_ROOT):
    """Mechanical cleanup of derived notebook outputs, never source code."""
    for path in sorted((Path(root) / "notebooks").glob("*.ipynb")):
        original = nbformat.read(path, as_version=4)
        cleaned = sanitized_notebook(original)
        text = nbformat.writes(cleaned)
        if anonymity_issues(text):
            raise ValueError(f"Identifying source text remains in notebooks/{path.name}; remove it explicitly")
        nbformat.write(cleaned, path)
    print("Removed cached notebook outputs and runtime metadata; source retained")


def export_files(root):
    """No recursive workspace export: data, environments, and .git cannot enter."""
    exact = ("README.md", "requirements.txt", "requirements.lock.txt", ".python-version", ".gitignore", ".gitattributes", "project.pdf", "docs/report_provenance.json")
    paths = [root / name for name in exact]
    for directory, suffixes in (("src", {".py"}), ("tests", {".py"}), ("notebooks", {".ipynb"}), ("docs", {".md"})):
        paths.extend(p for p in sorted((root / directory).glob("*")) if p.is_file() and p.suffix in suffixes)
    paths.extend(sorted((root / "results").glob("nested_*.csv")))
    paths = [p for p in paths if p.name != "nested_oof_predictions.csv"]
    paths.append(root / "results" / "nested_metadata.json")
    paths.extend(sorted((root / "results" / "figures").glob("nested_*.png")))
    return paths


def build_archive(root=PROJECT_ROOT):
    from src.verify import verify_artifacts
    from src.experiment import file_digest
    root = Path(root)
    metadata = verify_artifacts(root)
    if not (root / "project.pdf").exists() or not (root / "docs" / "project.rendered.md").exists():
        raise FileNotFoundError("Generate the technical report with python -m src.report first")
    provenance = json.loads((root / "docs" / "report_provenance.json").read_text())
    if provenance["experiment_artifact_sha256"] != metadata["artifact_sha256"] or not all(
        file_digest(root / name) == digest for name, digest in provenance["report_sha256"].items()
    ):
        raise ValueError("Report, plots, source template, or result provenance changed; regenerate with python -m src.report")
    # Verify the PDF's source as well as code and data-derived text.
    rendered = (root / "docs" / "project.rendered.md").read_text()
    if anonymity_issues(rendered):
        raise ValueError("The rendered report contains an identifying path or repository reference")
    payloads = {}
    for path in export_files(root):
        if not path.exists() or path.is_symlink():
            raise ValueError(f"Required export file missing or a symlink: {path.name}")
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".ipynb":
            text = nbformat.writes(sanitized_notebook(nbformat.read(path, as_version=4)))
            payload = text.encode()
        else:
            payload = path.read_bytes()
        if path.suffix not in {".pdf", ".png"} and anonymity_issues(payload.decode()):
            raise ValueError(f"Identifying text remains in {relative}")
        payloads[relative] = payload
    manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    archive = dist / "project.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zip_file:
        for name, payload in sorted(payloads.items()):
            zip_file.writestr(name, payload)
    with ZipFile(archive) as zip_file:
        names = zip_file.namelist()
        if len(names) != len(set(names)) or "project.pdf" not in names or "MANIFEST.json" not in names:
            raise ValueError("Invalid source archive structure")
        for name in names:
            if name.startswith(("data/", ".git/", ".venv/", ".repro-env/")) or "oof_predictions" in name:
                raise ValueError("Restricted data or local files entered the archive")
        for name, digest in manifest.items():
            if hashlib.sha256(zip_file.read(name)).hexdigest() != digest:
                raise ValueError("Archive manifest verification failed")
    print(f"Built dist/project.zip: {len(payloads)} allowlisted files; source/notebook anonymity patterns and manifest verified")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize-notebooks", action="store_true")
    args = parser.parse_args()
    if args.sanitize_notebooks:
        sanitize_repository_notebooks()
    else:
        build_archive()


if __name__ == "__main__":
    main()
