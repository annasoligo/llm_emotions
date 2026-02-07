"""
Provenance and result-writing utilities for experiment reproducibility.

Provides:
    - get_provenance(): captures git commit, script path, timestamp
    - ResultWriter: writes per-factor JSONL files with metadata headers

Usage:
    from steering_tests.steering_utils.provenance import get_provenance, ResultWriter

    # Simple: just get a metadata dict
    meta = get_provenance(script=__file__, extra={"model": "gemma-27b"})

    # Full: per-factor result files with automatic metadata headers
    writer = ResultWriter(
        base_dir=Path("results/portfolio/gemma27b/run_20260207"),
        script=__file__,
        extra_meta={"model": "google/gemma-3-27b-it", "layer": 30},
    )
    writer.write("baseline", {"sample_id": 0, "response": "..."})
    writer.write("fear_pos_50pct", {"sample_id": 0, "response": "..."})
    writer.close()
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Cache git info per-process (doesn't change during a run)
_git_cache: Dict[str, Any] = {}

REPO_ROOT = Path(__file__).parent.parent.parent  # research-tools/


def get_provenance(
    script: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a provenance metadata dict for experiment reproducibility.

    Args:
        script: Path to the script that generated this data (use __file__).
                Will be stored relative to repo root if possible.
        extra: Additional key-value pairs to include in the metadata.

    Returns:
        Dict with git_commit, git_dirty, script, timestamp, and any extras.
    """
    global _git_cache

    if not _git_cache:
        try:
            _git_cache["commit"] = subprocess.check_output(
                ["git", "rev-parse", "--short=10", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            _git_cache["commit"] = None

        try:
            _git_cache["dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=REPO_ROOT,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            _git_cache["dirty"] = None

    meta: Dict[str, Any] = {
        "git_commit": _git_cache["commit"],
        "git_dirty": _git_cache["dirty"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if script:
        try:
            meta["script"] = str(Path(script).resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            meta["script"] = str(script)

    if extra:
        meta.update(extra)

    return meta


class ResultWriter:
    """
    Write per-factor JSONL result files with provenance metadata headers.

    Each factor gets its own file in base_dir. The first line of each file
    is a {"meta": {...}} JSON object with full provenance. Subsequent lines
    are individual result records.

    Files are flushed after each write for crash safety.
    """

    def __init__(
        self,
        base_dir: Path,
        script: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            base_dir: Directory where per-factor JSONL files will be created.
            script: Script path for provenance (use __file__).
            extra_meta: Additional metadata to include in the header.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._provenance = get_provenance(script=script, extra=extra_meta)
        self._handles: Dict[str, Any] = {}  # factor -> open file handle
        self._counts: Dict[str, int] = {}

    def write(self, factor: str, result: dict) -> Path:
        """
        Write a result line to the factor's JSONL file.

        On first call for a given factor, writes the metadata header.

        Args:
            factor: Factor name (becomes the filename, e.g. "fear_pos_50pct").
            result: Dict to write as a JSON line.

        Returns:
            Path to the factor's JSONL file.
        """
        filepath = self.base_dir / f"{factor}.jsonl"

        if factor not in self._handles:
            f = open(filepath, "w")
            f.write(json.dumps({"meta": self._provenance}) + "\n")
            self._handles[factor] = f
            self._counts[factor] = 0

        f = self._handles[factor]
        f.write(json.dumps(result) + "\n")
        f.flush()
        self._counts[factor] += 1

        return filepath

    def close(self):
        """Close all open file handles."""
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    @property
    def output_dir(self) -> Path:
        return self.base_dir

    @property
    def files(self) -> Dict[str, Path]:
        """Map of factor names to file paths."""
        return {
            factor: self.base_dir / f"{factor}.jsonl"
            for factor in self._counts
        }

    @property
    def counts(self) -> Dict[str, int]:
        """Number of result lines written per factor (excluding meta header)."""
        return dict(self._counts)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        total = sum(self._counts.values())
        return f"ResultWriter({self.base_dir}, {len(self._counts)} factors, {total} results)"


def load_results(filepath: Path) -> list:
    """
    Load result lines from a JSONL file, skipping the meta header.

    Returns:
        List of result dicts (meta line excluded).
    """
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "meta" in record:
                continue
            results.append(record)
    return results


def load_meta(filepath: Path) -> Optional[dict]:
    """
    Load the meta header from a JSONL result file.

    Returns:
        The meta dict, or None if no meta header found.
    """
    with open(filepath) as f:
        first_line = f.readline().strip()
        if first_line:
            record = json.loads(first_line)
            if "meta" in record:
                return record["meta"]
    return None


def sanitize_factor_name(name: str) -> str:
    """
    Sanitize a condition/factor name for use as a filename.

    Replaces problematic characters while keeping the name readable.
    """
    # Replace common problematic chars
    name = name.replace("/", "_")
    name = name.replace(" ", "_")
    name = name.replace("%", "pct")
    name = name.replace("+", "pos")
    name = name.replace("-", "neg")
    # Remove any remaining special chars
    return "".join(c for c in name if c.isalnum() or c in "_.")
