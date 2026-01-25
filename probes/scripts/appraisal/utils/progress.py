"""Progress tracking and resume support for pipeline execution."""

import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set
import fcntl
import os


class StateManager:
    """Manages pipeline state for crash recovery and resume.

    Tracks:
    - Stage cursors (which items have been processed)
    - Completed IDs per stage
    - Run manifest (config hash, git commit, timestamps)

    Uses atomic writes for crash safety.
    """

    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """Initialize state manager.

        Args:
            output_dir: Directory for state files
            config: Full pipeline config dict (for hashing)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.output_dir / "state.json"
        self.manifest_path = self.output_dir / "run_manifest.json"

        self.config = config
        self.config_hash = self._compute_config_hash(config)

        # Load or initialize state
        self.state = self._load_or_init_state()

        # Initialize or update manifest
        self._init_manifest()

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 hash of config for change detection."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if in repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.output_dir,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _load_or_init_state(self) -> Dict[str, Any]:
        """Load existing state or initialize new state."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                state = json.load(f)

            # Check config hash
            if state.get("config_hash") != self.config_hash:
                print(f"Warning: Config has changed since last run")
                print(f"  Previous: {state.get('config_hash')}")
                print(f"  Current:  {self.config_hash}")

            return state

        # Initialize new state
        return {
            "config_hash": self.config_hash,
            "stage_cursors": {},
            "completed_ids": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _init_manifest(self) -> None:
        """Initialize or update run manifest."""
        manifest = {
            "config_hash": self.config_hash,
            "git_commit": self._get_git_commit(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "python_version": subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True
            ).stdout.strip(),
        }

        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                existing = json.load(f)
                manifest["created_at"] = existing.get("created_at", manifest["created_at"])

        self._atomic_write(self.manifest_path, manifest)

    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON atomically using temp file + rename."""
        tmp_path = path.with_suffix('.tmp')

        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.rename(path)

    def save_state(self) -> None:
        """Save current state to disk atomically."""
        self.state["updated_at"] = datetime.now().isoformat()
        self._atomic_write(self.state_path, self.state)

    def get_stage_cursor(self, stage_name: str) -> int:
        """Get cursor position for a stage (0 = not started)."""
        return self.state["stage_cursors"].get(stage_name, 0)

    def set_stage_cursor(self, stage_name: str, cursor: int) -> None:
        """Set cursor position for a stage and save."""
        self.state["stage_cursors"][stage_name] = cursor
        self.save_state()

    def get_completed_ids(self, stage_name: str) -> Set[str]:
        """Get set of completed item IDs for a stage."""
        ids = self.state["completed_ids"].get(stage_name, [])
        return set(ids)

    def mark_completed(self, stage_name: str, item_id: str) -> None:
        """Mark an item as completed for a stage."""
        if stage_name not in self.state["completed_ids"]:
            self.state["completed_ids"][stage_name] = []

        if item_id not in self.state["completed_ids"][stage_name]:
            self.state["completed_ids"][stage_name].append(item_id)
            self.save_state()

    def is_completed(self, stage_name: str, item_id: str) -> bool:
        """Check if an item is completed for a stage."""
        return item_id in self.get_completed_ids(stage_name)

    def reset_stage(self, stage_name: str) -> None:
        """Reset progress for a specific stage."""
        self.state["stage_cursors"].pop(stage_name, None)
        self.state["completed_ids"].pop(stage_name, None)
        self.save_state()

    def reset_all(self) -> None:
        """Reset all progress."""
        self.state["stage_cursors"] = {}
        self.state["completed_ids"] = {}
        self.save_state()

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of progress across all stages."""
        return {
            "config_hash": self.config_hash,
            "stages": {
                name: {
                    "cursor": self.state["stage_cursors"].get(name, 0),
                    "completed_count": len(self.state["completed_ids"].get(name, [])),
                }
                for name in set(
                    list(self.state["stage_cursors"].keys()) +
                    list(self.state["completed_ids"].keys())
                )
            },
            "created_at": self.state.get("created_at"),
            "updated_at": self.state.get("updated_at"),
        }

    def print_progress(self) -> None:
        """Print progress summary to console."""
        summary = self.get_progress_summary()

        print("\n" + "=" * 60)
        print("PIPELINE PROGRESS")
        print("=" * 60)
        print(f"Config hash: {summary['config_hash']}")
        print(f"Created: {summary['created_at']}")
        print(f"Updated: {summary['updated_at']}")
        print()

        if summary["stages"]:
            print("Stage Progress:")
            for name, info in summary["stages"].items():
                print(f"  {name}:")
                print(f"    Cursor: {info['cursor']}")
                print(f"    Completed: {info['completed_count']}")
        else:
            print("No stages started yet")

        print("=" * 60 + "\n")
