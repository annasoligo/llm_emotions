"""Crash-safe JSONL writer with file locking and atomic operations."""

import json
import fcntl
import os
from pathlib import Path
from typing import Dict, Any, Optional, Iterator
from contextlib import contextmanager


class JSONLWriter:
    """Append-only JSONL writer with crash safety.

    Features:
    - File locking to prevent concurrent writes
    - Immediate flush after each write
    - fsync for durability
    - Support for reading back written data

    Usage:
        writer = JSONLWriter(Path("output.jsonl"))
        writer.write({"id": "1", "data": "..."})
        writer.write({"id": "2", "data": "..."})

        # Read back
        for item in writer.read_all():
            print(item)
    """

    def __init__(self, path: Path, create_parents: bool = True):
        """Initialize JSONL writer.

        Args:
            path: Path to JSONL file
            create_parents: Create parent directories if needed
        """
        self.path = Path(path)

        if create_parents:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked_file(self, mode: str):
        """Context manager for file with exclusive lock.

        Args:
            mode: File open mode ('a', 'r', etc.)

        Yields:
            File handle with exclusive lock
        """
        f = open(self.path, mode)
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

    def write(self, item: Dict[str, Any]) -> None:
        """Write a single item to JSONL file.

        Args:
            item: Dict to write as JSON line

        Raises:
            TypeError: If item is not JSON-serializable
            OSError: If write fails
        """
        line = json.dumps(item, default=str) + '\n'

        with self._locked_file('a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def write_batch(self, items: list[Dict[str, Any]]) -> None:
        """Write multiple items atomically.

        All items are written in a single locked operation.

        Args:
            items: List of dicts to write

        Raises:
            TypeError: If any item is not JSON-serializable
            OSError: If write fails
        """
        lines = [json.dumps(item, default=str) + '\n' for item in items]

        with self._locked_file('a') as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> Iterator[Dict[str, Any]]:
        """Read all items from JSONL file.

        Yields:
            Each item as a dict

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not self.path.exists():
            return

        with self._locked_file('r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")

    def read_as_list(self) -> list[Dict[str, Any]]:
        """Read all items as a list.

        Returns:
            List of all items
        """
        return list(self.read_all())

    def count(self) -> int:
        """Count items in file.

        Returns:
            Number of items (non-empty lines)
        """
        if not self.path.exists():
            return 0

        count = 0
        with open(self.path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_ids(self) -> set[str]:
        """Get set of all IDs in file.

        Assumes each item has an 'id' field.

        Returns:
            Set of ID strings
        """
        ids = set()
        for item in self.read_all():
            if 'id' in item:
                ids.add(item['id'])
        return ids

    def exists(self) -> bool:
        """Check if file exists."""
        return self.path.exists()

    def clear(self) -> None:
        """Clear file contents (delete and recreate empty)."""
        if self.path.exists():
            self.path.unlink()

    def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID.

        Args:
            item_id: ID to search for

        Returns:
            Item dict if found, None otherwise
        """
        for item in self.read_all():
            if item.get('id') == item_id:
                return item
        return None

    def filter(self, predicate) -> Iterator[Dict[str, Any]]:
        """Filter items using a predicate function.

        Args:
            predicate: Function that takes item and returns bool

        Yields:
            Items where predicate returns True
        """
        for item in self.read_all():
            if predicate(item):
                yield item


class JSONLReader:
    """Read-only JSONL reader for loading existing data."""

    def __init__(self, path: Path):
        """Initialize JSONL reader.

        Args:
            path: Path to JSONL file
        """
        self.path = Path(path)

    def read_all(self) -> Iterator[Dict[str, Any]]:
        """Read all items from file.

        Yields:
            Each item as dict
        """
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")

        with open(self.path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")

    def read_as_list(self) -> list[Dict[str, Any]]:
        """Read all items as list."""
        return list(self.read_all())
