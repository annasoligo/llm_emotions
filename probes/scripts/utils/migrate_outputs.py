#!/usr/bin/env python3
"""
Migrate existing results from old structure to new unified outputs/ structure.

This script:
1. Maps old result directories to new unified structure
2. Creates symlinks to maintain backward compatibility
3. Optionally copies/moves files to new locations
4. Generates migration report
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent to path to import output_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from output_config import LEGACY_TO_NEW_PATHS, REPO_ROOT, OutputPaths


def scan_existing_results() -> Dict[str, Path]:
    """
    Scan repository for existing result directories.

    Returns:
        Dict mapping old path → actual filesystem path
    """
    existing = {}

    # Check old results/ directories
    old_results_root = REPO_ROOT / "results"
    if old_results_root.exists():
        for item in old_results_root.iterdir():
            if item.is_dir():
                rel_path = f"results/{item.name}"
                existing[rel_path] = item

    # Check old probes/results/ directories
    old_probes_results = REPO_ROOT / "probes" / "results"
    if old_probes_results.exists():
        for item in old_probes_results.iterdir():
            if item.is_dir():
                rel_path = f"probes/results/{item.name}"
                existing[rel_path] = item

    return existing


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except PermissionError:
        pass
    return total


def count_files(path: Path) -> int:
    """Count total files in directory."""
    try:
        return sum(1 for _ in path.rglob("*") if _.is_file())
    except PermissionError:
        return 0


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def create_migration_report(
    existing: Dict[str, Path],
    migrated: List[Tuple[Path, Path]],
    symlinked: List[Tuple[Path, Path]],
    skipped: List[Tuple[str, str]]
) -> str:
    """
    Generate migration report.

    Args:
        existing: Existing old directories found
        migrated: List of (old_path, new_path) that were migrated
        symlinked: List of (old_path, new_path) that were symlinked
        skipped: List of (old_path, reason) that were skipped

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("OUTPUT DIRECTORY MIGRATION REPORT")
    report.append("=" * 80)
    report.append("")

    report.append(f"📊 Summary")
    report.append(f"  Total old directories found: {len(existing)}")
    report.append(f"  Successfully migrated: {len(migrated)}")
    report.append(f"  Symlinks created: {len(symlinked)}")
    report.append(f"  Skipped: {len(skipped)}")
    report.append("")

    if migrated:
        report.append("✅ Migrated Directories:")
        report.append("-" * 80)
        for old_path, new_path in migrated:
            size = get_directory_size(old_path)
            n_files = count_files(old_path)
            report.append(f"  {old_path}")
            report.append(f"    → {new_path}")
            report.append(f"    ({n_files} files, {format_size(size)})")
            report.append("")

    if symlinked:
        report.append("🔗 Symlinks Created:")
        report.append("-" * 80)
        for old_path, new_path in symlinked:
            report.append(f"  {old_path} → {new_path}")
        report.append("")

    if skipped:
        report.append("⚠️  Skipped Directories:")
        report.append("-" * 80)
        for old_path, reason in skipped:
            report.append(f"  {old_path}")
            report.append(f"    Reason: {reason}")
        report.append("")

    report.append("=" * 80)
    report.append("Migration complete!")
    report.append("")
    report.append("Next steps:")
    report.append("  1. Test that scripts work with new paths")
    report.append("  2. Update any hardcoded paths in code")
    report.append("  3. Once verified, remove old directories")
    report.append("  4. Update .gitignore to exclude outputs/")
    report.append("=" * 80)

    return "\n".join(report)


def migrate_outputs(
    mode: str = "symlink",
    dry_run: bool = False,
    verbose: bool = False
) -> Tuple[List, List, List]:
    """
    Migrate outputs to new unified structure.

    Args:
        mode: "symlink" (create symlinks), "copy" (copy files), "move" (move files)
        dry_run: If True, only show what would be done
        verbose: Print detailed progress

    Returns:
        Tuple of (migrated, symlinked, skipped) lists
    """
    print("Scanning for existing result directories...")
    existing = scan_existing_results()
    print(f"Found {len(existing)} existing result directories\n")

    migrated = []
    symlinked = []
    skipped = []

    for old_rel_path, old_abs_path in existing.items():
        if verbose or dry_run:
            print(f"Processing: {old_rel_path}")

        # Get new path from mapping
        if old_rel_path not in LEGACY_TO_NEW_PATHS:
            skipped.append((old_rel_path, "No mapping defined"))
            if verbose:
                print(f"  ⚠️  Skipped: No mapping defined")
            continue

        new_path = LEGACY_TO_NEW_PATHS[old_rel_path]

        # Ensure new path is absolute
        if not new_path.is_absolute():
            new_path = REPO_ROOT / new_path

        if verbose or dry_run:
            print(f"  → New path: {new_path.relative_to(REPO_ROOT)}")

        if dry_run:
            print(f"  [DRY RUN] Would {mode}: {old_abs_path} → {new_path}")
            migrated.append((old_abs_path, new_path))
            continue

        # Create parent directory for new path
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if mode == "symlink":
                # Create symlink from new to old (so new path points to old data)
                if not new_path.exists():
                    new_path.symlink_to(old_abs_path, target_is_directory=True)
                    symlinked.append((new_path, old_abs_path))
                    if verbose:
                        print(f"  ✓ Created symlink")
                else:
                    if verbose:
                        print(f"  ⚠️  Target already exists, skipping")
                    skipped.append((old_rel_path, "Target already exists"))

            elif mode == "copy":
                # Copy old to new
                if not new_path.exists():
                    shutil.copytree(old_abs_path, new_path, symlinks=True)
                    migrated.append((old_abs_path, new_path))
                    if verbose:
                        print(f"  ✓ Copied to new location")
                else:
                    if verbose:
                        print(f"  ⚠️  Target already exists, skipping")
                    skipped.append((old_rel_path, "Target already exists"))

            elif mode == "move":
                # Move old to new
                if not new_path.exists():
                    shutil.move(str(old_abs_path), str(new_path))
                    migrated.append((old_abs_path, new_path))
                    if verbose:
                        print(f"  ✓ Moved to new location")
                else:
                    if verbose:
                        print(f"  ⚠️  Target already exists, skipping")
                    skipped.append((old_rel_path, "Target already exists"))

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            skipped.append((old_rel_path, f"Error: {str(e)}"))
            if verbose:
                print(f"  ❌ Error: {e}")

        if verbose:
            print("")

    return migrated, symlinked, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Migrate results to unified outputs/ structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be migrated (safe)
  python migrate_outputs.py --dry-run

  # Create symlinks (recommended - preserves old paths)
  python migrate_outputs.py --mode symlink

  # Copy files to new structure (keeps old files)
  python migrate_outputs.py --mode copy

  # Move files to new structure (removes old files)
  python migrate_outputs.py --mode move

  # Generate report only
  python migrate_outputs.py --report-only
        """
    )

    parser.add_argument(
        "--mode",
        choices=["symlink", "copy", "move"],
        default="symlink",
        help="Migration mode (default: symlink)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report of existing directories without migrating"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save report to file"
    )

    args = parser.parse_args()

    if args.report_only:
        # Just scan and report
        existing = scan_existing_results()
        print(f"\n📊 Found {len(existing)} existing result directories:\n")
        for old_path, abs_path in sorted(existing.items()):
            size = get_directory_size(abs_path)
            n_files = count_files(abs_path)
            new_path = LEGACY_TO_NEW_PATHS.get(old_path, "❓ No mapping")
            print(f"  {old_path}")
            print(f"    Files: {n_files}, Size: {format_size(size)}")
            if isinstance(new_path, Path):
                print(f"    → {new_path.relative_to(REPO_ROOT)}")
            else:
                print(f"    → {new_path}")
            print("")
        return

    # Perform migration
    migrated, symlinked, skipped = migrate_outputs(
        mode=args.mode,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # Generate report
    existing = scan_existing_results()
    report = create_migration_report(existing, migrated, symlinked, skipped)
    print("\n" + report)

    # Save report if requested
    if args.output:
        args.output.write_text(report)
        print(f"\n📄 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
