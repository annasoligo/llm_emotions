"""
Cleanup utilities for vLLM processes.

Ensures vLLM worker processes are killed when the main script exits,
preventing orphaned GPU processes.

Usage:
    from steering_tests.steering_utils.cleanup import register_cleanup

    # Call early in script, before creating LLM
    register_cleanup()

    llm = LLM(...)  # Now if script exits, cleanup will run

Note: For cleaning up orphans from PREVIOUS jobs, use the slurm-aware script:
    python /workspace-vast/annas/scripts/discover-gpu-processes.py --validate --kill --yes --user $USER
"""

import atexit
import os
import signal
import subprocess
import sys


def _get_descendant_pids(parent_pid: int) -> list[int]:
    """Get all descendant PIDs (children, grandchildren, etc.) of a process."""
    descendants = []
    try:
        # Get direct children
        result = subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            children = [int(p) for p in result.stdout.strip().split('\n') if p]
            for child in children:
                descendants.append(child)
                # Recursively get grandchildren
                descendants.extend(_get_descendant_pids(child))
    except Exception:
        pass
    return descendants


def _kill_vllm_children():
    """Kill any vLLM child processes spawned by this process."""
    try:
        my_pid = os.getpid()

        # Get ALL descendants (children, grandchildren, etc.)
        all_descendants = _get_descendant_pids(my_pid)

        # Kill any that are vLLM workers
        for pid in all_descendants:
            try:
                # Check if this is a vLLM process
                result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "comm="],
                    capture_output=True,
                    text=True
                )
                comm = result.stdout.strip()
                if "VLLM" in comm or "vllm" in comm or "ray" in comm.lower():
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError, OSError):
                pass

    except Exception as e:
        print(f"Cleanup warning: {e}", file=sys.stderr)


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up and exiting."""
    _kill_vllm_children()
    sys.exit(128 + signum)


def register_cleanup():
    """
    Register cleanup handlers for vLLM processes.

    Call this early in your script, before creating any LLM instances.
    It registers:
    - atexit handler for normal exits
    - SIGTERM handler for slurm cancellation
    - SIGINT handler for Ctrl+C

    This only cleans up processes spawned by THIS script.
    For cleaning orphans from previous jobs, use discover-gpu-processes.py.
    """
    atexit.register(_kill_vllm_children)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
