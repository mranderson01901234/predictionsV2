"""
Model version configuration.

This module provides the current model version and git commit SHA for tracking.
"""

import subprocess
import os
from pathlib import Path

# Model version (increment when model architecture or training changes)
MODEL_VERSION = "v1.0.0"


def get_git_sha(project_root: Path = None) -> str:
    """
    Get the current git commit SHA.
    
    Args:
        project_root: Path to project root (defaults to config parent)
    
    Returns:
        Git SHA string, or "unknown" if not available
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    # Try to get SHA from git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repo
        pass
    
    # Try environment variable (useful in CI)
    git_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
    if git_sha:
        return git_sha
    
    return "unknown"


def get_version_info(project_root: Path = None) -> dict:
    """
    Get version information including model version and git SHA.
    
    Args:
        project_root: Path to project root
    
    Returns:
        Dictionary with 'model_version' and 'git_sha' keys
    """
    return {
        "model_version": MODEL_VERSION,
        "git_sha": get_git_sha(project_root),
    }

