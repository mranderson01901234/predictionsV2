"""
Advanced Model Training Module (Backward Compatibility Wrapper)

This module provides backward compatibility for scripts that reference train_advanced.py.
It redirects to the unified trainer.py module.

Usage:
    python -m models.training.train_advanced --model ensemble
    python -m models.training.train_advanced --model ft_transformer
    python -m models.training.train_advanced --model tabnet
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import and run the unified trainer's main function
from models.training.trainer import main

if __name__ == "__main__":
    main()



