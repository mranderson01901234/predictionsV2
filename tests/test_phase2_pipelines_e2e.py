"""
End-to-End Tests for Phase 2 and Phase 2B Pipelines

Tests that pipelines are properly wired and can be invoked.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))




class TestPhase2PipelineEntryPoints:
    """Test that pipeline entry points are properly wired."""
    
    def test_phase2_pipeline_module_runnable(self):
        """Test that phase2_pipeline can be imported and has run_phase2 function."""
        from orchestration.pipelines.phase2_pipeline import run_phase2
        assert callable(run_phase2)
    
    def test_phase2b_pipeline_module_runnable(self):
        """Test that phase2b_pipeline can be imported and has run_phase2b function."""
        from orchestration.pipelines.phase2b_pipeline import run_phase2b
        assert callable(run_phase2b)
    
    def test_phase2_registered_in_registry(self):
        """Test that phase2 is registered in feature_table_registry."""
        from features.feature_table_registry import list_feature_tables, get_feature_table_path
        
        tables = list_feature_tables()
        assert "phase2" in tables, "phase2 should be registered"
        
        path = get_feature_table_path("phase2")
        assert path.name == "game_features_phase2.parquet", "Should map to correct filename"
    
    def test_phase2b_registered_in_registry(self):
        """Test that phase2b is registered in feature_table_registry."""
        from features.feature_table_registry import list_feature_tables, get_feature_table_path
        
        tables = list_feature_tables()
        assert "phase2b" in tables, "phase2b should be registered"
        
        path = get_feature_table_path("phase2b")
        assert path.name == "game_features_phase2b.parquet", "Should map to correct filename"
    
    def test_phase2_pipeline_imports_dependencies(self):
        """Test that phase2_pipeline imports all required dependencies."""
        from orchestration.pipelines.phase2_pipeline import (
            run_phase2,
            validate_phase2_output,
        )
        assert callable(run_phase2)
        assert callable(validate_phase2_output)
    
    def test_phase2b_pipeline_imports_dependencies(self):
        """Test that phase2b_pipeline imports all required dependencies."""
        from orchestration.pipelines.phase2b_pipeline import (
            run_phase2b,
            validate_phase2b_output,
        )
        assert callable(run_phase2b)
        assert callable(validate_phase2b_output)

