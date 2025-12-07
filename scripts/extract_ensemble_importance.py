"""
Extract feature importance from stacked ensemble meta-model.
"""

import sys
from pathlib import Path
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures.stacking_ensemble import StackingEnsemble

def extract_meta_model_importance(ensemble_path: Path, output_path: Path):
    """Extract logistic regression coefficients from meta-model."""
    print(f"Loading ensemble from {ensemble_path}")
    ensemble = StackingEnsemble.load(ensemble_path)
    
    importance = {}
    
    # Get base model names
    base_model_names = list(ensemble.base_models.keys())
    print(f"Base models: {base_model_names}")
    
    # Extract meta-model coefficients if it's logistic regression
    if ensemble.meta_model_type == 'logistic' and ensemble.meta_model is not None:
        if hasattr(ensemble.meta_model, 'coef_'):
            coef = ensemble.meta_model.coef_[0]
            
            # Map coefficients to base model names
            for i, model_name in enumerate(base_model_names):
                if i < len(coef):
                    importance[model_name] = float(coef[i])
            
            print("\nMeta-Model Feature Importance (Logistic Regression Coefficients):")
            print("=" * 60)
            sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for model_name, coef_value in sorted_importance:
                print(f"  {model_name:20s}: {coef_value:+.6f}")
        else:
            print("Meta-model doesn't have coef_ attribute")
    else:
        print(f"Meta-model type: {ensemble.meta_model_type}")
        if ensemble.meta_model is None:
            print("Meta-model not fitted yet")
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(importance, f, indent=2)
    
    print(f"\nSaved feature importance to {output_path}")
    return importance

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    ensemble_path = project_root / "artifacts" / "models" / "nfl_stacked_ensemble" / "ensemble_v1.pkl"
    output_path = project_root / "artifacts" / "models" / "nfl_stacked_ensemble" / "feature_importance.json"
    
    extract_meta_model_importance(ensemble_path, output_path)

