import joblib
import pickle
import os
from pathlib import Path

# Configuration
MODEL_DIR = "model"
COMPRESSION_LEVEL = 3  # Range: 1 (fast) to 9 (smaller files)

def compress_models():
    """Compress all .pkl models to smaller .joblib files"""
    model_names = [
        'logistic_regression', 
        'random_forest', 
        'svm', 
        'xgboost'
    ]

    for name in model_names:
        input_path = Path(MODEL_DIR) / f"{name}_model.pkl"
        output_path = Path(MODEL_DIR) / f"{name}_model_compressed.joblib"

        # Skip if already compressed
        if output_path.exists():
            print(f"Skipping {name} (already compressed)")
            continue

        try:
            # Load original pickle
            with open(input_path, 'rb') as f:
                vectorizer, model = pickle.load(f)

            # Save with compression
            joblib.dump(
                (vectorizer, model),
                output_path,
                compress=COMPRESSION_LEVEL
            )
            
            print(f"Compressed {name}:")
            print(f"  Original: {os.path.getsize(input_path) / 1024:.1f} KB")
            print(f"  Compressed: {os.path.getsize(output_path) / 1024:.1f} KB")

        except Exception as e:
            print(f"Error compressing {name}: {str(e)}")

if __name__ == "__main__":
    compress_models()