from __future__ import annotations

import argparse
from pathlib import Path

from model.inference import SignPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for a single BISINDO image.")
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    predictor = SignPredictor()
    if not predictor.model_loaded:
        raise FileNotFoundError("Model belum tersedia. Jalankan training terlebih dahulu.")

    image_bytes = args.image_path.read_bytes()
    result = predictor.predict_bytes(image_bytes)
    print(
        {
            "success": result.success,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence, 4),
            "top_predictions": result.top_predictions,
            "message": result.message,
        }
    )


if __name__ == "__main__":
    main()
