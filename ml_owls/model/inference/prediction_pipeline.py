from pathlib import Path
from typing import Any

import numpy as np

from interfaces.model.audio_processor import AudioProcessor
from interfaces.model.predictor import Predictor


class BirdCLEFPredictionPipeline:
    """Complete inference pipeline for BirdCLEF models."""

    def __init__(
        self,
        predictor: Predictor,
        audio_processor: AudioProcessor,
        species_names: list[str],
        top_k: int = 5,
        confidence_threshold: float = 0.1,
    ) -> None:
        """Initialize prediction pipeline."""
        self.predictor = predictor
        self.audio_processor = audio_processor
        self.species_names = species_names
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        print("🔮 Prediction Pipeline initialized")
        print(f"   Species: {len(species_names)}")
        print(f"   Top-K: {top_k}")
        print(f"   Threshold: {confidence_threshold}")

    def predict_file(self, audio_path: str, aggregate_method: str = "max") -> dict[str, Any]:
        """Predict bird species from audio file."""
        print(f"🔮 Predicting: {Path(audio_path).name}")

        # Process audio
        spectrograms, timestamps = self.audio_processor.process_file(audio_path)

        # Fix: spectrograms is np.ndarray, not list
        if spectrograms.size == 0:  # This works for np.ndarray
            return {
                "file_path": audio_path,
                "error": "No valid segments generated",
                "predictions": [],
            }

        # Get predictions
        predictions = self.predictor.predict_batch(spectrograms)

        # Aggregate predictions
        aggregated = self._aggregate_predictions(predictions, aggregate_method)

        # Get top predictions
        top_predictions = self._get_top_predictions(aggregated)

        return {
            "file_path": audio_path,
            "num_segments": len(spectrograms),
            "timestamps": timestamps,
            "aggregation_method": aggregate_method,
            "predictions": top_predictions,
        }

    def predict_bytes(
        self, audio_bytes: bytes, filename: str = "audio.ogg", aggregate_method: str = "max"
    ) -> dict[str, Any]:
        """Predict from audio bytes (for FastAPI)."""
        print(f"🔮 Predicting bytes: {filename}")

        # Process audio bytes
        spectrograms, timestamps = self.audio_processor.process_bytes(audio_bytes)

        if spectrograms.size == 0:
            return {"filename": filename, "error": "No valid segments generated", "predictions": []}

        # Get predictions
        predictions = self.predictor.predict_batch(spectrograms)

        # Aggregate and format
        aggregated = self._aggregate_predictions(predictions, aggregate_method)
        top_predictions = self._get_top_predictions(aggregated)

        return {
            "filename": filename,
            "num_segments": len(spectrograms),
            "timestamps": timestamps,
            "aggregation_method": aggregate_method,
            "predictions": top_predictions,
        }

    def _aggregate_predictions(
        self, predictions: list[np.ndarray], method: str = "max"
    ) -> np.ndarray:
        """Aggregate predictions from multiple segments."""
        if not predictions:
            return np.array([])

        pred_stack = np.stack(predictions, axis=0)

        if method == "max":
            return np.max(pred_stack, axis=0)
        elif method == "mean":
            return np.mean(pred_stack, axis=0)
        elif method == "vote":
            # Voting based on top-1 predictions
            votes = np.zeros_like(pred_stack[0])
            for pred in predictions:
                top_class = np.argmax(pred)
                votes[top_class] += 1
            return votes / len(predictions)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _get_top_predictions(self, probabilities: np.ndarray) -> list[dict[str, Any]]:
        """Get top-k predictions with species information."""
        if len(probabilities) == 0:
            return []

        # Get indices of top predictions
        top_indices = np.argsort(probabilities.flatten())[::-1][: self.top_k]

        predictions = []
        for rank, idx in enumerate(top_indices):
            confidence = float(probabilities.flatten()[idx])

            # Apply threshold
            if confidence < self.confidence_threshold:
                continue

            predictions.append(
                {
                    "rank": rank + 1,
                    "species_idx": int(idx),
                    "species_name": self.species_names[idx]
                    if idx < len(self.species_names)
                    else f"Species_{idx}",
                    "confidence": confidence,
                    "confidence_percent": f"{confidence * 100:.1f}%",
                }
            )

        return predictions
