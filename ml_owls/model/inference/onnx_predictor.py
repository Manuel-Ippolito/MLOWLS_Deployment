from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from interfaces.model.predictor import Predictor


class ONNXPredictor(Predictor):
    """ONNX Runtime predictor implementation."""

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Initialize ONNX predictor."""
        self.model_path = model_path

        # Default providers
        if providers is None:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

        # Create session
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print("⚡ ONNX Predictor initialized")
            print(f"   Model: {Path(model_path).name}")
            print(f"   Providers: {providers}")

            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape

            print(f"   Input: {self.input_name} {input_shape}")
            print(f"   Output: {self.output_name} {output_shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model {model_path}: {str(e)}")

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make prediction on single input."""
        # Ensure input is float32
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)

        # Run inference
        try:
            outputs = self.session.run([self.output_name], {self.input_name: inputs})
            return outputs[0]

        except Exception as e:
            raise RuntimeError(f"ONNX prediction failed: {str(e)}")

    def predict_batch(self, spectrograms: np.ndarray) -> list[np.ndarray]:
        """Predict batch of spectrograms."""

        # Ensure input is float32
        if spectrograms.dtype != np.float32:
            spectrograms = spectrograms.astype(np.float32)

        try:
            # Run ONNX inference on the entire batch
            raw_predictions = self.session.run(None, {self.input_name: spectrograms})[0]

            # Apply softmax to convert logits to probabilities
            def softmax(x: np.ndarray) -> np.ndarray:
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

            # Normalize each prediction
            normalized_predictions = softmax(raw_predictions)

            # Return as list of individual predictions
            return [normalized_predictions[i] for i in range(len(normalized_predictions))]

        except Exception as e:
            raise RuntimeError(f"ONNX batch prediction failed: {str(e)}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers(),
            "input_shape": self.session.get_inputs()[0].shape,
            "output_shape": self.session.get_outputs()[0].shape,
        }
