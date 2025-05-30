from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.onnx

from interfaces.model.model_converter import ModelConverter


class ONNXConverter(ModelConverter):
    """ONNX implementation of model converter."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 1, 128, 938),
        opset_version: int = 11,
        dynamic_batch: bool = True,
    ):
        """Initialize ONNX converter.

        Args:
            input_shape: Expected input shape
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size
        """
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch

        print("🔄 ONNX Converter initialized")
        print(f"   Input shape: {input_shape}")
        print(f"   Opset version: {opset_version}")
        print(f"   Dynamic batch: {dynamic_batch}")

    def convert(self, model: torch.nn.Module, output_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Convert PyTorch model to ONNX format."""
        print(f"🚀 Converting to ONNX: {output_path}")

        # Ensure model is in evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(self.input_shape)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Configure dynamic axes
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {"spectrogram": {0: "batch_size"}, "predictions": {0: "batch_size"}}

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=["spectrogram"],
                output_names=["predictions"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
            print("✅ ONNX export completed")

        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {str(e)}")

        # Return conversion metadata
        return {
            "format": "ONNX",
            "input_shape": self.input_shape,
            "output_path": output_path,
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "conversion_status": "success",
        }
