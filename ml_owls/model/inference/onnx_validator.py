from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from interfaces.model.model_validator import ModelValidator


class ONNXValidator(ModelValidator):
    """ONNX implementation of model validator."""

    def __init__(self, tolerance: float = 1e-5):
        """Initialize ONNX validator.

        Args:
            tolerance: Maximum allowed difference
        """
        self.tolerance = tolerance
        print(f"🔍 ONNX Validator initialized (tolerance: {tolerance:.2e})")

    def validate(
        self,
        original_model: torch.nn.Module,
        onnx_path: str,
        test_input: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Validate ONNX conversion."""
        print("🔍 Validating ONNX conversion...")

        # Debug: Check if test_input was provided
        if test_input is not None:
            print(f"✅ Received test input with shape: {test_input.shape}")
        else:
            print("⚠️  No test input provided, will create from ONNX model shape")

        try:
            # Load ONNX model
            session = ort.InferenceSession(onnx_path)

            # Get expected input shape from ONNX model itself
            onnx_input_info = session.get_inputs()[0]
            expected_shape = onnx_input_info.shape
            print(f"📊 ONNX model expects: {expected_shape}")

            # Create test input matching ONNX model's expected shape
            if test_input is None:
                print("🔧 Creating test input from ONNX shape...")
                # Use the actual expected shape from the ONNX model
                if any(isinstance(dim, str) for dim in expected_shape):
                    # Handle dynamic dimensions
                    concrete_shape = []
                    for i, dim in enumerate(expected_shape):
                        print(f"🔧 Processing dimension {i}: {dim} (type: {type(dim)})")
                        if isinstance(dim, str):
                            concrete_shape.append(1)  # Use 1 for dynamic dims
                            print("   → Dynamic dimension, using 1")
                        else:
                            concrete_shape.append(dim)
                            print(f"   → Static dimension, using {dim}")

                    print(f"🔧 Final concrete shape: {concrete_shape}")
                    test_input = torch.randn(concrete_shape)
                else:
                    print(f"🔧 All dimensions are static: {expected_shape}")
                    test_input = torch.randn(expected_shape)
            else:
                print("🔧 Using provided test input")

            print(f"🔧 Final test input shape: {test_input.shape}")

            # Test PyTorch model
            original_model.eval()
            with torch.no_grad():
                pytorch_output = original_model(test_input)

            # Test ONNX model
            onnx_input = {onnx_input_info.name: test_input.numpy()}
            onnx_output = session.run(None, onnx_input)[0]

            # Compare outputs
            pytorch_output_np = pytorch_output.detach().numpy()

            max_diff = float(np.max(np.abs(pytorch_output_np - onnx_output)))
            mean_diff = float(np.mean(np.abs(pytorch_output_np - onnx_output)))

            print("📊 Output comparison:")
            print(f"   Max difference: {max_diff:.2e}")
            print(f"   Mean difference: {mean_diff:.2e}")
            print(f"   Tolerance: {self.tolerance:.2e}")

            success = max_diff < self.tolerance

            if success:
                print("✅ Validation passed!")
            else:
                print(
                    f"❌ Validation failed: max difference {max_diff:.2e} exceeds tolerance {self.tolerance:.2e}"
                )

            return {
                "validation_passed": success,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "tolerance": self.tolerance,
                "pytorch_output_shape": list(pytorch_output.shape),
                "onnx_output_shape": list(onnx_output.shape),
                "test_input_shape": list(test_input.shape),
            }

        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return {"validation_passed": False, "error": str(e)}
