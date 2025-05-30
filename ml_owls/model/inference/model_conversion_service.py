from pathlib import Path
from typing import Any, Optional

import torch

from ml_owls.model.config import Config
from interfaces.model.metadata_handler import MetadataHandler
from interfaces.model.model_converter import ModelConverter
from interfaces.model.model_loader import ModelLoader
from interfaces.model.model_validator import ModelValidator


class ModelConversionService:
    """Service for orchestrating model conversion with dependency injection."""

    def __init__(
        self,
        converter: ModelConverter,
        model_loader: ModelLoader,
        validator: Optional[ModelValidator] = None,
        metadata_handler: Optional[MetadataHandler] = None,
    ) -> None:
        """Initialize conversion service.

        Args:
            converter: Model converter implementation
            model_loader: Model loader implementation
            validator: Optional model validator
            metadata_handler: Optional metadata handler
        """
        self.converter = converter
        self.model_loader = model_loader
        self.validator = validator
        self.metadata_handler = metadata_handler

        print("🏭 Model Conversion Service initialized")
        print(f"   Converter: {type(converter).__name__}")
        print(f"   Model Loader: {type(model_loader).__name__}")
        print(f"   Validator: {type(validator).__name__ if validator else 'None'}")
        print(
            f"   Metadata Handler: {type(metadata_handler).__name__ if metadata_handler else 'None'}"
        )

    def convert_model(
        self,
        model_path: str,
        config_path: str,
        output_path: str,
        validate: bool = True,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert model with full pipeline."""
        print("🔄 Starting model conversion...")
        print(f"   Model: {Path(model_path).name}")
        print(f"   Config: {Path(config_path).name}")
        print(f"   Output: {output_path}")

        # Load config if not provided
        if config is None:
            config = Config(config_path)

        # Load model
        model = self.model_loader.load_model(model_path, config_path=config_path)

        # Convert model
        conversion_results = self.converter.convert(model, output_path, **kwargs)

        conversion_results["original_model"] = {
            "model_path": model_path,
            "config_path": config_path,
        }

        # Validate if requested
        if validate and self.validator:
            print("🔍 Validating converted model...")

            # Use config to determine input shape
            input_shape = (1, 1, config.n_mels, 3001)  # batch_size, channels, n_mels, time_frames
            test_input = torch.randn(input_shape)
            print(f"🔧 Using config-based input shape: {input_shape}")
            print(f"🔧 Created test input with shape: {test_input.shape}")

            validation_results = self.validator.validate(model, output_path, test_input)
            conversion_results["validation"] = validation_results

        # Save metadata with config info
        if self.metadata_handler:
            metadata_path = output_path.replace(".onnx", "_conversion.json")

            # Add config details to conversion results
            conversion_results["config_info"] = {
                "n_mels": config.n_mels,
                "sample_rate": config.sample_rate,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "input_shape": input_shape,
            }

            self.metadata_handler.save_metadata(conversion_results, metadata_path)
            print(f"💾 Conversion metadata saved: {metadata_path}")

        print("✅ Model conversion completed successfully!")
        return conversion_results
