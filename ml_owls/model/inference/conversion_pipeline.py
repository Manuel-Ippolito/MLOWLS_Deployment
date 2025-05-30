from typing import Any

from ml_owls.model.config import Config
from ml_owls.model.inference.json_metadata_handler import JSONMetadataHandler
from ml_owls.model.inference.model_conversion_service import ModelConversionService
from ml_owls.model.inference.onnx_converter import ONNXConverter
from ml_owls.model.inference.onnx_validator import ONNXValidator
from ml_owls.model.inference.trained_model_loader import TrainedModelLoader


class ConversionPipeline:
    """Complete pipeline for model conversion."""

    @staticmethod
    def create_onnx_pipeline(config_path: str) -> ModelConversionService:
        """Create a complete ONNX conversion pipeline."""
        print("🏭 Creating ONNX conversion pipeline...")

        # Load config for model loader
        config = Config(config_path)

        # Create components
        converter = ONNXConverter(config=config)
        model_loader = TrainedModelLoader(config)
        validator = ONNXValidator()
        metadata_handler = JSONMetadataHandler()

        # Create service
        service = ModelConversionService(
            converter=converter,
            model_loader=model_loader,
            validator=validator,
            metadata_handler=metadata_handler,
        )

        print("✅ ONNX conversion pipeline ready!")
        return service

    @staticmethod
    def convert_model(
        model_path: str, config_path: str, output_path: str, validate: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Convert model using the complete pipeline."""
        # Pass config_path to create_onnx_pipeline
        pipeline = ConversionPipeline.create_onnx_pipeline(config_path)

        return pipeline.convert_model(
            model_path=model_path,
            config_path=config_path,
            output_path=output_path,
            validate=validate,
            **kwargs,
        )
