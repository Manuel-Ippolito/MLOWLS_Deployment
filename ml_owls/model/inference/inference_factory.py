import json
from pathlib import Path
from typing import Any

from ml_owls.model.config import Config
from ml_owls.model.inference.audio_processing_factory import AudioProcessingFactory
from ml_owls.model.inference.onnx_predictor import ONNXPredictor
from ml_owls.model.inference.prediction_pipeline import BirdCLEFPredictionPipeline


class InferenceFactory:
    """Factory for creating complete inference pipelines."""

    @staticmethod
    def create_pipeline(
        onnx_model_path: str,
        config_path: str,
        species_names: list[str] | None = None,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
        use_gpu: bool = True,
    ) -> BirdCLEFPredictionPipeline:
        """Create complete inference pipeline."""
        print("🏭 Creating inference pipeline...")
        print(f"   Model: {Path(onnx_model_path).name}")
        print(f"   Config: {Path(config_path).name}")

        # Load configuration
        config = Config(config_path)

        # Create audio processor using EXACT training pipeline
        audio_processor = AudioProcessingFactory.create_processor(config)

        # Create predictor
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        predictor = ONNXPredictor(onnx_model_path, providers=providers)

        # Load species names
        if species_names is None:
            species_names = InferenceFactory._load_species_names(config)

        # Create prediction pipeline
        pipeline = BirdCLEFPredictionPipeline(
            audio_processor=audio_processor,
            predictor=predictor,
            species_names=species_names,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        print("✅ Inference pipeline ready!")
        return pipeline

    @staticmethod
    def _load_species_names(config: Config) -> list[str]:
        """Load species names from taxonomy."""
        try:
            # Use the taxonomy path from config
            paths_config = getattr(config, "paths", {})
            taxonomy_path = paths_config.get("taxonomy_csv", "data/taxonomy.csv")

            print(f"📋 Loading species names from: {taxonomy_path}")

            import pandas as pd

            taxonomy_df = pd.read_csv(taxonomy_path)

            if "primary_label" in taxonomy_df.columns:
                species_list = taxonomy_df["primary_label"].tolist()
                species_names: list[str] = [
                    str(name) for name in species_list
                ]  # Convert to strings
                print(f"📋 Loaded {len(species_names)} species from taxonomy")
                return species_names
            else:
                print("⚠️  'primary_label' column not found in taxonomy")

        except Exception as e:
            print(f"⚠️  Could not load taxonomy: {e}")

        print("📋 Using generic species names (206 classes)")
        return [f"species_{i:03d}" for i in range(206)]

    @staticmethod
    def create_from_converted_model(
        conversion_metadata_path: str,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
        use_gpu: bool = True,
    ) -> BirdCLEFPredictionPipeline:
        """Create pipeline from conversion metadata."""
        print(f"🔧 Creating pipeline from metadata: {Path(conversion_metadata_path).name}")

        # Load conversion metadata
        with open(conversion_metadata_path, "r") as f:
            metadata: dict[str, Any] = json.load(f)

        # Extract paths
        onnx_path = str(metadata["output_path"])
        config_path = str(metadata["original_model"]["config_path"])

        print(f"   ONNX model: {Path(onnx_path).name}")
        print(f"   Config: {Path(config_path).name}")

        return InferenceFactory.create_pipeline(
            onnx_model_path=onnx_path,
            config_path=config_path,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )
