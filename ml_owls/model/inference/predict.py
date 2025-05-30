# src/inference/predict.py
"""
Command-line tool for BirdCLEF inference.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ml_owls.model.inference.inference_factory import InferenceFactory
from ml_owls.model.inference.prediction_pipeline import BirdCLEFPredictionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BirdCLEF Species Prediction")

    # Model specification (choose one)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Path to ONNX model file")
    model_group.add_argument("--metadata", type=str, help="Path to conversion metadata JSON")

    # Required arguments
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to OGG audio file or directory"
    )

    # Optional model arguments (only used with --model)
    parser.add_argument(
        "--config", type=str, help="Path to training config (required with --model)"
    )
    parser.add_argument("--taxonomy", type=str, help="Path to taxonomy CSV file")

    # Prediction parameters
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top predictions (default: 5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="max",
        choices=["max", "mean", "vote"],
        help="Aggregation method (default: max)",
    )

    # Performance options
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU inference (disable GPU)")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing (default: 8)"
    )

    # Output options
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--detailed", action="store_true", help="Show detailed segment information")

    return parser.parse_args()


def predict_single_file(
    pipeline: BirdCLEFPredictionPipeline, audio_path: str, args: argparse.Namespace
) -> dict[str, Any]:
    """Predict species for a single audio file."""
    start_time = time.time()

    try:
        results = pipeline.predict_file(audio_path=audio_path, aggregate_method=args.aggregate)

        # Add timing info
        results["processing_time"] = time.time() - start_time
        results["success"] = True

        return results

    except Exception as e:
        return {
            "file_path": audio_path,
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }


def predict_directory(
    pipeline: BirdCLEFPredictionPipeline, audio_dir: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Predict species for all OGG files in directory."""
    audio_path = Path(audio_dir)
    ogg_files = list(audio_path.glob("*.ogg"))

    if not ogg_files:
        print(f"❌ No OGG files found in {audio_dir}")
        return []

    print(f"📁 Found {len(ogg_files)} OGG files")

    all_results = []

    for i, ogg_file in enumerate(ogg_files, 1):
        if not args.quiet:
            print(f"\n🔮 [{i}/{len(ogg_files)}] Processing: {ogg_file.name}")

        results = predict_single_file(pipeline, str(ogg_file), args)
        all_results.append(results)

        # Quick summary
        if not args.quiet and results["success"]:
            if results["predictions"]:
                top_pred = results["predictions"][0]
                print(f"   🏆 Top: {top_pred['species_name']} ({top_pred['confidence_percent']})")
            else:
                print("   ❌ No confident predictions")

    return all_results


def format_predictions(results: dict[str, Any], args: argparse.Namespace) -> None:
    """Format and display prediction results."""
    if not results["success"]:
        print(f"❌ Error: {results['error']}")
        return

    file_name = Path(results["file_path"]).name

    if args.quiet:
        # Minimal output
        if results["predictions"]:
            top = results["predictions"][0]
            print(f"{file_name}: {top['species_name']} ({top['confidence_percent']})")
        else:
            print(f"{file_name}: No confident predictions")
        return

    # Detailed output
    print(f"\n🎵 File: {file_name}")
    print(f"⏱️  Processing time: {results['processing_time']:.2f}s")
    print(f"🔪 Segments: {results['num_segments']}")
    print(f"🔄 Aggregation: {results['aggregation_method']}")

    if not results["predictions"]:
        print("❌ No predictions above confidence threshold")
        return

    print("\n🏆 Top Predictions:")
    for pred in results["predictions"]:
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(pred["rank"] - 1, 4)]
        print(f"   {rank_emoji} {pred['species_name']}: {pred['confidence_percent']}")

    # Show segment details if requested
    if args.detailed and "timestamps" in results:
        print("\n📊 Segment Timestamps:")
        for i, timestamp in enumerate(results["timestamps"]):
            print(f"   Segment {i + 1}: {timestamp:.1f}s - {timestamp + 30:.1f}s")


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    print("🔮 BirdCLEF Species Predictor")
    print("=" * 50)

    # Validate arguments
    if args.model and not args.config:
        raise ValueError("--config is required when using --model")

    # Check input paths
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio path not found: {audio_path}")

    try:
        # Create pipeline
        if args.metadata:
            print(f"🔧 Loading from metadata: {args.metadata}")
            pipeline = InferenceFactory.create_from_converted_model(
                conversion_metadata_path=args.metadata,
                top_k=args.top_k,
                confidence_threshold=args.threshold,
                use_gpu=not args.cpu_only,
            )
        else:
            print(f"🔧 Creating pipeline from model: {args.model}")

            # Load species names from taxonomy if provided
            species_names = None
            if args.taxonomy:
                import pandas as pd

                taxonomy_df = pd.read_csv(args.taxonomy)
                if "primary_label" in taxonomy_df.columns:
                    species_names = taxonomy_df["primary_label"].tolist()
                    print(f"📋 Loaded {len(species_names)} species from taxonomy")

            pipeline = InferenceFactory.create_pipeline(
                onnx_model_path=args.model,
                config_path=args.config,
                species_names=species_names,
                top_k=args.top_k,
                confidence_threshold=args.threshold,
                use_gpu=not args.cpu_only,
            )

        # Process audio
        if audio_path.is_file():
            # Single file
            print(f"\n🎵 Processing single file: {audio_path.name}")
            results = predict_single_file(pipeline, str(audio_path), args)
            format_predictions(results, args)

            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")

        else:
            # Directory
            print(f"\n📁 Processing directory: {audio_path}")
            all_results = predict_directory(pipeline, str(audio_path), args)

            # Summary
            successful = sum(1 for r in all_results if r["success"])
            total_time = sum(r["processing_time"] for r in all_results)

            print("\n📊 Summary:")
            print(f"   Files processed: {len(all_results)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {len(all_results) - successful}")
            print(f"   Total time: {total_time:.1f}s")
            print(f"   Average time: {total_time / len(all_results):.1f}s per file")

            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")

        print("\n🎉 Prediction completed successfully!")

    except Exception as e:
        print(f"\n❌ Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
