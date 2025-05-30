import argparse

from ml_owls.model.inference.conversion_pipeline import ConversionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert BirdCLEF PyTorch model to ONNX")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to PyTorch model file (.pth)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--no-validate", action="store_true", help="Skip conversion validation")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    parser.add_argument(
        "--static-batch", action="store_true", help="Use static batch size (no dynamic batching)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Validation tolerance (default: 1e-5)"
    )

    return parser.parse_args()


def main() -> None:
    """Main conversion function."""
    args = parse_args()

    print("🔄 Starting model conversion...")
    print(f"   Model: {args.model}")
    print(f"   Config: {args.config}")
    print(f"   Output: {args.output}")

    try:
        results = ConversionPipeline.convert_model(
            model_path=args.model,
            config_path=args.config,
            output_path=args.output,
            validate=not args.no_validate,
            opset_version=args.opset,
            dynamic_batch=not args.static_batch,
        )

        print("\n🎉 Conversion completed successfully!")
        print(f"📁 ONNX model: {results['output_path']}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
