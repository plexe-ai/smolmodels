#!/usr/bin/env python
"""
CLI interface for SmolModels, allowing model building, prediction, and management from the command line.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

import smolmodels as sm


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_data_from_file(file_path: str) -> pd.DataFrame:
    """Load data from a CSV or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text()))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Please use CSV or JSON.")


def load_json_schema(file_path: str) -> Dict[str, type]:
    """Load schema from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {file_path}")

    schema_dict = json.loads(path.read_text())

    # Convert string type names to actual types
    type_map = {"str": str, "int": int, "float": float, "bool": bool}
    return {key: type_map.get(value, str) for key, value in schema_dict.items()}


def parse_key_value_pairs(pairs: List[str]) -> Dict[str, Any]:
    """Parse key-value pairs from command line arguments."""
    if not pairs:
        return {}

    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid key-value pair: {pair}. Expected format: key=value")

        key, value = pair.split("=", 1)
        # Try to convert value to appropriate type
        try:
            # First try as int
            result[key] = int(value)
        except ValueError:
            try:
                # Then as float
                result[key] = float(value)
            except ValueError:
                # Finally as string or bool
                if value.lower() in ("true", "yes"):
                    result[key] = True
                elif value.lower() in ("false", "no"):
                    result[key] = False
                else:
                    result[key] = value

    return result


def build_model(args) -> None:
    """Build a model from CLI arguments."""
    # Load input and output schemas if provided
    input_schema = None
    if args.input_schema:
        input_schema = load_json_schema(args.input_schema)

    output_schema = None
    if args.output_schema:
        output_schema = load_json_schema(args.output_schema)

    # Create the model
    model = sm.Model(
        intent=args.intent,
        input_schema=input_schema,
        output_schema=output_schema,
    )

    # Load dataset(s)
    datasets = []
    if args.dataset:
        for dataset_path in args.dataset:
            datasets.append(load_data_from_file(dataset_path))

    # Build the model
    print(f"Building model with provider: {args.provider}")
    model.build(
        datasets=datasets,
        provider=args.provider,
        timeout=args.timeout,
        max_iterations=args.max_iterations,
        run_timeout=args.run_timeout,
        verbose=False,
    )

    # Save the model if requested
    if args.output:
        output_path = args.output
        if not output_path.endswith(".tar.gz"):
            output_path = f"{output_path}.tar.gz"

        sm.save_model(model, output_path)
        print(f"Model saved to: {output_path}")

    # Display model information
    print("\nModel Information:")
    print(f"Intent: {model.intent}")
    print(f"State: {model.get_state().value}")

    metrics = model.get_metrics()
    if metrics:
        print("\nPerformance Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")

    metadata = model.get_metadata()
    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            if key not in ("provider", "creation_date"):
                print(f"  {key}: {value}")


def predict(args) -> None:
    """Make predictions using a loaded model."""
    # Load the model
    try:
        model = sm.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Handle different input methods
    input_data = {}
    if args.input_file:
        # Load from file
        data = load_data_from_file(args.input_file)
        if len(data) != 1:
            print("Input file must contain exactly one row for prediction")
            sys.exit(1)
        input_data = data.iloc[0].to_dict()
    elif args.input_values:
        # Parse from command line arguments
        input_data = parse_key_value_pairs(args.input_values)
    else:
        print("No input data provided. Use --input-file or --input-values")
        sys.exit(1)

    # Make prediction
    try:
        result = model.predict(input_data, validate_input=args.validate)

        # Print the result
        if args.output_file:
            # Save to file
            output_path = Path(args.output_file)
            if output_path.suffix.lower() == ".json":
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
            elif output_path.suffix.lower() == ".csv":
                pd.DataFrame([result]).to_csv(output_path, index=False)
            else:
                print(f"Unsupported output format: {output_path.suffix}. Using JSON.")
                with open(f"{output_path}.json", "w") as f:
                    json.dump(result, f, indent=2)

            print(f"Prediction saved to: {output_path}")
        else:
            # Print to console
            print("\nPrediction:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def show_model_info(args) -> None:
    """Display information about a model."""
    try:
        model = sm.load_model(args.model)

        # Get model description
        description = model.describe()

        if args.format == "json":
            # Output as JSON
            print(json.dumps(description.to_dict(), indent=2))
        else:
            # Output as text
            print(description.as_markdown())

    except Exception as e:
        print(f"Error showing model info: {e}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SmolModels CLI - Build, use, and manage ML models from the command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a new model")
    build_parser.add_argument("--intent", required=True, help="Natural language description of model intent")
    build_parser.add_argument("--input-schema", help="Path to JSON file defining input schema")
    build_parser.add_argument("--output-schema", help="Path to JSON file defining output schema")
    build_parser.add_argument("--dataset", "-d", action="append", help="Path to dataset file(s) (CSV or JSON)")
    build_parser.add_argument("--provider", default="openai/gpt-4o-mini", help="LLM provider to use")
    build_parser.add_argument("--timeout", type=int, default=3600, help="Maximum build time in seconds")
    build_parser.add_argument("--max-iterations", type=int, help="Maximum number of model iterations")
    build_parser.add_argument("--run-timeout", type=int, default=1800, help="Timeout for individual model runs")
    build_parser.add_argument("--output", "-o", help="Output path for the model file (.tar.gz)")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions using a model")
    predict_parser.add_argument("model", help="Path to the model file (.tar.gz)")
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument("--input-file", help="Path to input data file (CSV or JSON)")
    predict_group.add_argument("--input-values", nargs="+", help="Input values as key=value pairs")
    predict_parser.add_argument("--output-file", help="Path to save prediction output")
    predict_parser.add_argument("--validate", action="store_true", help="Validate input against schema")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display information about a model")
    info_parser.add_argument("model", help="Path to the model file (.tar.gz)")
    info_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    # Execute requested command
    if args.command == "build":
        build_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "info":
        show_model_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
