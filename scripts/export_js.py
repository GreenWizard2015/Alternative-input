#!/usr/bin/env python3
"""Export TensorFlow.js weights from trained ModelWrapper.

This script loads a trained ModelWrapper and exports weights for each sub-model
as raw float32 binary files packaged in .zip archives. Optionally generates
test data and inference results for validation.

Individual exports use --target flag, all other exports are always consolidated.

Usage:
    # Export specific target (individual model/layer) - uses random weights
    conda run -n myenv python export_js.py --target AdapterMLP --output ../web-client/src/models/
    conda run -n myenv python export_js.py --target sMLP --output ../web-client/src/models/

    # Export all weights to single consolidated ModelWrapper.zip from checkpoint
    conda run -n myenv python export_js.py --output ../web-client/src/models/

    # Export with test data from checkpoint
    conda run -n myenv python export_js.py --output ../web-client/src/models/ --test

    # With conda environment
    conda run -n myenv python scripts/export_js.py --output ../web-client/src/models/
"""

import argparse
import os
from pathlib import Path

# Import utility functions from export_utils module
from export_utils import (
    export_target,
    get_layer_class_targets,
    get_model_class_targets,
)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(args: argparse.Namespace) -> None:
    """Main entry point for weight export.

    Args:
        args: Parsed command-line arguments
    """
    os.makedirs(args.output, exist_ok=True)
    print("Starting export workflow...")
    targets = [args.target]
    # Handle "all" target case
    if args.target.lower() == "all":
        print("Exporting all targets...")
        targets = get_model_class_targets() + get_layer_class_targets()

    for target in targets:
        print(f"Exporting {target}...")
        export_target(target=target, args=args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export TensorFlow.js weights from trained ModelWrapper with universal export workflow"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target model/layer to export. Use 'all' to export all targets. Available: Model classes: EyeEncoder, Face2StepModel, FaceMeshEncoder, GazePredictionModel, PredictorBlock, Step2LatentModel, EmbeddingsProcessor, EmbeddingsTable. Layer classes: sMLP, MultiHeadAttention, TransformerEncoderBlock, LinearAttentionMixer, CoordsEncodingLayer, LearnablePositionalEncoding, ConvPE, RolloutTimesteps, ShallowEncoderLayer, TimeEncodingLayer, PredictorGaze. Note: When --target is specified, --checkpoint is ignored and models use random weights.",
        choices=get_layer_class_targets() + get_model_class_targets() + ["all"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/",
        help="Output directory for .zip files (default: exports/)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, also package test results and test_inputs into .zip files",
    )

    parser.add_argument(
        "--checkpoint",
        default="models/",
        help="Path to checkpoint directory containing trained ModelWrapper (default: models/). "
        "Only used when exporting the full ModelWrapper (when --target is not specified).",
    )

    args = parser.parse_args()
    main(args)
