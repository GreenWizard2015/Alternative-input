"""Utility functions for exporting models to TensorFlow.js format.

This module contains general utility functions and helper functions that support
the export_js.py script.
"""

import argparse
import io
import json
import os
import numpy as np
import zipfile
from typing import List, Optional, Type

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_model_class_targets() -> List[str]:
    """Get list of supported model class targets for export.

    Returns:
        List of model class names that can be exported
    """
    return [
        "EyeEncoder",
        "EyeEncoderConv",
        "FaceMeshEncoder",
        "Face2StepModel",
        "GazePredictionModel",
        "ModelWrapper",
        "EyeEncoderStage",
        "EmbeddingsProcessor",
        "EmbeddingsTable",
        "Step2LatentModel",
    ]


def get_layer_class_targets() -> List[str]:
    """Get list of supported layer class targets for export.

    Returns:
        List of layer class names that can be exported
    """
    return [
        "CoordsEncodingLayer",
        "LearnablePositionalEncoding",
        "ConvPE",
        "LinearAttentionMixer",
        "MultiHeadAttention",
        "PredictorGaze",
        "PredictorBlock",
        "ShallowEncoderLayer",
        "TimeEncodingLayer",
        "TransformerEncoderBlock",
        "sMLP",
    ]


def serialize_array_to_bytes(
    array: np.ndarray, dtype: Optional[Type] = None
) -> io.BytesIO:
    """Serialize array to BytesIO with optional dtype conversion.

    Args:
        array: NumPy array to serialize
        dtype: Optional dtype to use, defaults to array.dtype

    Returns:
        BytesIO object with serialized data (no shape metadata)
    """
    bio = io.BytesIO()
    target_dtype = dtype if dtype is not None else array.dtype
    bio.write(np.ascontiguousarray(array, dtype=target_dtype).tobytes())
    bio.seek(0)  # Reset pointer to beginning for reading
    return bio


def export_target(
    target: str,
    args: argparse.Namespace = None,
) -> None:
    """Wrapper function for backward compatibility - calls appropriate export function.

    Args:
        target: Target model/layer name to export
        args: Command line arguments object with test attribute
    """
    # Import all export functions at module level for real imports
    from scripts.exportjs.EyeEncoder import export_eyeencoder
    from scripts.exportjs.EyeEncoderConv import export_eyeencoderconv
    from scripts.exportjs.FaceMeshEncoder import export_facemeshencoder
    from scripts.exportjs.EyeEncoderStage import export_eyeencoderstage
    from scripts.exportjs.Face2StepModel import export_face2stepmodel
    from scripts.exportjs.GazePredictionModel import export_gazepredictionmodel
    from scripts.exportjs.ModelWrapper import export_modelwrapper
    from scripts.exportjs.EmbeddingsProcessor import export_embeddingsprocessor
    from scripts.exportjs.EmbeddingsTable import export_embeddingstable
    from scripts.exportjs.Step2LatentModel import export_step2latentmodel
    from scripts.exportjs.CoordsEncodingLayer import export_coordsencodinglayer
    from scripts.exportjs.LearnablePositionalEncoding import export_learnablepositionale
    from scripts.exportjs.ConvPE import export_convpe
    from scripts.exportjs.LinearAttentionMixer import export_linearattentionmixer
    from scripts.exportjs.MultiHeadAttention import export_multiheadattention
    from scripts.exportjs.PredictorGaze import export_predictorgaze
    from scripts.exportjs.PredictorBlock import export_predictorblock
    from scripts.exportjs.ShallowEncoderLayer import export_shallowencoderlayer
    from scripts.exportjs.TimeEncodingLayer import export_timeencodinglayer
    from scripts.exportjs.TransformerEncoderBlock import export_transformerencoderblock
    from scripts.exportjs.Smlp import export_smlp

    # Map target names to corresponding export functions using dictionary
    export_functions = {
        "EyeEncoder": export_eyeencoder,
        "EyeEncoderConv": export_eyeencoderconv,
        "FaceMeshEncoder": export_facemeshencoder,
        "EyeEncoderStage": export_eyeencoderstage,
        "Face2StepModel": export_face2stepmodel,
        "GazePredictionModel": export_gazepredictionmodel,
        "ModelWrapper": export_modelwrapper,
        "EmbeddingsProcessor": export_embeddingsprocessor,
        "EmbeddingsTable": export_embeddingstable,
        "Step2LatentModel": export_step2latentmodel,
        "CoordsEncodingLayer": export_coordsencodinglayer,
        "LearnablePositionalEncoding": export_learnablepositionale,
        "ConvPE": export_convpe,
        "LinearAttentionMixer": export_linearattentionmixer,
        "MultiHeadAttention": export_multiheadattention,
        "PredictorGaze": export_predictorgaze,
        "PredictorBlock": export_predictorblock,
        "ShallowEncoderLayer": export_shallowencoderlayer,
        "TimeEncodingLayer": export_timeencodinglayer,
        "TransformerEncoderBlock": export_transformerencoderblock,
        "sMLP": export_smlp,
    }

    if target not in export_functions:
        raise ValueError(f"Unsupported target: {target}")

    # Pass args object directly
    res = export_functions[target](args)
    data = {}
    for key, val in res.items():
        assert isinstance(val, dict)
        for k, v in val.items():
            if v.dtype == np.float32:
                v = v.astype(np.float64)
            data[f"{key}/{k}.bin"] = v

    shapes = {
        k: {"shape": v.shape if v.shape != () else (1,), "type": str(v.dtype)}
        for k, v in data.items()
    }
    filename = os.path.join(args.output, f"{target}.zip")

    # Save data as zip file
    with zipfile.ZipFile(filename, "w") as zipf:
        for key, value in data.items():
            bio = serialize_array_to_bytes(value)
            zipf.writestr(key, bio.getvalue())

        # Save shapes.json
        zipf.writestr("shapes.json", json.dumps(shapes, indent=2))
