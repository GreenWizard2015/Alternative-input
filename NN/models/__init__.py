"""Neural network model implementations for gaze prediction and filter tasks.

Provides end-to-end models for gaze prediction tasks:
- AdapterMLP: Ultra-thin adapter for knowledge distillation feature matching
- Face2StepModel: Encodes facial data to latent representation
- Step2LatentModel: Processes temporal sequences of latent features
- GazePredictionModel: Combines Face2Step and Step2Latent for end-to-end prediction
- PredictorBlock: Predicts outputs (gaze points or full features) from latent features
- EmbeddingsProcessor: Manages embedding matrices and vocabulary statistics
- EmbeddingsTable: Provides storage functionality for embedding matrices
- EyeEncoder: Encodes eye features for gaze prediction
- FaceMeshEncoder: Encodes facial mesh features

Provides binary classification models for filter tasks:
- FilterModel: Binary classification model for filter validation
"""

from NN.models.AdapterMLP import AdapterMLP
from NN.models.EmbeddingsProcessor import EmbeddingsProcessor
from NN.models.EmbeddingsTable import EmbeddingsTable
from NN.models.EyeEncoder import EyeEncoder, EyeEncoderConv
from NN.models.Face2StepModel import Face2StepModel
from NN.models.FaceMeshEncoder import FaceMeshEncoder
from NN.models.FilterModel import FilterModel
from NN.models.GazePredictionModel import GazePredictionModel
from NN.models.PredictorBlock import PredictorBlock
from NN.models.Step2LatentModel import Step2LatentModel
from NN.models.ResidualAE import ResidualAE

__all__ = [
    "AdapterMLP",
    "EmbeddingsProcessor",
    "EmbeddingsTable",
    "EyeEncoder",
    "EyeEncoderConv",
    "Face2StepModel",
    "FaceMeshEncoder",
    "FilterModel",
    "GazePredictionModel",
    "PredictorBlock",
    "Step2LatentModel",
    "ResidualAE",
]
