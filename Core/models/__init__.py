"""Gaze prediction model trainers and wrappers.

Provides orchestration classes for training and inference:
- ModelWrapper: Coordinates gaze prediction pipeline with embeddings and neural networks
- ModelStudentTrainer: Specialized trainer for student models with teacher guidance
- ModelWrapperProxies: Fast proxy functions for testing without model instantiation
"""

from .ModelWrapper import ModelWrapper
from .ModelStudentTrainer import ModelStudentTrainer
from .ModelWrapperProxies import ModelWrapperProxies
from .PredictionOutputTypes import PredictionOutput, PredictionOutputNumpy
from .FilterWrapper import FilterWrapper

__all__ = [
    "ModelWrapper",
    "ModelStudentTrainer",
    "ModelWrapperProxies",
    "PredictionOutput",
    "PredictionOutputNumpy",
    "FilterWrapper",
]
