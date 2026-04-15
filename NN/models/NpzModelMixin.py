"""Mixin class for adding NPZ loading and saving functionality to Keras models.

This mixin provides standardized save() and load() methods for Keras models
that use NPZ format for saving weights and architecture.
"""

from NN.models.npz_utils import load_model_from_npz, save_model_to_npz


class NpzModelMixin:
    """Mixin class that adds NPZ save/load functionality to Keras models.

    To use this mixin, inherit from both this class and tf.keras.Model:

        class MyModel(NpzModelMixin, tf.keras.Model):
            def __init__(self, ...):
                # Initialize model
                pass

    The mixin provides:
    - save_npz(model_path: str) -> None: Save model to NPZ format
    - load_npz(model_path: str, force: bool = False) -> None: Load model from NPZ format
    """

    def save_npz(self, model_path: str) -> None:
        """Save model architecture and weights to NPZ format.

        Args:
            model_path: Path to save model file (without extension)
        """
        # self is already a tf.keras.Model when mixed in properly
        save_model_to_npz(self, model_path)

    def load_npz(self, model_path: str, force: bool = False) -> None:
        """Load model weights from NPZ format.

        Loads weights into existing model instance (architecture is immutable).
        Assumes model architecture already defined in __init__.

        Args:
            model_path: Path to weights file (without extension)
            force: Whether to continue with random initialization if weights are missing (default: False)
        """
        # self is already a tf.keras.Model when mixed in properly
        load_model_from_npz(self, model_path, force=force)
