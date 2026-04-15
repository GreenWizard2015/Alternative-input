"""EmbeddingsTable component for managing embedding matrices and vocabulary statistics.

This component provides pure storage functionality for embedding matrices without
any mixing or processing logic. Vocab is fixed at initialization time (immutable after
instantiation). Default IDs must be passed to call() method.

Usage Example:
    from NN.models.EmbeddingsTable import EmbeddingsTable

    # Create vocab dict from your dataset
    vocab = {"userId": 100, "screenId": 50, "cameraId": 20,
             "monitorId": 30, "placeId": 25}

    # Create embeddings table with vocab at init (all layers created immediately)
    table = EmbeddingsTable(vocab=vocab, embedding_size=64)

    # Call with ID tensors (layers already created)
    # Pass defaults for any None IDs
    embeddings = table(
        userId=user_ids_tensor,
        screenId=screen_ids_tensor,
        cameraId=camera_ids_tensor,
        monitorId=monitor_ids_tensor,
        placeId=place_ids_tensor,
        default_ids={"userId": 0, "screenId": 0, ...},
        shape=shape_tensor,
    )
"""

from typing import Any, Dict

import tensorflow as tf
from NN.models.NpzModelMixin import NpzModelMixin
from Core.logging_config import get_logger
from Core.Constants import HIERARCHY_LEVELS

logger = get_logger(__name__)


class EmbeddingsTable(NpzModelMixin, tf.keras.Model):
    """Manages embedding matrices and vocabulary statistics.

    This component is responsible for:
    - Creating and managing embedding matrices for different ID types
    - Looking up embeddings from ID indices
    - Handling default ID substitution
    - No mixing or processing logic (pure storage only)

    Args:
        vocab: Dict mapping ID types to vocabulary sizes. MUST include:
               userId, screenId, cameraId, monitorId, placeId
               Vocab is IMMUTABLE after initialization. Create new instance
               to use different vocab.
        embedding_size: Dimension of embedding vectors (default: 64)
        **kwargs: Additional keyword arguments

    Note:
        Vocab is passed to __init__() and is IMMUTABLE (create new instance
        to change vocab). Default IDs are passed to call() method when needed.
        This design ensures the model is fully built at initialization time.

    Raises:
        ValueError: If vocab is missing required HIERARCHY_LEVELS keys
    """

    def __init__(
        self, vocab: Dict[str, int], embedding_size: int = 64, **kwargs: Any
    ) -> None:
        """Initialize EmbeddingsTable with vocab.

        Args:
            vocab: Dict mapping ID types to vocabulary sizes.
                   MUST include all of: userId, screenId, cameraId,
                   monitorId, placeId
                   Example: {"userId": 1000, "screenId": 50, ...}
                   Vocab is IMMUTABLE after init - create new instance
                   for different vocab.
            embedding_size: Dimension of embedding vectors (default: 64)
            **kwargs: Additional kwargs passed to tf.keras.Model

        Raises:
            ValueError: If vocab missing any required HIERARCHY_LEVELS keys or if
                       embedding_size is not positive.
        """
        super().__init__(**kwargs)

        # Input validation
        if embedding_size <= 0:
            raise ValueError(f"embedding_size must be positive, got {embedding_size}")

        # Validate vocab has all required keys
        missing = set(HIERARCHY_LEVELS) - set(vocab.keys())
        if missing:
            raise ValueError(f"vocab missing required keys: {missing}")

        self.embedding_size = embedding_size
        self._vocab = vocab

        # Create ALL layers at init time - Keras tracks them immediately
        self._embedding_layers: Dict[str, tf.keras.layers.Embedding] = {}

        # Pattern from EmbeddingBlock.py (proven to work)
        for id_key in HIERARCHY_LEVELS:  # Consistent order
            layer = self._create_embedding_layer(id_key, vocab[id_key])
            self._embedding_layers[id_key] = layer
            # CRITICAL: Use setattr() so Keras sees dynamically-created layers
            setattr(self, f"_{id_key}_embedding", layer)
            # Build the layer with expected input shape (batch, 1)
            layer.build((None, 1))

        # Mark model as built so Keras knows it's ready to save
        # Use super().build() instead of direct assignment to maintain Keras state
        super().build({})

    def _create_embedding_layer(
        self, id_key: str, vocab_size: int
    ) -> tf.keras.layers.Embedding:
        """Create embedding layer for specific ID type.

        Args:
            id_key: Name of ID type (userId, screenId, etc.)
            vocab_size: Size of vocabulary for this ID type

        Returns:
            Keras embedding layer
        """
        embedding_layer = tf.keras.layers.Embedding(
            vocab_size,
            self.embedding_size,
            name=f"{id_key}_embedding",
        )
        return embedding_layer

    def call(
        self,
        userId=None,
        screenId=None,
        cameraId=None,
        monitorId=None,
        placeId=None,
        default_ids=None,
        shape=None,
        training=False,
    ) -> tf.Tensor:
        """Generate embeddings from ID tensors using pre-created layers.

        Args:
            userId, screenId, cameraId, monitorId, placeId: ID tensors or None (use defaults)
            default_ids: Dict mapping ID types to default indices for None tensors.
                        Example: {"userId": 0, "screenId": 0, ...}
            shape: [batch_size, timesteps] - used for batch size inference
            training: Training mode flag

        Returns:
            Concatenated embeddings (batch, 1, 5*embedding_size)

        Raises:
            ValueError: If tensor missing and no default available
        """
        # Extract ID tensors using HIERARCHY_LEVELS (eliminates hardcoding)
        # Build dict explicitly to get the parameter values
        all_locals = {
            "userId": userId,
            "screenId": screenId,
            "cameraId": cameraId,
            "monitorId": monitorId,
            "placeId": placeId,
        }
        id_tensors = {id_key: all_locals[id_key] for id_key in HIERARCHY_LEVELS}

        # Use provided defaults (or empty dict if not provided)
        defaults = default_ids or {}
        batch_size = shape[0]
        embeddings_list = []
        for id_key in HIERARCHY_LEVELS:
            tensor = id_tensors[id_key]

            # Use provided tensor or fill with default
            if tensor is None:
                if id_key not in defaults:
                    raise ValueError(f"No value or default for '{id_key}'")

                # Fill with default: (batch_size, 1)
                tensor = tf.fill([batch_size, 1], defaults[id_key])

            # Extract first timestep
            if len(tensor.shape) < 2:
                tensor_timestep = tf.reshape(tensor, [1, 1])
            tensor_timestep = tensor[:, :1]  # all values in sample are same

            # Remove extra dimensions
            if len(tensor_timestep.shape) > 2:
                tensor_timestep = tf.squeeze(tensor_timestep, axis=-1)

            # Use PRE-CREATED layer (no creation here!)
            layer = self._embedding_layers[id_key]
            embedding = layer(tensor_timestep, training=training)
            embeddings_list.append(embedding)

        return tf.concat(embeddings_list, axis=-1)

    def reset_embeddings(self):
        for v in self._embedding_layers.values():
            for w in v.variables:
                w.assign(tf.random.normal(w.shape, 0.0, 1e-2, w.dtype))
