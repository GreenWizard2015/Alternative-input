"""Layer for predicting eye images from latent features."""

from typing import Any
import tensorflow as tf
import tensorflow.keras.layers as L
from NN.Constants import PREDICTOR_BLOCK_ACTIVATION, PREDICTOR_BLOCK_MLP_SIZES
from NN.layers.EncodingLayers import ConvPE
from NN.layers.sMLP import sMLP


class PredictorEyes(tf.keras.layers.Layer):
    """Predicts left and right eye images from latent representations.

    Architecture:
        - sMLP: Processes latent features through multiple dense layers
        - Dense projection: Expands latent to initial spatial features (8x8)
        - Conv decoder: Progressively upsamples and refines eye images
          - 3 upsampling blocks with Conv2DTranspose
          - Each block: Upsample + Conv2D + ReLU
        - Clipping: Constrains predictions to valid range [0, 1]

    Attributes:
        eye_size: Size of eye images (height/width)
        _mlp: sMLP for feature processing
        _dense_proj: Dense layer projecting latent to 8x8x64 features
        _conv_decoder: Sequential decoder with upsampling layers and ConvPE at each stage
    """

    def __init__(
        self,
        eye_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """Initialize the PredictorEyes layer.

        Args:
            eye_size: Size of eye images (height/width). Default: 32.
            **kwargs: Additional keyword arguments passed to parent Layer class.
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)
        self.eye_size = eye_size

        # Initial spatial size for decoder (8x8)
        self._init_size = 8
        self._init_channels = 64

        # Store sublayer names directly

    def build(self, input_shape):
        """Build the layer with input shape.

        Args:
            input_shape: Input tensor shape (batch, seq_len, latent_dim)
        """
        # Create sMLP for feature processing
        self._mlp = sMLP(
            sizes=PREDICTOR_BLOCK_MLP_SIZES,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="MLP",
        )
        self._mlp.build(input_shape)

        # Dense projection receives MLP output (batch, seq_len, MLP_OUTPUT_SIZE)
        # MLP output size is PREDICTOR_BLOCK_MLP_SIZES[-1]
        mlp_output_size = PREDICTOR_BLOCK_MLP_SIZES[-1]
        self._dense_proj = L.Dense(
            units=self._init_size * self._init_size * self._init_channels,
            activation=PREDICTOR_BLOCK_ACTIVATION,
            name="DenseProj",
        )
        self._dense_proj.build((input_shape[0], input_shape[1], mlp_output_size))

        # Create convolutional decoder with upsampling and positional encoding
        self._conv_decoder = tf.keras.Sequential(
            [
                # Upsample to 16x16
                L.Conv2DTranspose(
                    32,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Upsample1",
                ),
                ConvPE(
                    channels=16,
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="ConvPE1",
                ),
                # Conv1a and Conv1b blocks
                L.Conv2D(
                    32,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Conv1a",
                ),
                L.Conv2D(
                    32,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Conv1b",
                ),
                # Upsample to 32x32
                L.Conv2DTranspose(
                    16,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Upsample2",
                ),
                ConvPE(
                    channels=16,
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="ConvPE2",
                ),
                # Conv2a, Conv2b, Conv2c blocks
                L.Conv2D(
                    16,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Conv2a",
                ),
                L.Conv2D(
                    16,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Conv2b",
                ),
                L.Conv2D(
                    16,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="Conv2c",
                ),
                # Final positional encoding before output
                ConvPE(
                    channels=16,
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="ConvPE3",
                ),
                L.Conv2D(
                    8,
                    kernel_size=3,
                    padding="same",
                    activation=PREDICTOR_BLOCK_ACTIVATION,
                    name="ConvFinal",
                ),
                # Final output layer: 2 channels (left and right eye)
                L.Conv2D(
                    2,
                    kernel_size=3,
                    padding="same",
                    activation="linear",
                    name="Output",
                ),
            ],
            name="ConvDecoder",
        )

        # Build the conv decoder sublayer
        # Conv decoder receives (batch*seq_len, 8, 8, 64)
        self._conv_decoder.build((input_shape[0] * input_shape[1], 8, 8, 64))

        super().build(input_shape)

    def call(
        self, latent_features: tf.Tensor, training: bool = False, **kwargs: Any
    ) -> tf.Tensor:
        """Predict left and right eye images from latent features.

        Args:
            latent_features: Input tensor of shape (batch, seq_len, latent_dim).
            training: Whether in training mode (default: False).
            **kwargs: Additional call arguments.

        Returns:
            Eye images tensor of shape (batch, seq_len, eye_size, eye_size, 2)
            where the last dimension contains [left_eye, right_eye], clipped to [0, 1]

        Example:
            >>> predictor = PredictorEyes(eye_size=32)
            >>> latent = tf.random.normal((4, 10, 256))
            >>> eyes = predictor(latent)
            >>> assert eyes.shape == (4, 10, 32, 32, 2)
        """
        batch_size = tf.shape(latent_features)[0]
        sequence_length = tf.shape(latent_features)[1]

        # Process through sMLP: (batch, seq_len, latent_dim) -> (batch, seq_len, MLP_output_dim)
        mlp_output = self._mlp(latent_features, training=training)

        # Project latent to initial spatial features: (batch, seq_len, 8*8*64)
        projected_features = self._dense_proj(mlp_output, training=training)

        # Reshape to spatial: (batch, seq_len, 8, 8, 64)
        spatial_features = tf.reshape(
            projected_features, (batch_size, sequence_length, 8, 8, 64)
        )

        # Flatten batch and seq_len for conv decoder
        # (batch*seq_len, 8, 8, 64)
        flattened_features = tf.reshape(
            spatial_features, (batch_size * sequence_length, 8, 8, 64)
        )

        # Apply convolutional decoder with ConvPE at each stage: (batch*seq_len, 8, 8, 64) -> (batch*seq_len, 32, 32, 2)
        eye_images = self._conv_decoder(flattened_features, training=training)

        # Reshape back to separate batch and seq_len: (batch, seq_len, 32, 32, 2)
        eye_images = tf.reshape(
            eye_images, (batch_size, sequence_length, self.eye_size, self.eye_size, 2)
        )

        return eye_images
