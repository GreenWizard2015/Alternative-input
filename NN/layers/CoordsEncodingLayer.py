"""Learnable coordinate encoding layer with positional embeddings.

Implements learnable Fourier positional encodings with trainable frequencies
and phase shifts for encoding 2D coordinate inputs.
"""

from typing import Callable, Optional, Any, NamedTuple
import tensorflow as tf
import math

# Frequency scaling constants
MAX_FREQUENCY_POWER = 32


class Config(NamedTuple):
    """Configuration for coordinate encoding layer.

    Contains all configurable parameters for the coordinate encoding layer.
    - raw: Whether to concatenate raw input with encoded features
    - use_shifts: Whether to learn phase shifts for sinusoids
    - hidden_n: Multiplier for internal feature dimension
    - scaling: Frequency scaling method ('pow' or 'linear')
    - max_frequency: Maximum frequency value for scaling
    - use_low_bands: Whether to include low-frequency bands
    - use_high_bands: Whether to include high-frequency bands
    - final_dropout: Dropout rate applied to encoded output
    - bands_dropout: Whether to use per-band dropout strategy
    - shared_transformation: Whether to share weights across bands
    - N: Number of output channels/frequency bands
    - max_n: Internal maximum number of bands
    """

    raw: bool
    use_shifts: bool
    hidden_n: float
    scaling: str
    max_frequency: float
    use_low_bands: bool
    use_high_bands: bool
    final_dropout: float
    bands_dropout: bool
    shared_transformation: bool
    N: int
    max_n: int


class CoordsEncodingLayer(tf.keras.layers.Layer):
    """Learnable coordinate encoding with sinusoidal basis.

    Encodes 2D coordinates using learnable sinusoidal functions with
    trainable frequencies, phase shifts, and gating mechanisms.

    Attributes:
        config: Configuration object containing all layer parameters
        _bottleneck: Output projection layer
        _shifts: Learnable phase shifts for sinusoids
        _freq_deltas: Learnable frequency modulations
        _frequency: Learnable scaling factor for frequencies
        _fusion_w: Fusion weights for combining sine/cosine
        _fusion_b: Fusion bias terms
        _gates: Learnable gating for each band
        _dropout: Dropout function (either element-wise or band-wise)
    """

    def __init__(
        self,
        N: int,
        raw: bool = True,
        use_shifts: bool = False,
        hidden_n: float = 1.0,
        scaling: str = "pow",
        max_frequency: float = 1e4,
        use_low_bands: bool = True,
        use_high_bands: bool = True,
        final_dropout: float = 0.0,
        bands_dropout: bool = False,
        shared_transformation: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize coordinate encoding layer with learnable sinusoidal positional encodings.

        This layer encodes 2D coordinates using learnable sinusoidal basis functions with
        trainable frequencies, phase shifts, and gating mechanisms. The encoding helps neural
        networks learn position-dependent patterns.

        Args:
            N: Number of output channels/frequency bands.
               - Controls dimensionality of encoded output
               - Larger N means richer frequency coverage but more parameters
               - Internal dimension is computed as int(N * hidden_n)
               - Example: N=64 produces encoding with 64 frequency bands

            raw: If True, concatenate raw input with encoded features (default: True).
               - When True: output shape = (batch, points, N + 2) [includes raw coords]
               - When False: output shape = (batch, points, N) [only encoded]
               - Concatenating raw helps preserve fine-grained coordinate details
               - Typically kept True for better coordinate localization

            use_shifts: If True, learn phase shifts for sinusoids (default: False).
               - When True: adds trainable phase shifts to sin/cos inputs
               - When False: phase shifts are zero (standard positional encoding)
               - Phase shifts enable richer function representations
               - Increases model capacity by N additional parameters

            hidden_n: Multiplier for internal feature dimension (default: 1.0).
               - Internal bands = int(N * hidden_n)
               - Allows decoupling of output dimension from internal computation
               - hidden_n < 1.0: compression (fewer internal bands than output)
               - hidden_n > 1.0: expansion (more internal bands than output)
               - Example: N=64, hidden_n=2.0 → 128 internal bands projected to 64

            scaling: Frequency scaling method for band generation (default: 'pow').
               - 'pow': Exponential/logarithmic spacing
                 * Creates bands: [1, base, ..., base^(N-1)]
                 * Better frequency coverage for wide range
                 * Recommended for general use
               - 'linear': Linear spacing
                 * Creates bands: [1, 2, 3, ..., N]
                 * Simpler but less effective frequency coverage
               - Choice affects learned frequency modulation patterns

            max_frequency: Maximum frequency value (default: 1e4).
               - Scales the base frequency bands
               - Higher values enable encoding of high-frequency details
               - Lower values focus on low-frequency structure
               - Affects coefs property computation: coefs *= softplus(frequency)
               - Example: max_frequency=1000 captures finer coordinate details

            use_low_bands: Include low-frequency bands (default: True).
               - When True: includes bands like [1/freq, 1/(2*freq), ...]
               - Low frequencies capture global structure and coarse patterns
               - Typically kept True for capturing coordinate ranges
               - False: only high frequencies (fine details only)

            use_high_bands: Include high-frequency bands (default: True).
               - When True: includes bands like [freq, 2*freq, ...]
               - High frequencies capture fine-grained positional details
               - Typically kept True for coordinate precision
               - False: only low frequencies (coarse structure only)
               - Setting both False to False raises ValueError

            final_dropout: Dropout rate applied to encoded output (default: 0.0).
               - Range: [0.0, 1.0]
               - 0.0: No dropout
               - 0.5: Drop 50% of connections during training
               - Applied after fusion and gating
               - Reduces overfitting to specific frequency patterns

            bands_dropout: Dropout strategy (default: False).
               - When False: Standard element-wise dropout (random elements dropped)
               - When True: Per-band dropout (entire bands conditionally dropped)
                 * Bands with low gate values (less important) drop more frequently
                 * Encourages model to learn important frequency bands
                 * Requires higher final_dropout values for effect

            shared_transformation: Weight sharing across bands (default: False).
               - When False: Each band has unique fusion weights (W shape: N × fusion_dim)
               - When True: Single shared fusion weight for all bands (W shape: 1 × fusion_dim)
               - shared_transformation=True: Reduces parameters, assumes bands are similar
               - shared_transformation=False: More capacity, allows band-specific processing
               - Typically False for coordinate encoding (bands represent different scales)

            **kwargs: Additional Keras layer arguments
               - name: Layer name for tracking in model
               - trainable: Whether weights are updated during training
               - dtype: Data type for computations

        Example usage:
            # Standard coordinate encoding (recommended)
            encoder = CoordsEncodingLayer(N=64)
            # → outputs (batch, num_points, 66) with raw + encoded

            # Low-resource variant (fewer parameters)
            encoder = CoordsEncodingLayer(N=32, hidden_n=0.5, raw=False)
            # → outputs (batch, num_points, 32), internal uses 16 bands

            # High-capacity variant (more frequency coverage)
            encoder = CoordsEncodingLayer(
                N=128,
                use_shifts=True,
                max_frequency=1e5,
                bands_dropout=True,
                final_dropout=0.1
            )
            # → enhanced encoding with learned phase shifts and per-band dropout
        """
        kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"
        super().__init__(**kwargs)

        N = int(N * hidden_n)
        max_n = 1 + N // 2 if use_high_bands and use_low_bands else N

        # Create configuration object
        self.config = Config(
            raw=raw,
            use_shifts=use_shifts,
            hidden_n=hidden_n,
            scaling=scaling,
            use_low_bands=use_low_bands,
            use_high_bands=use_high_bands,
            final_dropout=final_dropout,
            bands_dropout=bands_dropout,
            shared_transformation=shared_transformation,
            N=N,
            max_n=max_n,
            max_frequency=max_frequency,
        )

        self._dropout: Callable = lambda x, **_: x
        if 0.0 < final_dropout:
            if bands_dropout:
                self._dropout = self._create_bands_dropout(final_dropout)
            else:
                dropout_layer = tf.keras.layers.Dropout(
                    final_dropout, name="final_dropout"
                )
                self._dropout = lambda x, **kwargs: dropout_layer(x, **kwargs)

    def build(self, input_shape: tuple) -> None:
        """Build fusion weights and gates.

        Args:
            input_shape: Input tensor shape (batch, points, coords)
        """
        assert 3 == len(input_shape), "Only rank 3 allowed"
        # Build bottleneck layer dynamically based on raw flag
        if self.config.raw:
            # When raw=True, bottleneck expects input of shape (batch, points, N + coord_dim)
            expected_input_dim = self.config.N + input_shape[-1]
        else:
            # When raw=False, bottleneck expects input of shape (batch, points, N)
            expected_input_dim = self.config.N

        self._bottleneck = tf.keras.layers.Dense(
            units=self.config.N,
            use_bias=False,
            activation=None,
            name="_bottleneck",
        )
        # Build bottleneck with expected input shape
        test_shape = list(input_shape)
        test_shape[-1] = expected_input_dim
        self._bottleneck.build(tuple(test_shape))

        # Create shifts weight
        self._shifts = self.add_weight(
            name="CEL_shifts",
            shape=(
                (1, 1, 1, self.config.N) if self.config.use_shifts else (1, 1, 1, 1)
            ),
            initializer=(
                tf.keras.initializers.RandomNormal()
                if self.config.use_shifts
                else tf.keras.initializers.Zeros()
            ),
            trainable=self.config.use_shifts,
        )

        # Create trainable weights
        self._freq_deltas = self.add_weight(
            name="CEL_freq_deltas",
            shape=(self.config.N,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )
        self._frequency = self.add_weight(
            name="CEL_frequency",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.config.max_frequency),
            trainable=True,
        )

        num_points = input_shape[1]
        num_fusion_params = 1 if self.config.shared_transformation else self.config.N
        fusion_dim = self._transform(tf.zeros([1, num_points, input_shape[-1]])).shape[
            -1
        ]

        self._fusion_w = self.add_weight(
            name="CEL_fusion_w",
            shape=[1, num_fusion_params, fusion_dim],
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )
        self._fusion_b = self.add_weight(
            name="CEL_fusion_b",
            shape=[1, num_fusion_params],
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )

        self._gates = self.add_weight(
            name="CEL_gates",
            shape=(1, 1, self.config.N),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def _fusion(self, x: tf.Tensor) -> tf.Tensor:
        """Fuse sinusoidal basis functions using learnable weights and bias.

        Combines the sine and cosine basis functions from the transform operation
        into a single representation using learnable fusion weights and biases.
        This is a learned linear combination of the basis functions.

        Args:
            x: Transformed tensor with sine/cosine values of shape
               (batch, num_points, num_bands, coord_dim * num_functions).
               Contains both sine and cosine transformed coordinates.

        Returns:
            Fused output tensor of shape (batch, num_points, num_bands) after
            weighted sum reduction along the last dimension.

        Mathematical operation:
            For each band i: result[i] = sum(x * W[i]) + B[i]
            where W are learnable fusion weights and B are learnable biases.
        """
        result = tf.reduce_sum(x * self._fusion_w, axis=-1) + self._fusion_b
        return result

    def _transform(self, coordinates: tf.Tensor) -> tf.Tensor:
        """Transform coordinates to sinusoidal basis functions.

        Applies learnable sine and cosine transformations to input coordinates
        using learnable frequency coefficients. This implements positional encoding
        with frequencies that can be learned during training.

        The transformation process:
        1. Expands coordinates to include frequency dimension
        2. Scales coordinates by learnable frequency coefficients
        3. Applies sine and cosine transformations
        4. Returns concatenated sine and cosine values

        Args:
            coordinates: Input coordinates of shape (batch, num_points, coord_dim)
                        where batch is the batch size, num_points is the number of
                        coordinate points, and coord_dim is the dimensionality
                        (typically 2 for 2D coordinates).

        Returns:
            Transformed tensor of shape (batch, num_points, num_bands, coord_dim * 2)
            containing both sine and cosine transformations for all frequency bands.
            The last dimension contains concatenated sine (first half) and cosine
            (second half) values for each coordinate dimension.

        Technical details:
            - Frequency coefficients are learnable via self.coefs
            - Phase shifts can optionally be applied via self.shifts
            - Multiplies by 2π before sine/cosine transformation (standard PE)
            - Output shape allows flexible band-wise processing
        """
        batch_size = tf.shape(coordinates)[0]
        num_points = coordinates.shape[1]
        coord_dim = coordinates.shape[2]
        num_bands = self.config.N
        transformed_data = []
        # (batch, num_points, coord_dim, num_bands)
        scaled_coords = (tf.expand_dims(coordinates, -1) * self.coefs) + self.shifts
        scaled_coords = scaled_coords * (2.0 * math.pi)
        scaled_coords = tf.transpose(scaled_coords, (0, 1, 3, 2))
        # (batch, num_points, num_bands, coord_dim, 1)
        scaled_coords = tf.expand_dims(scaled_coords, -1)
        for transform_func in [tf.sin, tf.cos]:
            transformed_func = transform_func(scaled_coords)
            transformed_data.append(transformed_func)

        result = tf.concat(transformed_data, axis=-1)
        tf.debugging.assert_equal(
            tf.shape(result)[:-1], (batch_size, num_points, num_bands, coord_dim)
        )
        return tf.reshape(
            result, (batch_size, num_points, num_bands, result.shape[-1] * coord_dim)
        )

    def call(
        self, coordinates: tf.Tensor, training: Optional[bool] = None
    ) -> tf.Tensor:
        """Encode coordinates using learnable sinusoidal functions.

        Args:
            coordinates: Input coordinates of shape (batch, num_points, coord_dim)
            training: Whether in training mode for dropout

        Returns:
            Encoded coordinates of shape (batch, num_points, num_bands) or (batch, num_points, num_bands+coord_dim) if raw=True
        """
        tf.debugging.assert_rank(
            coordinates, 3, "Input must be (batch, num_points, coord_dim)"
        )
        # coordinates is (batch, num_points, coord_dim)
        # output is (batch, num_points, num_bands)
        batch_size = tf.shape(coordinates)[0]
        num_points = coordinates.shape[1]
        num_bands = self.config.N

        # (batch, num_points, num_bands, coord_dim * num_functions)
        transformed = self._transform(coordinates)
        # (batch, num_points, num_bands)
        result = self._fusion(transformed) * self.gates
        tf.debugging.assert_equal(tf.shape(result), (batch_size, num_points, num_bands))
        result = self._dropout(result, training=training)

        if self.config.raw:
            # When raw=True, concatenate coordinates before bottleneck
            # Result has shape (batch, num_points, N), concatenate coords (batch, num_points, 2)
            result = tf.concat([coordinates, result], axis=-1)
        return self._bottleneck(result)

    @property
    def coefs(self) -> tf.Tensor:
        """Learnable frequency coefficients for sinusoidal encoding.

        Computes the final frequency coefficients used in coordinate transformation.
        The coefficients are computed as a modulation of base frequencies by learnable
        parameters, allowing the model to adapt the frequency bands during training.
        """
        # Compute base frequencies on the fly to avoid scope issues
        base_freq, freq_range = self._create_bands()

        coefficients = base_freq + tf.nn.tanh(self._freq_deltas) * freq_range
        frequency = tf.nn.softplus(self._frequency)
        return coefficients[None, None, None] * frequency[None, None, None]

    @property
    def shifts(self) -> tf.Tensor:
        """Learnable phase shifts for sinusoidal basis functions."""
        return self._shifts

    @property
    def gates(self) -> tf.Tensor:
        """Learnable gating tensor for frequency band importance weighting."""
        return tf.nn.tanh(self._gates)

    def _create_bands_dropout(self, max_rate: float) -> Callable:
        """Create a band-wise dropout function for selective band suppression.

        Creates a dropout function that applies different dropout rates to each
        frequency band based on the learned gate values. This allows the model
        to selectively suppress less important frequency bands during training.

        The dropout rate per band is computed as:
            dropout_rate[i] = (1 - normalized_gate[i]) * max_rate

        where normalized_gate values range from 0 to 1 after normalization.

        Args:
            max_rate: Maximum dropout rate (float between 0 and 1).
                    The actual dropout rate per band is scaled by the normalized
                    gate values, so max_rate defines the upper bound.

        Returns:
            A callable function with signature F(x: Tensor, training: Optional[bool]) -> Tensor
            that applies band-wise dropout during training and returns identity during inference.

        Implementation details:
            - Uses gate values to determine per-band dropout rates
            - Gates are normalized by the maximum gate value for stability
            - Applies stochastic dropout mask during training
            - Uses tf.stop_gradient to prevent gradient flow through the mask
            - Respects Keras training flag for proper training/inference behavior
        """

        def apply(x: tf.Tensor) -> tf.Tensor:
            """Apply per-band dropout to input tensor.

            Implements dropout where each frequency band can have a different dropout
            rate based on its learned gate value. Bands with low gate magnitudes (less
            important) have higher dropout rates, encouraging the model to learn which
            bands are necessary.

            Args:
                x: Input tensor with shape (batch, num_points, num_bands) containing
                   the fused encoding values before dropout.

            Returns:
                Tensor with per-band dropout applied. Same shape as input.
                Dropout is applied during training, identity is returned during inference.

            Implementation details:
                - Normalizes gate values to [0, 1] range for dropout rate computation
                - Computes per-band dropout probability as (1 - gate) * max_rate
                - Applies binomial dropout mask with per-band rates
                - Rescales by (1 - dropout_rate) for unbiased expectation
                - Uses tf.stop_gradient on the mask to prevent optimization issues
            """
            normed = tf.abs(self.gates)
            normed = tf.math.divide_no_nan(
                normed, tf.reduce_max(normed, axis=-1, keepdims=True)
            )
            normed = (1.0 - normed) * max_rate
            noise = tf.random.uniform(tf.shape(x), minval=0.0, maxval=1.0)

            mask = tf.cast(normed < noise, x.dtype) / (1.0 - normed)
            return x * tf.stop_gradient(mask)

        def function_wrapper(
            x: tf.Tensor, training: Optional[bool] = None
        ) -> tf.Tensor:
            # Use provided training flag if available, otherwise use Keras learning phase
            if training is None:
                training_mode = tf.keras.backend.learning_phase()
                training_mode = tf.cast(training_mode, tf.bool)
            else:
                training_mode = tf.cast(training, tf.bool)
            return tf.cond(training_mode, lambda: apply(x), lambda: tf.identity(x))

        return function_wrapper

    def _to_bands(self, freq):
        bands = []
        if self.config.use_low_bands:
            bands.append(1.0 / freq[::-1])
        if self.config.use_high_bands:
            bands.append(freq)
        base_freq = tf.concat(
            [(x[1:] + x[:-1])[: self.config.max_n] / 2.0 for x in bands],
            axis=-1,
        )
        freq_range = tf.concat(
            [(x[1:] - x[:-1])[: self.config.max_n] / 2.0 for x in bands],
            axis=-1,
        )
        return base_freq, freq_range

    def _create_bands(self) -> tf.Tensor:
        """Create frequency bands for positional encoding.

        Generates base frequency bands using either exponential (power) or linear
        scaling. These frequencies form the foundation for the sinusoidal positional
        encoding. The generated bands are then used to compute learnable frequency
        coefficients via tanh modulation in the coefs property.

        Returns:
            Frequency band values as a tensor of shape (num_bands,).

            For 'pow' scaling:
                - Base = 2^(1/num_bands)
                - Returns [1, base, base^2, ..., base^(num_bands-1)]
                - Constrained by MAX_FREQUENCY_POWER for numerical stability

            For 'linear' scaling:
                - Returns linearly spaced values from 1.0 to num_bands

        Raises:
            ValueError: If scaling method is neither 'pow' nor 'linear'

        Technical notes:
            - These bands are normalized by 2.0 in __init__
            - Further modulated with learnable deltas and frequency scaling
            - Exponential scaling (pow) is preferred as it provides better coverage
              of frequency space in positional encodings
        """
        num_bands = self.config.max_n
        if self.config.scaling == "pow":
            # 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, ... 1 / 2^num_bands
            max_frequency = 2.0 ** min((num_bands, MAX_FREQUENCY_POWER))
            base = math.pow(max_frequency, 1.0 / float(num_bands))
            freq = tf.pow(base, tf.cast(tf.range(num_bands), tf.float32))
            return self._to_bands(freq)

        if self.config.scaling == "linear":
            freq = tf.linspace(1.0, float(num_bands), num_bands)
            return self._to_bands(freq)

        raise ValueError(f"Unknown scaling: {self.config.scaling}")
