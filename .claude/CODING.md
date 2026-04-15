# Coding Standards

## DON'T

### Suppressions & Quality
- Suppress with `# type: ignore`, `# noqa` — fix root causes instead
  - Exception: `# type: ignore[unused-argument]` for required protocol params (explain why)
  - Always be specific: `# type: ignore[assignment]` (never blanket)
- Cryptic names (`x`, `y`) — use `embedding_size`, `trajectory_indices`
- Magic numbers — extract to constants: `DEFAULT_TIME_DELTA = 0.01`
- Public methods without type hints & docstrings
- Bare exceptions (`except:`, `except Exception:`) — catch specific types
  - Wrong: `except Exception: pass`
  - Right: `except FileNotFoundError: ...` or `except ValueError as e: logger.error(...)`
- Try-except blocks for IO operations — never use try for IO, fail fast
  - Wrong: Never wrap IO with try-except, even to log and re-raise
    ```python
    # BAD: try-except wrapper adds no value, swallows stack trace
    try:
        with open(cache_file, "wb") as f:
            f.write(content)
    except IOError as e:
        logger.error(f"Failed to write: {e}")
        raise

    # BAD: Catching and re-wrapping masks original error
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with h5py.File(filepath, "w") as f:
            f.create_dataset(...)
    except (IOError, OSError) as e:
        raise IOError(f"Error writing HDF5 file {filepath}: {e}") from e
    ```
  - Right: Direct IO without try-except; let exceptions propagate as-is
    ```python
    # GOOD: IO operations fail immediately and visibly
    with open(cache_file, "wb") as f:
        f.write(content)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset(...)
    ```
  - Rationale: IO errors are environment problems (disk full, permissions, etc.); they must fail fast and propagate with full stack trace. Try-except masks the real issue and wastes debugging time.
- Silent IO errors — distinguish expected (missing file) vs unexpected (error)
  - Wrong: `try: data = load(path) except: data = None`
  - Right:
    ```python
    try:
        data = load(path)
    except FileNotFoundError:
        logger.warning(f"Missing cache: {path}")
        return None  # Expected
    except IOError as e:
        logger.error(f"IO error reading {path}: {e}")
        raise  # Unexpected
- Ignore tensor shape mismatches — assert explicitly
  - Wrong: `embeddings = encode(inputs)  # expect (batch, 256)`
  - Right: `embeddings = encode(inputs); tf.debugging.assert_rank(embeddings, 2, message="Expected (batch, dim)"); assert embeddings.shape[1] == 256`
- Scattered config — centralize in `__init__`, merge kwargs: `{**self._defaults, **kwargs}`
- Deep inheritance — use composition instead
- `print()` in production — use structured logging
- Files over 500 lines (aim 150-350)
- `else:` after `if` returns — use early exits to reduce nesting
  ```python
  # BAD: unnecessary else after return
  if condition:
      return result
  else:
      return default

  # GOOD: early return eliminates else
  if condition:
      return result
  return default

  # BAD: elif chains when first condition returns
  if x == "a":
      return func_a()
  elif x == "b":
      return func_b()
  else:
      raise ValueError(...)

  # GOOD: replace elif/else with subsequent if/raise
  if x == "a":
      return func_a()
  if x == "b":
      return func_b()
  raise ValueError(...)
  ```
- Stateless classes — classes need meaningful state; use functions otherwise

### Keras Models
- **NPZ format only** — Do NOT implement `get_config()` or `from_config()`
  - Models use NpzModelMixin for save/load (NPZ format)
  - Wrong: `get_config()`, `from_config()`, keras JSON config
  - Right: Save/load via NpzModelMixin interface (weight-based, not config-based)
  - Rationale: Architecture is complex with custom layers; NPZ stores weights directly and is more reliable than config-based serialization

- Never directly set `.built = True` — use `super().build()` instead
  - Wrong: `self.built = True` (bypasses Keras' internal state tracking)
  - Right: `super().build(input_shape)` or `self.built = True` only after weights are created
  - Rationale: Direct assignment prevents Keras from properly tracking model state and weight initialization,
    leading to serialization warnings ("model has not yet been built") and incomplete state tracking.
    Always invoke parent build() to maintain Keras' internal consistency.

### **Weight and Variable Creation: build() vs __init__()**
**ABSOLUTE RULE**: All variables, weights, and sublayers must be created in `build()`, never in `__init__()`

#### **WHY THIS MATTERS**
1. **Serialization Safety**: Models created in `__init__` can't be serialized properly because Keras tracks layer state through `build()`
2. **Input Shape Flexibility**: `build()` receives input shape, enabling adaptive weight creation
3. **Consistent State**: Keras' internal state tracking depends on proper build sequence
4. **Export Compatibility**: Export scripts require weights to be created in `build()` for consistent test data generation
5. **Memory Efficiency**: Only create weights when input shapes are known

#### **WRONG: Creating weights in __init__()**
```python
# ❌ NEVER do this - creates weights without input shape knowledge
class BadLayer(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.units = units
        # WRONG: Creating weights in constructor
        self.dense = tf.keras.layers.Dense(units)  # ❌ No input_shape!
        self.dropout = tf.keras.layers.Dropout(0.1)  # ❌ Will fail to build

    def call(self, inputs):
        x = self.dense(inputs)  # ❌ Random weights, inconsistent shape
        x = self.dropout(x)
        return x
```

#### **RIGHT: All weights created in build()**
```python
# ✅ CORRECT - weights created with input shape information
class GoodLayer(tf.keras.layers.Layer):
    def __init__(self, units: int):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        # ✅ Create weights AFTER input shape is known
        self.dense = tf.keras.layers.Dense(self.units, name="dense")  # ✅ Can infer input_shape
        self.dropout = tf.keras.layers.Dropout(0.1, name="dropout")  # ✅ Now ready to build

        # ✅ Mark as built after all sublayers are created
        super().build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)  # ✅ Properly initialized weights
        x = self.dropout(x)
        return x
```

#### **SUMMARY: KEY PRINCIPLES**
1. **`__init__` is for configuration** - Store hyperparameters, validate inputs
2. **`build()` is for weight creation** - Create all weights and sublayers here
3. **Never access weights in `__init__`** - They don't exist yet
4. **Always call `super().build(input_shape)`** - Maintain Keras state tracking
5. **Explicit names for all sublayers** - Critical for serialization consistency

- **ALWAYS set explicit names** on every tf.keras.layers.Layer and tf.keras.Model
  - Wrong: `L.Dense(64)` or `TimeEncodingLayer()` (relies on Keras auto-naming)
  - Right: `L.Dense(64, name="output_dense")` or `TimeEncodingLayer(name="time_encoder")`
  - For loops: use f-strings with loop variable: `L.Dense(64, name=f"dense_{i}")`
  - **ALWAYS validate names in kwargs before passing to layer constructors**
    - Right: `def __init__(self, **kwargs): kwargs["name"] = kwargs.get("name") or f"{self.__class__.__name__}_layer"`
  - Rationale: Without explicit names, Keras auto-generates names that:
    1. Differ between fresh model instances (causes save/load shape mismatches)
    2. Can create unintended duplicates when layers are created without unique names
    3. Make debugging harder (generic `"dense"` vs `"output_projection"`)
  - **Critical for save/load**: Two identical models initialized fresh will have different layer names without explicit naming, causing load() to fail with shape mismatches
  ```python
  # BAD: Keras auto-naming causes inconsistent names across instances
  def __init__(self):
      self.mlp = sMLP(sizes=[256, 128])  # Gets auto-name like "s_mlp" or "s_mlp_1"
      self.mixer = LinearAttentionMixer()  # Gets auto-name like "linear_attention_mixer"

  # GOOD: Explicit names ensure consistency
  def __init__(self):
      self.mlp = sMLP(sizes=[256, 128], name="feature_mlp")
      self.mixer = LinearAttentionMixer(name="feature_mixer")

  # GOOD: F-strings in loops for unique names
  def __init__(self):
      self.blocks = [
          TransformerBlock(name=f"transformer_block_{i}")
          for i in range(3)
      ]

  # GOOD: Fallback for optional names
  def __init__(self, name: Optional[str] = None):
      self.layer = L.Dense(64, name=name or "default_dense")
  ```

### Architecture
- `C` prefix on classes: `CBaseDataSampler` → `BaseDataSampler`
- Multiple classes per file (except `Utils.py` & `Constants.py`)
  - Filename matches classname: `DataSampler` in `DataSampler.py`
  - Tightly coupled helpers can coexist; independent code gets own file
- `Union[TypeA, TypeB]` — create Protocol/Interface instead
  - Pattern: `{Name}Interface`, factory `create_{name}()`, naming `{Name}Factory`
- `sys.path` hacks — use `Path` for file operations
- Unused parameters — keep for API compat; use `# type: ignore[unused-argument]` + comment
- Unused returns — use `_, x = ...` instead of dropping them

### **CRITICAL: Never Access Private Fields of Private Fields**
**ABSOLUTE PROHIBITION**: Never access `self._a._b` (private fields of private fields)

**WHY**: This violates encapsulation and creates brittle, unmaintainable code
- `self._model_wrapper._model` ❌
- `self._model_wrapper._predictor` ❌
- `self._teacher._model_wrapper` ❌

**SOLUTIONS**:
1. **Add public methods** to expose necessary functionality
2. **Use composition properly** - design classes with clear public APIs
3. **Extract interfaces** - define protocols for expected behavior

**GOOD**:
```python
# ModelWrapper provides public API
class ModelWrapper:
    def get_model_summary(self) -> str:  # Public method
        return self._model.summary()

    def get_predictor(self) -> PredictorBlock:  # Public access
        return self._predictor
```

**BAD**:
```python
# Private field access violation
class ModelTrainer:
    def something(self):
        model = self._model_wrapper._model  # ❌ Never do this!
        predictor = self._model_wrapper._predictor  # ❌ Never do this!
```

## DO

### Core
- **Type hints & docstrings**: All public methods (Args/Returns/Raises + examples)
  ```python
  def predict(self, face_mesh: np.ndarray, eyes: np.ndarray) -> Optional[np.ndarray]:
      """Predict gaze direction from face and eye data.

      Args:
          face_mesh: Face landmark coordinates, shape (N, 3).
          eyes: Eye region crops, shape (batch, height, width, 3).

      Returns:
          Predicted gaze vectors with shape (batch, 2), or None if prediction fails.

      Raises:
          ValueError: If eyes shape is invalid.

      Example:
          >>> gaze = predictor.predict(face_mesh, eyes)
          >>> assert gaze.shape == (32, 2)
      """
  ```
- **Input validation**: Fail-fast in `__init__`: `if x <= 0: raise ValueError(...)`
  ```python
  def __init__(self, learning_rate: float, batch_size: int):
      if learning_rate <= 0:
          raise ValueError(f"learning_rate must be positive, got {learning_rate}")
      if batch_size < 1:
          raise ValueError(f"batch_size must be >= 1, got {batch_size}")
      self._lr = learning_rate
      self._batch_size = batch_size
  ```
- **Composition**: Swap implementations easily
- **@property**: Immutable computed values
- **@lru_cache(None)**: Expensive pure functions
- **@tf.function**: TensorFlow critical paths
- **Return handling**: `None` for runtime failures; raise for construction errors
  ```python
  # Constructor: fail hard on bad config
  def __init__(self, path: str):
      if not Path(path).exists():
          raise ValueError(f"Model path does not exist: {path}")

  # Method: return None for expected runtime failures (retry-able)
  def load_sample(self, sample_id: int) -> Optional[np.ndarray]:
      try:
          return self._storage[sample_id]
      except (KeyError, IOError):
          return None  # Caller handles retry
  ```
- **NamedTuple**: Structured returns (`result.latent` > `result[0]`)
  ```python
  from typing import NamedTuple

  class PredictionResult(NamedTuple):
      gaze: np.ndarray  # shape (batch, 2)
      confidence: np.ndarray  # shape (batch,)

  def predict(self, face: np.ndarray) -> PredictionResult:
      gaze = self._model(face)
      conf = self._confidence(face)
      return PredictionResult(gaze=gaze, confidence=conf)

  # Usage: result.gaze and result.confidence (clear, self-documenting)
  ```
- **Config pattern**: `.get()` for optional params, merge defaults
  ```python
  def __init__(self, defaults: Optional[Dict] = None):
      self._defaults = defaults or {}

  def train(self, **kwargs):
      config = {**self._defaults, **kwargs}  # kwargs override defaults
      return self._run_training(
          learning_rate=config.get("learning_rate", 0.001),
          epochs=config.get("epochs", 10)
      )

  # Usage: trainer.train(learning_rate=0.01)  # overrides default
  ```
- **Tensor shapes**: Document & assert (`tf.debugging.assert_rank(inputs, 3)`)
  ```python
  def encode(self, face_mesh: tf.Tensor) -> tf.Tensor:
      """Encode face mesh to embeddings.

      Args:
          face_mesh: Shape (batch, 468, 3) — 468 face landmarks.

      Returns:
          Embeddings with shape (batch, 256).
      """
      tf.debugging.assert_rank(face_mesh, 3, "Expected rank-3 tensor")
      assert face_mesh.shape[1] == 468, f"Expected 468 landmarks, got {face_mesh.shape[1]}"
      return self._encoder(face_mesh)
  ```
- **Assertions**: Internal consistency only; include values in messages
  ```python
  # Good: Assert invariants with context
  assert len(embeddings) == len(labels), f"Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels"

  # Bad: Silent assertion or validation of user input
  assert x > 0  # No context; should raise ValueError instead
  ```

### Patterns

**Configuration**: Merge defaults + runtime kwargs
```python
def __init__(self, defaults: Optional[Dict] = None):
    self._defaults = defaults or {}
def process(self, **kwargs): return self._impl({**self._defaults, **kwargs})
```

**Dispatch**: Strategy dict replaces if/elif
```python
strategies = {"uniform": self._uniform, "time": self._uniformTime}
return strategies[key](**kwargs)
```

**Multiple APIs**: `sample()`, `sampleById()`, `sampleByIds()` for different use cases
```python
def sample(self) -> np.ndarray: ...  # Random sample
def sample_by_id(self, sample_id: int) -> Optional[np.ndarray]: ...  # Single ID
def sample_by_ids(self, sample_ids: List[int]) -> np.ndarray: ...  # Multiple IDs
```

**Failure**: Return `None` for runtime; raise for construction; callers handle retries
```python
# Construction: raise immediately
class DataLoader:
    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

# Runtime: return None for expected failures
def load_next(self) -> Optional[np.ndarray]:
    try:
        return next(self._iterator)
    except (StopIteration, IOError):
        return None  # Caller checks for None and retries
```

**Validation**: Iterate, convert, validate per level; include full path in errors
```python
def load_config(config_path: str) -> Dict:
    """Validate hierarchical config with clear error paths."""
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file missing: {config_path}")

    # Validate each level
    if "model" not in config:
        raise ValueError(f"Missing 'model' key in {config_path}")
    if "learning_rate" not in config["model"]:
        raise ValueError(f"Missing 'model.learning_rate' in {config_path}")

    # Convert and validate values
    try:
        lr = float(config["model"]["learning_rate"])
        if lr <= 0:
            raise ValueError(f"model.learning_rate must be positive, got {lr}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid model.learning_rate in {config_path}: {e}")

    return config
```

### Organization
- **Imports**: stdlib → third-party → local (explicit; use `Path`)
- **Tests**: Classes (see @TESTING.md); one file per class
- **Constants**: `Utils.py` or `Constants.py`: `FACE_MESH_KEYPOINTS = 468`
- **Layout**: `Core/` (data), `NN/` (models), `App/` (UI), `scripts/` (utils), `tests/` (fixtures)
- **Model/layer separation**: `tf.keras.Model` classes go in `NN/models/`, `tf.keras.layers.Layer` classes go in `NN/layers/`
- **Rule**: Single responsibility; no circular deps

## Checklist
✅ One class/file | ✅ Public methods typed | ✅ No magic numbers | ✅ Input validation | ✅ No bare except | ✅ Shapes asserted | ✅ Config centralized | ✅ Under 500 lines | ✅ Tests follow @TESTING.md | ✅ No private field access violations (no `self._a._b`)

## Architecture Patterns (NEW)

### **Dependency Injection (REQUIRED)**
**Problem**: ModelTrainer inherits from ModelWrapper causing tight coupling and recursive memory issues.

**Solution**: Use dependency injection instead of inheritance.

```python
# DON'T: Inheritance coupling
class ModelTrainer(ModelWrapper):  # ❌
    def __init__(self, timesteps, stats, ...):
        super().__init__(timesteps, stats, ...)  # Tight coupling
        self._teacher = ModelTrainer(...)  # Recursive creation!

# DO: Dependency injection
class ModelTrainer:
    def __init__(self, model_wrapper: ModelWrapper, teacher_model: Optional[tf.keras.Model] = None, ...):
        self._model_wrapper = model_wrapper  # Composition over inheritance
        self._teacher_model = teacher_model  # Inject teacher, no recursive creation
```

### **Loss Calculation Standardization**
**Problem**: Inconsistent latent key handling in loss calculation.

**Solution**: Always use `Core.losses.calculate_losses` with standardized pattern.

```python
# MANDATORY: Always use this exact pattern
def compute_losses_standardized(self, predictions, y_validated, scale=1.0):
    # Make a copy to avoid modifying original input (required by @tf.function)
    y_validated = dict(y_validated)

    # MANDATORY: Add latent keys to both predictions and y_validated
    if "latent_intermediate" in predictions:
        y_validated["latent_intermediate"] = teacher_intermediate_latents
    if "latent_final" in predictions:
        y_validated["latent_final"] = teacher_final_latents

    # MANDATORY: Always use Core.losses.calculate_losses
    computed_losses = calculate_losses(predictions, y_validated, training=True)

    # MANDATORY: Apply scale to latent losses only
    for key in list(computed_losses.keys()):
        if key.startswith("latent_") and scale != 1.0:
            computed_losses[key] *= scale

    return computed_losses
```

### **Test Performance Optimization**
**Problem**: Tests create full 600-800MB models just for parameter validation.

**Solution**: Use proxy functions for architectural validation.

```python
# DON'T: Full model instantiation for validation
def test_trainable_variables_count():
    model = ModelWrapper(...)  # 600MB instantiation!
    variables = model.trainable_variables
    assert len(variables) == expected_count

# DO: Proxy functions for fast validation
def test_layer_sizes_architecture():
    sizes = ModelWrapperProxies.validate_layer_sizes(...)
    assert sizes['expected_latent_shape'] == (timesteps, latent_size)
```

### **Teacher-Student Architecture Separation**
**Problem**: Single ModelTrainer class handles both teacher and student responsibilities.

**Solution**: Split into separate classes with clear responsibilities.

```python
# Clear separation of concerns
class ModelStudentTrainer:
    """Handles student model training with teacher guidance."""
    - Student model training logic
    - Multi-level loss computation with teacher supervision
    - Gradient computation and optimization

class ModelTrainer:  # Factory/orchestrator
    """Creates appropriate trainer based on configuration."""
    - Factory pattern for teacher-student coordination
```
