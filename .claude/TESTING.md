# Testing Guide

## Quick Start: Running Tests

To run all tests:
```bash
make test
```

This will activate the conda environment and run all tests via pytest.

## DON'T

- DON'T test/access private fields/methods
  - Example: `layer._weights`, `model._internal_state`
  - Why: Tests should verify public contracts, not internal implementation

- DON'T test Exceptions
  - Wrong: `try: ... except ValueError: ...` in tests — testing the exception type itself
  - Right: Test the behavior/side-effects that result from invalid input
  ```python
  # Wrong
  def test_invalid_input():
      try:
          model.predict(np.zeros((2, 99)))  # wrong shape
      except ValueError:
          pass  # Just checking if exception is raised

  # Right
  def test_invalid_shape_returns_none():
      result = model.predict(np.zeros((2, 99)))  # wrong shape
      assert result is None, "Should return None for invalid input"
  ```
  - Why: Exception handling is implementation detail; test behavior instead

- DON'T put code to run test as main
  - Example: `if __name__ == "__main__": test_something()`
  - Why: Use pytest runner; ad-hoc test runners hide test structure

- DON'T Add parent directory to path for imports
  - Wrong: `sys.path.insert(0, str(Path(__file__).parent.parent))`
    ```python
    # In test file — DON'T DO THIS
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Core import DataSampler
    ```
  - Right: Use pytest configuration
    ```ini
    # In setup.cfg or pytest.ini
    [tool:pytest]
    pythonpath = .
    ```
    Then import normally: `from Core import DataSampler`
  - Why: Configure import paths via pytest/setup.cfg; don't hardcode in tests

- DON'T test object creation alone
  - Example: `wrapper = Wrapper(...); assert wrapper is not None`
  - Why: Object creation is trivial; test what the object actually DOES

- DON'T create one-line helper methods that add no value
  - Wrong: `def _get_weights(self, wrapper): return [emb.get_weights() for emb in wrapper._embedding_block._embeddings.values()]`
  - Right: Inline 1-2 line operations directly in tests; create helpers only if they save 3+ lines
  ```python
  # Wrong: Helper that adds no clarity
  def _extract_embedding(sampler): return sampler.get_embedding()
  def test_something(self):
      emb = self._extract_embedding(sampler)

  # Right: Inline trivial operations
  def test_sampling_produces_embeddings(self):
      emb = sampler.get_embedding()
      assert emb.shape == (256,)

  # Right: Extract complex logic (saves 3+ lines)
  def _create_valid_dataset(self):
      """Create dataset with consistent shapes and valid ids."""
      ds = Dataset(...)
      ds.add(...)
      ds.validate(...)
      return ds

  def test_dataset_operations(self):
      ds = self._create_valid_dataset()  # Worth extracting
  ```
  - Why: Adds indirection without reducing duplication or improving readability; inline the code in tests
  - Exception: Multi-line helpers that genuinely reduce duplication (3+ lines saved)

- DON'T put multiple test classes in one file
  - Example: `test_data.py` with both `TestDataLoading` and `TestDataValidation` classes
  - Why: Each class should be in its own file for clarity and organization
  - Solution: Split into `test_data_loading.py` and `test_data_validation.py`

- DON'T test properties

- NEVER assert logs
  - Wrong: `assert "Initialized mixer: dense" in caplog.text`
  - Why: Logging is an implementation detail; log messages can change without affecting behavior
  - Right: Test actual behavior/side-effects instead (e.g., verify layer is built, weights are initialized)

- NEVER use print statements in tests
  - Wrong: `print(f"Performance: {result:.3f}s")`
  - Why: Print statements are for debugging, not test validation; they clutter test output and provide no value
  - Right: Use assertions to verify expected behavior
  ```python
  # Wrong: Printing for debugging
  def test_performance_regression(self):
      result = compute_performance()
      print(f"Performance: {result:.3f}s")  # Debug statement in test
      assert result < 1.0

  # Right: Test actual performance bounds
  def test_performance_regression(self):
      result = compute_performance()
      assert result < 1.0, f"Performance {result:.3f}s exceeded threshold"
  ```

- NEVER use logging in tests
  - Wrong: `logger.info("Test started")`, `logging.debug("Processing data")`
  - Why: Logging is implementation detail; tests should be self-contained and validate behavior
  - Right: Use assertions to verify outcomes, not logs to track process
  ```python
  # Wrong: Logging test progress
  def test_data_processing(self):
      logger.info("Starting data processing")
      data = load_data()
      logger.debug(f"Loaded {len(data)} items")
      result = process(data)
      assert len(result) > 0

  # Right: Direct validation
  def test_data_processing(self):
      data = load_data()
      result = process(data)
      assert len(result) > 0, "Data processing should produce results"
  ```

## Examples of Print/Log Usage in Tests (DON'T DO THESE)

Found in the test suite:

**Performance Debug Prints:**
```python
# In tests/model_wrapper/test_model_wrapper_save_load_optimized.py:107
print("✅ Proxy save/load validation completed in <0.1s (was 47.75s)")
# In tests/model_wrapper/test_model_wrapper_proxy_validation.py:269-272
print("Performance comparison:")
print(f"  Proxy validation: {proxy_time:.4f}s")
print(f"  Mock model: {mock_time:.4f}s")
print(f"  Speed improvement: {mock_time / max(proxy_time, 0.001):.1f}x")
```

**Test Status Prints:**
```python
# In tests/model_wrapper/test_model_wrapper_save_load_optimized.py:141,175,208,247,349
print("✅ Proxy save/load validation passed")
print(f"✅ Tested {len(configs)} configurations successfully")
print("✅ File structure validation tests passed")
print("✅ Proxy performance benchmarks completed")
print("✅ Backward compatibility test passed (optimized)")
```

**Mock Debug Prints:**
```python
# In tests/fixtures/model_fixtures.py:118,123
print(f"Mock save to {filepath}")
print(f"Mock load from {filepath}")
```

**Test Categorization Prints:**
```python
# In tests/conftest.py:94-105
print("\n📊 Test Performance Categorization:")
print(f"   Total tests: {total_tests}")
print(f"   Fast tests: {fast_tests}")
print(f"   Medium tests: {medium_tests}")
print(f"   Slow tests: {slow_tests}")
print(f"   Memory tests: {memory_tests}")
print(f"   Integration tests: {integration_tests}")
```

- NEVER use is_finite checks
  - Wrong: `assert tf.reduce_all(tf.math.is_finite(result))`
  - Why: Checking for finite values is an implementation detail; test actual results/behavior instead
  - Right: Compare against expected values or verify meaningful properties
  ```python
  # Wrong: Just checking values aren't NaN/inf
  result = model(inputs)
  assert tf.reduce_all(tf.math.is_finite(result))

  # Right: Check actual output values or shapes
  result = model(inputs)
  assert result.shape == expected_shape
  assert tf.reduce_max(tf.abs(result)) < 100.0  # Reasonable bounds
  ```
  - Why: Finite value checks hide what the test actually validates; test real outputs, not meta-properties

- NEVER filter/suppress warnings in pytest.ini or setup.cfg
  - Wrong: Adding `filterwarnings = ignore:.*:UserWarning` to suppress warnings
  - Why: Warnings indicate real problems in code; filtering them hides issues from developers
  - Right: Fix the root cause of warnings in the code itself
  ```python
  # Wrong: Suppressing the warning
  # In setup.cfg:
  # filterwarnings = ignore:Layer.*has unbuilt state:UserWarning

  # Right: Add proper build() method to the layer
  class MyLayer(tf.keras.layers.Layer):
      def __init__(self, ...):
          super().__init__(...)
          self._sublayer = SomeLayer(...)

      def build(self, input_shape):
          """Build all sub-layers with proper input shapes."""
          # Build sub-layers with their expected input shapes
          self._sublayer.build(expected_shape)
          super().build(input_shape)

      def call(self, x, **kwargs):
          return self._sublayer(x, **kwargs)
  ```
  - Why: Proper layer building ensures Keras can trace and optimize layers; it prevents real runtime issues

- ALWAYS use bounds for diff comparisons (< 1e-5 for tight tolerance)
  - Use when comparing predictions before/after save/load, or comparing outputs from equivalent models
  - Pattern: `max_diff = tf.reduce_max(tf.abs(output_a - output_b)).numpy()`
  - Tolerance target: `< 1e-5` for identical operations (ideal)
  ```python
  # Right: Compare predictions with tight tolerance
  result_before = model(inputs, training=False)
  model.save(path)
  model_loaded = Model(...)
  model_loaded.load(path)
  result_after = model_loaded(inputs, training=False)

  max_diff = tf.reduce_max(tf.abs(result_after - result_before)).numpy()
  print(f"Max difference: {max_diff:.2e} (target: < 1e-5)")
  assert max_diff < 1e-5, f"Predictions diverged by {max_diff:.2e}"
  ```
  - Why: Bounds validate numeric stability; `< 1e-5` is tight enough to catch regressions but loose enough for float precision

## DO

- use classes for test cases
  - Organize tests logically with TestClassName convention
  ```python
  # Good: Grouped by behavior
  class TestDataLoaderInitialization:
      def test_raises_on_missing_path(self):
          with pytest.raises(FileNotFoundError):
              DataLoader("/nonexistent/path")

      def test_accepts_valid_path(self):
          loader = DataLoader(valid_path)
          assert loader is not None

  class TestDataLoaderSampling:
      def test_returns_valid_shape(self):
          loader = DataLoader(valid_path)
          sample = loader.sample()
          assert sample.shape == (height, width, 3)

      def test_sequential_sampling_deterministic(self):
          ...
  ```

- test actual results
  - Call methods and verify outputs, not just object existence
  - Right: `result = model(inputs); assert result.shape == (batch, 2)`
  - Wrong: `model = Model(...); assert model is not None`
  ```python
  # Wrong: Just testing creation
  def test_model_creation():
      model = GazePredictor(...)
      assert model is not None

  # Right: Test actual behavior
  def test_model_produces_valid_output():
      model = GazePredictor(input_size=256)
      output = model(np.random.randn(32, 256))
      assert output.shape == (32, 2), f"Expected (32, 2), got {output.shape}"
  ```

- one file = one class (STRICT)
  - Filename must match class name: `test_feature_name.py` → `class TestFeatureName`
  - Pattern: `test_{component}_{aspect}.py` for clarity (e.g., `test_sample_filter_initialization.py`)
  - **VIOLATION**: Multiple test classes in one file
    - Wrong: `test_user.py` with both `TestUserInit` and `TestUserValidation`
    - Right: `test_user_initialization.py` + `test_user_validation.py`
  - Keeps tests organized, focused, and easy to locate

- **NEVER access private fields of private fields** (CRITICAL)
  - **ABSOLUTELY FORBIDDEN**: `obj._a._b`, `self._model_wrapper._model`, `self._trainer._model_wrapper._predictor`
  - Why: Violates encapsulation, creates brittle tests that break with internal implementation changes
  - Right: Test public APIs only; if internal access is needed, add public methods to the class
  ```python
  # Wrong: Testing private implementation details
  def test_internal_state(self):
      wrapper = ModelWrapper(...)
      model = wrapper._model  # Bad enough
      weights = model._weights  # FORBIDDEN - private of private!

  # Right: Test public behavior or request public API
  def test_model_output_shape(self):
      wrapper = ModelWrapper(...)
      result = wrapper.call(inputs)  # Test public interface
      assert result.shape == expected_shape
  ```

- all fixtures in tests/fixtures/
  - Share fixtures in `tests/fixtures/embedding_fixtures.py`
  - Load via `conftest.py` for automatic discovery

- assert's always with a clear msg
  ```python
  # Wrong: Silent assertion
  assert result.shape == (batch_size, 256)

  # Right: Include context
  assert result.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {result.shape}"

  # Or use pytest with assertion details (implicit message)
  assert result.shape[0] == batch_size
  ```
- organize test files into subject-based subdirectories (STRICT)
  - **Rule**: Group related test files into subdirectories by component/subject
  - **Pattern**: `tests/{subject}/test_{aspect}.py`
  - **Wrong**: Root-level files like `tests/test_embedding_block.py`, `tests/test_convpe_initialization.py`
  - **Right**: Organized in subdirectories: `tests/layers/test_embedding_block.py`, `tests/layers/test_convpe_initialization.py`
  - **Examples**:
    - `tests/test_filtered_dataset_edge_cases.py` → `tests/filtered_dataset/test_edge_cases.py`
    - `tests/test_model_wrapper_save_load.py` → `tests/model_wrapper/test_model_wrapper_save_load.py`
    - `tests/test_embedding_block.py` → `tests/layers/test_embedding_block.py`
  - **Grouping strategy**:
    - **Principle**: Consider the subject/domain of each test class and group by logical component
    - **Examples**:
      - Data/sampling components: `tests/base_data_sampler/`, `tests/filtered_dataset/`, `tests/sample_filter/`, `tests/inpainting_sampler/`
      - Neural network layers: `tests/layers/` (for all layer tests)
      - Model components: `tests/model_wrapper/`, `tests/model_trainer/` (separate by model class/responsibility)
    - **Decision guide**: If tests focus on a specific class (e.g., ModelTrainer, ModelWrapper), create a directory for that class
  - **Why**: Keeps test directory organized by subject matter; makes it easy to find and maintain all tests for a specific component

## NN Instantiation: Avoid When Possible, Reuse When Needed

**Key Principle**: Only instantiate real NN models when necessary to validate behavior that cannot be tested with mocks or fixtures.

### When NOT to Instantiate NN Models

- ❌ **Configuration/parameter validation**: Use mocks or fixtures
  ```python
  # Wrong: Instantiate full model to test parameter storage
  def test_model_stores_scale_mult():
      model = ModelTeacher(scale_mult=1.5)
      assert model._scale_mult == 1.5

  # Better: Mock the object
  class MockModel:
      def __init__(self, scale_mult):
          self._scale_mult = scale_mult

  model = MockModel(scale_mult=1.5)
  assert model._scale_mult == 1.5
  ```

- ❌ **Attribute/property storage**: Test logic, not object creation
  ```python
  # Wrong: Just checking object exists
  def test_model_creation():
      model = ModelTrainer(...)
      assert model is not None

  # Right: Test what the model DOES
  def test_model_validation_parameters():
      with pytest.raises(ValueError):
          ModelTrainer(latent_size=-1)  # Invalid parameter
  ```

- ❌ **Input validation**: Use mocks for quick validation
  ```python
  # Wrong: Full model instantiation for validation
  def test_invalid_scale_mult_raises():
      with pytest.raises(ValueError):
          model = ModelTeacher(scale_mult=-0.5)  # Full model created

  # Right: Mock the class for validation testing
  with pytest.raises(ValueError):
      ModelTeacher(scale_mult=-0.5)  # Constructor validates immediately
  ```

### When TO Instantiate NN Models (Must)

- ✅ **Forward pass validation**: Verify architecture produces correct shapes
  - Cannot mock tensor computations
  - Requires real model to test data flow through layers
  ```python
  def test_forward_returns_correct_shapes():
      model = ModelTeacher(latent_size=64, scale_mult=1.5)
      inputs = {...}
      outputs = model(inputs)
      assert outputs["final_latent"].shape[-1] == int(64 * 1.5)
  ```

- ✅ **Gradient flow**: Verify backpropagation behavior
  - Cannot mock gradient computation
  - Requires real model with @tf.function or tape
  ```python
  def test_gradients_computed():
      model = ModelTeacher(...)
      with tf.GradientTape() as tape:
          outputs = model(inputs)
          loss = compute_loss(outputs)
      grads = tape.gradient(loss, model.trainable_variables)
      assert all(g is not None for g in grads)
  ```

- ✅ **Training/convergence**: Verify loss decreases over steps
  - Cannot mock loss computation
  - Requires real model with real loss values
  ```python
  def test_loss_decreases_over_steps():
      trainer = ModelTrainer(...)
      loss_step_1 = trainer.train_step(batch)
      loss_step_3 = trainer.train_step(batch)
      assert loss_step_3 < loss_step_1 * 0.9
  ```

- ✅ **Integration tests**: Verify components work together
  - Only when testing end-to-end behavior
  - Example: ModelTeacher + ModelTrainer + embeddings

### Decision Matrix for NN Instantiation

| Test Purpose | Instantiate? | Why | Example |
|--------------|--------------|-----|---------|
| Parameter validation | ❌ NO | Pure logic, can mock | `test_invalid_latent_size_raises` |
| Attribute storage | ❌ NO | Configuration, can mock | `test_stores_scale_mult` |
| Error handling | ❌ NO | Logic, not behavior | `test_invalid_mode_raises` |
| Forward pass shape | ✅ YES | Tensor computation | `test_forward_output_shape` |
| Gradient computation | ✅ YES | Real backprop needed | `test_gradients_flow` |
| Training step | ✅ YES | Real loss needed | `test_train_step_reduces_loss` |
| Numerical stability | ✅ YES | Real tensors needed | `test_loss_is_finite` |
| Save/load | MAYBE | Can mock or real | `test_model_save_load_produces_same_output` |

### Cost-Benefit Analysis

**Memory per Model Instantiation**:
- EmbeddingsTable: ~50 MB
- Face2StepModel: ~200 MB
- Step2LatentModel: ~150 MB
- Full ModelTrainer: ~600-800 MB

**Time per Instantiation**: 1-2 seconds

**Impact of Avoiding Instantiation**:
- 10 tests with mocks: <1 MB, ~0.1s total
- 10 tests with real models: ~6-8 GB, ~10-20s total
- **Savings**: 600-800× memory, 100-200× time

**Use Mocks For**: Configuration, validation, logic (saves time/memory)
**Use Real Models For**: Behavior, integration, training (required for accuracy)

### Test Organization Strategy

```
tests/
├─ test_model_initialization.py (MOCKS)
│  └─ 8 tests: parameter validation, attribute storage
│     • No NN instantiation
│     • <1 second execution
│
└─ test_model_forward_pass.py (REAL MODELS)
   └─ 6 tests: shapes, gradients, integration
      • ~40% real instantiation (only forward pass tests)
      • 5-10 seconds execution
```

**Ratio Goal**: 80% mock tests, 20% real model tests
- Real model tests: Only test behavior that requires real tensors
- Mock tests: Everything else (validation, configuration, logic)

### Fixture Reuse Pattern

Reuse model fixtures across related tests to minimize instantiation cost:

```python
# Good: Fixture reused by multiple tests
@pytest.fixture
def teacher_model():
    """Create teacher once, reuse for multiple tests."""
    return ModelTeacher(latent_size=64, scale_mult=1.5)

class TestModelTeacher:
    def test_forward_shape(self, teacher_model):
        # Fixture reused here - model created once
        outputs = teacher_model(inputs)
        assert outputs["final_latent"].shape[-1] == 96

    def test_gradient_flow(self, teacher_model):
        # Fixture reused here - same model instance
        with tf.GradientTape() as tape:
            outputs = teacher_model(inputs)
            loss = compute_loss(outputs)
        grads = tape.gradient(loss, teacher_model.trainable_variables)
        assert grads is not None
```

**Savings**: Fixture reuse can reduce instantiation overhead by 50-70%

### Mocking Pattern for NN Components

When testing logic that interfaces with NN models, use lightweight mocks:

```python
# Instead of real model
class MockModel(tf.keras.Model):
    """Minimal mock for testing without real computation."""
    def call(self, inputs):
        return {
            "intermediate_latent": tf.zeros((inputs["points"].shape[0], 32, 64)),
            "final_latent": tf.zeros((inputs["points"].shape[0], 32, 64)),
        }

# Now test training logic with mock
def test_training_accepts_model_output():
    mock_model = MockModel()
    trainer = ModelTrainer(model=mock_model, ...)
    assert trainer._model is not None  # Logic test, no real computation
```

**Benefits**:
- ✅ Fast: No tensor computation
- ✅ Predictable: Outputs are deterministic
- ✅ Memory-efficient: Minimal overhead
- ✅ Focused: Tests logic, not neural net behavior

## Performance Testing Guidelines (NEW)

### **Current Problems Identified**

**Issue 1: Full Model Instantiation for Parameter Validation**
```python
# PROBLEM in test_model_wrapper.py lines 176-192:
def test_trainable_variables_sufficient_for_training(self, stats_data):
    wrapper = ModelWrapper(timesteps=2, stats=stats_data, embeddingSize=8)  # 600-800MB!
    trainable_vars = wrapper.trainable_variables  # Just counting variables!
    assert len(trainable_vars) >= 50  # Very expensive check
```

**Issue 2: Recursive Teacher Model Creation**
```python
# PROBLEM in ModelTrainer.py lines 129-138:
def __init__(self, teacher_scale_mult=2.0, ...):
    self._teacher = ModelTrainer(...)  # Creates another full ModelTrainer!
    # Memory: student + teacher + adapters = 3x+ memory usage
```

### **Solution: Proxy Functions for Fast Validation**

Create proxy functions to validate architectural properties without NN instantiation:

```python
# DO: Create proxy functions for fast validation
class ModelWrapperProxies:
    @staticmethod
    def validate_layer_sizes(timesteps, stats, embedding_size, latent_size, mode):
        """Fast validation without NN instantiation (KB memory vs 600MB)."""
        # Input validation (fail-fast approach from CODING.md)
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {latent_size}")
        if not stats:
            raise ValueError("stats dictionary cannot be empty")

        # Return expected dimensions for all submodels
        return {
            'embeddings_vocab': {k: len(v) for k, v in stats.items()},
            'expected_latent_shape': (timesteps, latent_size),
            'predictor_output_shape': 2,
            'model_modes_supported': ['full', 'encoder']
        }

    @staticmethod
    def validate_teacher_student_compatibility(student_config, teacher_config):
        """Validate teacher-student dimension compatibility without instantiation."""
        try:
            # Check embedding dimension compatibility
            student_embeddings = student_config.get('stats', {})
            teacher_embeddings = teacher_config.get('stats', {})

            for key in student_embeddings:
                if key in teacher_embeddings:
                    if len(student_embeddings[key]) != len(teacher_embeddings[key]):
                        raise ValueError(f"Embedding size mismatch for {key}: "
                                       f"student={len(student_embeddings[key])}, "
                                       f"teacher={len(teacher_embeddings[key])}")

            return True
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid configuration for teacher-student validation: {e}")
```

### **Test Migration Examples**

**Before (Slow - 600MB model instantiation):**
```python
def test_trainable_variables_sufficient_for_training(self, stats_data):
    wrapper = ModelWrapper(timesteps=2, stats=stats_data, embeddingSize=8)  # 600MB
    trainable_vars = wrapper.trainable_variables  # Expensive operation
    assert len(trainable_vars) >= 50
```

**After (Fast - Proxy function, KB memory):**
```python
def test_layer_sizes_architecture():
    sizes = ModelWrapperProxies.validate_layer_sizes(
        timesteps=2,
        stats=stats_data,
        embedding_size=8,
        latent_size=64,
        mode="full"
    )
    assert sizes['expected_latent_shape'] == (2, 64)
    assert len(sizes['embeddings_vocab']) >= 5  # userId, placeId, etc.
```

### **Performance Targets**

- **Memory Reduction**: 90% less memory usage for architecture validation tests
  - Current: 600-800MB per model instantiation
  - Target: <10MB per proxy validation
  - **Savings**: 60-80x memory reduction

- **Time Reduction**: 80% faster test execution for architecture validation
  - Current: 1-2 seconds per model instantiation
  - Target: <0.01 seconds per proxy validation
  - **Savings**: 100-200x time reduction

- **Test Categories**:
  - ✅ **Fast Tests (Proxy-Based)**: Architecture validation, parameter checking, compatibility
  - ✅ **Slow Tests (Real NN)**: Forward pass, training steps, gradient flow, save/load

### **When to Use Proxy Functions vs Real Models**

| Test Type | Use Proxy Functions | Use Real Models | Why |
|-----------|-------------------|----------------|-----|
| Parameter validation | ✅ | ❌ | Logic validation, no computation needed |
| Layer size compatibility | ✅ | ❌ | Mathematical validation, no NN required |
| Input range checking | ✅ | ❌ | Pure logic, can validate with numbers |
| Forward pass shapes | ❌ | ✅ | Requires real tensor computation |
| Training step execution | ❌ | ✅ | Requires real gradient computation |
| Save/load integration | ❌ | ✅ | Requires real serialization |

### **Private Field Access in Tests: Absolute Prohibition**
- **FORBIDDEN**: `obj._a._b` in tests (private fields of private fields)
- **REASON**: Tests should validate public contracts, not internal implementation
- **IMPACT**: Avoids creating brittle tests that break with internal refactorings
- **ALTERNATIVE**: Use proxy functions or request public API methods

### **Implementation Strategy**

1. **Phase 1**: Create proxy functions for all architectural validation
2. **Phase 2**: Migrate slow tests to use proxy functions
3. **Phase 3**: Keep real model tests only for essential behavior validation
4. **Phase 4**: Benchmark performance improvements

### **Expected Performance Impact**

**Current Test Performance**:
- `test_trainable_variables_sufficient_for_training()`: ~1.5s, 600MB memory
- ModelTrainer tests with teacher: ~3s, 1.2GB memory (due to recursion)

**Target Test Performance**:
- `test_layer_sizes_architecture()`: ~0.01s, <10MB memory
- Proxy-based teacher validation: ~0.02s, <20MB memory

**Overall Test Suite**:
- Current: ~30 seconds, ~6-8 GB memory usage
- Target: ~5 seconds, ~500MB memory usage
- **Improvement**: 6x faster, 16x less memory