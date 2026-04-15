# Debugging Guide: FaceMeshEncoder Variable Tracking Bug

## The Bug
Save/load cycle tests were failing with predictions diverging between before and after loading. The model wrapper would save weights but predictions wouldn't match after loading.

## How I Found It

### Step 1: Initial Observation
```
FAILED tests/model_wrapper/test_model_wrapper_save_load.py::test_save_load_preserves_predictions
Expected: predictions_before ≈ predictions_after (atol=1e-6)
Actual: Large differences in predicted values
```

### Step 2: Root Cause Investigation
The initial hypothesis was that `only_valid_points()` was modifying inputs and causing different predictions. But after removing it temporarily, tests still failed - revealing the real issue was elsewhere.

**Key insight**: The problem wasn't in the logic flow, but in **model state management**.

### Step 3: Variable Tracking Deep Dive
Created a simple test to inspect what variables were being tracked:

```python
encoder = FaceMeshEncoder(latent_size=64)
dummy_input = tf.random.normal((2, 478, 2))
encoder(dummy_input)

print('Trainable variables:')
for v in encoder.trainable_variables:
    print(f'  {v.name}: {v.shape}')
```

**Result**: `InvalidEmbedding` was NOT in the trainable variables list!

This meant:
- The variable existed in the encoder
- But Keras didn't know about it
- So it wasn't saved/loaded with model weights
- After loading, it was reinitialized with random values
- This caused different predictions

### Step 4: The Root Cause
The original code created the variable as:
```python
self.invalid_embedding = tf.Variable(
    tf.random.normal((internal_latent_size,)),
    trainable=True,
    name="InvalidEmbedding",
)
```

**Problem**: Raw `tf.Variable()` instances are NOT automatically tracked by Keras models for serialization.

Keras has special tracking mechanisms that work with:
- Layer instances (e.g., `self.some_layer = L.Dense(...)`)
- Weights created via `self.add_weight()`

### Step 5: The Fix
Changed to use Keras's weight management system:
```python
self.invalid_embedding = self.add_weight(
    name="InvalidEmbedding",
    shape=(internal_latent_size,),
    initializer="random_normal",
    trainable=True,
)
```

This tells Keras:
- Track this variable
- Save it with model weights
- Restore it when loading
- Use the proper naming convention

## Verification

After the fix, the variable appeared in trainable_variables:
```
InvalidEmbedding: (8,)  ← Now tracked!
```

And all tests passed:
```
====================== 118 passed, 39 warnings ======================
```

## Key Lessons

1. **Model State ≠ Code Logic**: A bug in predictions doesn't always mean logic is wrong - check if model state is being preserved

2. **Keras Variable Tracking**: Use `self.add_weight()` or layer instances, not raw `tf.Variable()`, for automatic tracking
   - This applies to ALL trainable variables in custom Keras layers
   - Found this pattern in 2 locations: `FaceMeshEncoder` and `CoordsEncodingLayer`

3. **Test the Simplest Case**: A minimal reproduction test revealed the issue immediately:
   - Create encoder
   - Build it
   - Check `trainable_variables`
   - This is faster than debugging complex test failures

4. **Follow Framework Conventions**: Notice other layers use `self.name` in naming - consistency with the framework is important

5. **Save/Load as Integration Test**: This type of test catches serialization bugs that unit tests miss

6. **Systemic Issues**: Once you identify a pattern (like using raw `tf.Variable`), search the codebase for other instances of the same anti-pattern

## Debug Strategy Used
1. Observe symptom (predictions differ after save/load)
2. Form hypothesis (only_valid_points modifying inputs)
3. Test hypothesis (remove the code, still fails)
4. Investigate deeper (inspect variable tracking)
5. Create minimal reproduction (print trainable_variables)
6. Find root cause (variable not tracked)
7. Apply fix (use add_weight)
8. Verify solution (tests pass + variable tracked)
