"""Default augmentation parameters used across data processing pipelines.

These constants define the standard augmentation settings used by:
- Training scripts (train.py)
- Sample viewers (show_samples.py)
- Data samplers and loaders
"""

# Default augmentation parameters for data processing
DEFAULT_AUGMENTATION_PARAMS = {
    "pointsNoise": 0.002,  # Standard deviation of Gaussian noise for face mesh points
    "pointsDropout": 0.5,  # Probability of dropping (masking) face mesh points
    "eyesDropout": 0.1,  # Probability of dropping (masking) eye image regions
    "eyesAdditiveNoise": 0.01,  # Standard deviation of Gaussian noise for eye images
    "brightnessFactor": 1.1,  # Multiplicative factor for brightness augmentation
    "lightBlobFactor": 1.1,  # Multiplicative factor for light blob augmentation
    "modalityDropout": 0.05,
    "regionFactor": 1.0,
}
