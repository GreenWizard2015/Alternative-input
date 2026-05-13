"""Tensor conversion utilities for data processing.

Provides the main toTensor function that converts sample data to augmented
tensors with clean/augmented variants, applying various data augmentations.
"""

from typing import Dict, Tuple
import tensorflow as tf
from Core.logging_config import get_logger
from Core.landmarks import FACE_MESH_INVALID_VALUE
from Core.data.gaussian_utils import addLightBlob
from Core.data.augmentation import (
    apply_brightness_augmentation,
    apply_additive_noise,
    apply_dropout,
    apply_points_noise,
    apply_points_dropout,
    BRIGHTNESS_TRUNCATED_NORMAL_STDDEV,
)

logger = get_logger(__name__)


# Constants for data processing
EYE_IMAGE_SIZE = 48  # Original eye image dimensions (48x48)
EYE_CROP_SIZE = 32  # Cropped eye image dimensions (32x32)
# Fraction of eye image to use after cropping
EYE_CROP_FRACTION = EYE_CROP_SIZE / EYE_IMAGE_SIZE
LIGHT_BLOB_POSITION_MIN = 0.0  # Minimum position for light blob generation


def withCrop(x: tf.Tensor, pos: tf.Tensor) -> tf.Tensor:
    """Apply center crop to image tensor.

    Args:
        x: Input image tensor of shape (N, H, W).

    Returns:
        Center-cropped image tensor of shape (N, EYE_CROP_SIZE, EYE_CROP_SIZE).
    """
    flat_n = tf.shape(x)[0]
    flat_n = tf.cast(flat_n, tf.int32)
    return tf.image.crop_and_resize(
        tf.expand_dims(x, -1),
        boxes=pos,
        box_indices=tf.range(flat_n),
        crop_size=[EYE_CROP_SIZE, EYE_CROP_SIZE],
    )[..., 0]


def _generate_random_crop_boxes(
    flat_n: tf.Tensor, region_factor: tf.Tensor
) -> tf.Tensor:
    """Generate random crop boxes for eye images.

    Generates random top-left corner positions ensuring crops stay within [0, 1].
    crop_and_resize expects boxes in [y1, x1, y2, x2] normalized format.

    Args:
        flat_n: Number of images to generate crops for.

    Returns:
        Boxes tensor of shape (flat_n, 4) in [y1, x1, y2, x2] format.
    """

    def rnd():
        # Generate random top-left corner positions ensuring crop stays within [0, 1]
        max_pos = 1.0 - EYE_CROP_FRACTION  # Ensure bottom-right corner stays <= 1.0
        pos_y1 = tf.random.uniform([flat_n], minval=0.0, maxval=max_pos)
        pos_x1 = tf.random.uniform([flat_n], minval=0.0, maxval=max_pos)

        # square box size
        # if x=12, y=15 => size=48 - max(12, 15) = 48 - 15 = 33
        max_size = 1.0 - tf.maximum(pos_x1, pos_y1)
        size = tf.random.uniform([flat_n], minval=EYE_CROP_FRACTION, maxval=max_size)
        # Create boxes in [y1, x1, y2, x2] format
        boxes = tf.stack(
            [pos_y1, pos_x1, pos_y1 + size, pos_x1 + size],
            axis=-1,
        )
        # replace some with central crops
        size = tf.random.uniform([flat_n], minval=EYE_CROP_FRACTION, maxval=1.0)
        central_boxes = tf.stack(
            [0.5 - size / 2.0, 0.5 - size / 2.0, 0.5 + size / 2.0, 0.5 + size / 2.0],
            axis=-1,
        )
        mask = tf.random.uniform([flat_n, 1]) < 0.5
        tf.assert_equal(tf.shape(central_boxes), tf.shape(boxes))
        return tf.where(mask, central_boxes, boxes)

    boxes = tf.cond(0.0 < region_factor, rnd, lambda: _generate_central_crop(flat_n))
    tf.assert_equal(tf.shape(boxes), (flat_n, 4))
    return boxes


def _generate_central_crop(flat_n: tf.Tensor) -> tf.Tensor:
    pos = tf.constant(
        value=[[0.0, 0.0, 1.0, 1.0]],
        dtype=tf.float32,
    )
    return tf.repeat(pos, flat_n, axis=0)


# helper function for masking
def _applyMasking_helper(src, maskSize, maskFraction=0.3):
    total = maskSize * maskSize
    minC = 0
    maxC = tf.cast(tf.cast(total, tf.float32) * maskFraction, tf.int32)
    shp = tf.shape(src)
    B = shp[0]
    # number of masked cells per image
    maskedCellsN = tf.random.uniform((B, 1), minC, maxC + 1, dtype=tf.int32)
    # generate probability mask for each image
    mask = tf.random.uniform((B, total), 0.0, 1.0)
    # get sorted indices of the mask
    cellsOrdered = tf.argsort(mask, axis=-1, direction="DESCENDING")
    # get value of maskedCellsN-th element in each row
    indices = tf.gather(cellsOrdered, maskedCellsN, batch_dims=1)
    threshold = tf.gather(mask, indices, batch_dims=1)
    tf.assert_equal(tf.shape(threshold), (B, 1))
    # make binary mask, where 1 means NOT masked
    mask = tf.cast(mask <= threshold, tf.float32)
    # reshape mask to (B, size, size, 1)
    mask = tf.reshape(mask, (B, maskSize, maskSize, 1))
    # scale mask to image size
    imageSize = shp[1:3]
    mask = tf.image.resize(mask, imageSize, method="nearest")
    mask = tf.reshape(mask, shp)
    tf.assert_equal(tf.shape(mask), shp)
    # apply mask to source image
    maskValue = tf.random.uniform((B, 1, 1), minval=0.0, maxval=1.0)
    return src * mask + (1.0 - mask) * maskValue


@tf.function(
    input_signature=[
        (
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ),
        tf.TensorSpec(shape=(9,), dtype=tf.float32),
        # userId, screenId, cameraId, monitorId, placeId
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def toTensor(
    data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    params: tf.Tensor,
    userId: tf.Tensor,
    screenId: tf.Tensor,
    cameraId: tf.Tensor,
    monitorId: tf.Tensor,
    placeId: tf.Tensor,
) -> Dict[str, Dict[str, tf.Tensor]]:
    """Convert sample data to augmented tensors with clean/augmented variants.

    Applies various augmentations (noise, dropout, brightness, light blobs)
    to face mesh points and eye images, returning both clean and augmented
    versions.

    Args:
        data: Tuple of (points, left_eye, right_eye, time) tensors
        params: Augmentation parameters (7 values: noise, dropout, etc.)
        userId: User ID scalar
        screenId: Screen ID scalar
        cameraId: Camera ID scalar
        monitorId: Monitor ID scalar
        placeId: Place ID scalar

    Returns:
        Dictionary with 'clean' and 'augmented' keys, each containing:
        - time, points, left eye, right eye, userId, screenId, cameraId, monitorId, placeId
    """
    logger.debug("Converting sample data to tensor format")
    (
        pointsNoise,
        pointsDropout,
        eyesAdditiveNoise,
        eyesDropout,
        brightnessFactor,
        lightBlobFactor,
        modality_dropout,
        region_factor,
        timesteps,
    ) = tf.unstack(params)
    timesteps = tf.cast(timesteps, tf.int32)
    points, imgA, imgB, T = data
    # normalize and convert time to float32 to reduce memory usage
    shp = tf.shape(T)
    delta_before = T[:, 1:] - T[:, :-1]
    min_time = tf.reduce_min(T, axis=-1, keepdims=True)
    T = tf.cast(T - min_time, tf.float32)
    delta_after = T[:, 1:] - T[:, :-1]
    T = tf.reshape(T, shp)
    delta_after = tf.cast(delta_after, delta_before.dtype)
    # sanity check
    tf.debugging.assert_near(delta_before, delta_after, rtol=1e-4)
    ###########
    flat_n = tf.shape(points)[0]
    flat_n = tf.cast(flat_n, tf.int32)
    imgA = tf.cast(imgA, tf.float32) / 255.0
    imgB = tf.cast(imgB, tf.float32) / 255.0

    tf.assert_equal(tf.shape(imgA), (flat_n, EYE_IMAGE_SIZE, EYE_IMAGE_SIZE))
    tf.assert_equal(tf.shape(imgA), tf.shape(imgB))

    userId = tf.fill((flat_n, 1), userId)
    screenId = tf.fill((flat_n, 1), screenId)
    cameraId = tf.fill((flat_n, 1), cameraId)
    monitorId = tf.fill((flat_n, 1), monitorId)
    placeId = tf.fill((flat_n, 1), placeId)

    def reshape(x: tf.Tensor) -> tf.Tensor:
        """Reshape tensor from flat batch to (batches, timesteps, ...) format.

        Converts (N, ...) tensor to (N // timesteps, timesteps, ...) for temporal processing.

        Args:
            x: Input tensor with shape (N, ...)

        Returns:
            Reshaped tensor with temporal dimension added.
        """
        # Reshape x from (N, ...) to (N // timesteps, timesteps, ...)
        remaining_shape = tf.shape(x)[1:]
        new_shape = tf.concat(
            [tf.stack([flat_n // timesteps, timesteps]), remaining_shape],
            axis=0,
        )
        return tf.reshape(x, shape=new_shape)

    # apply center crop
    pos = _generate_central_crop(flat_n)
    clean = {
        "time": reshape(T),
        "points": reshape(points),
        "left eye": tf.expand_dims(reshape(withCrop(imgA, pos)), -1),
        "right eye": tf.expand_dims(reshape(withCrop(imgB, pos)), -1),
        "userId": reshape(userId),
        "screenId": reshape(screenId),
        "cameraId": reshape(cameraId),
        "monitorId": reshape(monitorId),
        "placeId": reshape(placeId),
    }
    ##########################
    # Apply random crop eye images
    random_pos = _generate_random_crop_boxes(flat_n, region_factor=region_factor)
    imgA = withCrop(imgA, random_pos)
    imgB = withCrop(imgB, random_pos)
    ##########################

    # Apply brightness augmentation
    imgA, imgB = apply_brightness_augmentation(imgA, imgB, brightnessFactor, flat_n)

    # Apply light blob augmentation
    def apply_light_blob() -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply light blob augmentation to images.

        Returns:
            Tuple of light-blob-augmented (imgA, imgB) tensors.
        """

        def sampleBrightness(a: tf.Tensor, b: tf.Tensor, mid: float = 1.0) -> tf.Tensor:
            """Sample brightness factor from truncated normal distribution.

            Args:
                a: Lower bound tensor.
                b: Upper bound tensor.
                mid: Midpoint value (default: 1.0).

            Returns:
                Sampled brightness factors.
            """
            # Note: flat_n is a scalar tensor; use tf.stack to create proper shape [flat_n]
            TN = tf.random.truncated_normal(
                tf.stack([flat_n]), mean=0.0, stddev=BRIGHTNESS_TRUNCATED_NORMAL_STDDEV
            )
            return tf.where(TN < 0.0, a + (mid - a) * (TN + 1.0), mid + (b - mid) * TN)

        LightBlobPower = sampleBrightness(1.0 / lightBlobFactor, lightBlobFactor)
        return addLightBlob(imgA, imgB, LightBlobPower, shared=False)

    imgA, imgB = tf.cond(0.0 < lightBlobFactor, apply_light_blob, lambda: (imgA, imgB))

    # Apply additive noise
    imgA, imgB = apply_additive_noise(imgA, imgB, eyesAdditiveNoise)

    # Apply masking
    imgA = _applyMasking_helper(
        imgA, maskSize=tf.random.uniform((), 16, 24 + 1, dtype=tf.int32)
    )
    imgB = _applyMasking_helper(
        imgB, maskSize=tf.random.uniform((), 16, 24 + 1, dtype=tf.int32)
    )

    # Apply dropout to images
    imgA, imgB = apply_dropout(imgA, imgB, eyesDropout, flat_n)

    ##########################
    validPointsMask = tf.reduce_all(
        FACE_MESH_INVALID_VALUE != points, axis=-1, keepdims=True
    )

    # Apply noise to points
    points = apply_points_noise(points, pointsNoise)
    # Apply dropout to points
    points = apply_points_dropout(points, pointsDropout, flat_n)
    points = tf.where(validPointsMask, points, FACE_MESH_INVALID_VALUE)

    ##########################
    # modality dropout
    def apply_modality_dropout():
        """Apply modality dropout with random masking."""
        modality_mask = tf.random.uniform([flat_n]) < modality_dropout
        points_modality_mask = tf.random.uniform([flat_n]) < 0.5
        points_mask = tf.logical_and(modality_mask, points_modality_mask)
        eyes_mask = tf.logical_and(modality_mask, tf.logical_not(points_modality_mask))
        return (
            tf.where(points_mask[:, None, None], FACE_MESH_INVALID_VALUE, points),
            tf.where(eyes_mask[:, None, None], 0.0, imgA),
            tf.where(eyes_mask[:, None, None], 0.0, imgB),
        )

    points, imgA, imgB = tf.cond(
        0.0 < modality_dropout,
        apply_modality_dropout,
        lambda: (points, imgA, imgB),
    )

    # Validate augmented points are in valid range
    return {
        "augmented": {
            "time": reshape(T),
            "points": reshape(points),
            "left eye": tf.expand_dims(reshape(imgA), -1),
            "right eye": tf.expand_dims(reshape(imgB), -1),
            "userId": reshape(userId),
            "screenId": reshape(screenId),
            "cameraId": reshape(cameraId),
            "monitorId": reshape(monitorId),
            "placeId": reshape(placeId),
        },
        "clean": clean,
    }
