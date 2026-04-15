"""Interactive viewer for clean/augmented eye image samples.

Displays eye images side-by-side with 8x zoom for visual inspection of
data augmentation effects. Also overlays face mesh points for reference.
"""

from pathlib import Path
import cv2
from Core.data.DatasetLoader import DatasetLoader
from Core.data.DataSampler import DataSampler
from Core.data.sample_viewer import create_visualization
from Core.logging_config import get_logger
from Core import Utils
from Core.data.AugmentationDefaults import DEFAULT_AUGMENTATION_PARAMS

logger = get_logger(__name__)

# Key codes for interactive control
KEY_ESC = 27  # Escape key
KEY_SPACE = 32  # Space key


def main() -> None:
    """Main entry point for sample viewer.

    Displays eye image samples side-by-side with augmentation effects.
    User can navigate between samples using keyboard controls (ESC=exit, SPACE=next).

    Returns:
        None (interactive viewer, runs until user exits).

    Example:
        >>> main()
        # Opens interactive window showing augmented vs clean eye images
    """
    # Configuration
    timesteps = 5
    zoom_factor = 8
    num_samples = 4

    # Setup paths
    folder = Path(__file__).parent.parent / "Data" / "remote"
    stats = str(folder / "stats.json")

    # Initialize loader with augmentation config
    samplerArgs = {
        "batch_size": num_samples,
        "minFrames": timesteps,
        "maxT": 1.0,
        "defaults": {
            "timesteps": timesteps,
            "stepsSampling": "uniform",
            **DEFAULT_AUGMENTATION_PARAMS,
        },
    }

    loader = DatasetLoader(
        json_path=stats,
        samplerArgs=samplerArgs,
        sampler_class=DataSampler,
        batchPerEpoch=1,
    )

    # Continuous sampling loop until Escape is pressed
    sample_count = 0
    combined_with_header = None
    logger.info("Starting continuous sampling...")
    logger.info(
        "Press ESC to exit, SPACE for next sample, any other key to show previous"
    )

    while True:
        # Sample new batch
        sample_count += 1
        X, Y = loader.sample(batch_size=num_samples)

        # Convert tensors to numpy for visualization
        X = Utils.to_numpy(X)

        # Create visualization
        combined_with_header = create_visualization(X, zoom_factor)

        # Display
        cv2.imshow(
            winname="Eye Samples: Clean vs Augmented (ESC=exit, SPACE=next, other=prev)",
            mat=combined_with_header,
        )

        # Wait for key press with longer timeout for viewing
        key = cv2.waitKey(delay=0) & 0xFF

        if key == KEY_ESC:
            logger.info("Exiting. Viewed %d samples.", sample_count)
            break
        elif key == KEY_SPACE:
            # Continue to next sample
            continue
        else:
            # Any other key: show previous sample again
            if combined_with_header is not None:
                cv2.imshow(
                    winname="Eye Samples: Clean vs Augmented (ESC=exit, SPACE=next, other=prev)",
                    mat=combined_with_header,
                )
                cv2.waitKey(delay=0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
