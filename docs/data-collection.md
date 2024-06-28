# Data collection instructions

This guide provides instructions how to use `app.py` for collecting an eye movement dataset under various lighting conditions. The goal is to focus your gaze on the center of the target (small sphere) while staying within the webcam's field of view. Remember to keep the sessions short, around 10-15 minutes each, to avoid straining yourself.

## General Guidelines

**ATTENTION!** Your position in front of the monitor and webcam can vary. However, the positions of the monitor and webcam must remain constant. If there is even a slight change, it is recommended to delete the `Dataset` folder and retrain the system.

- Maintain a relaxed posture and focus on the target's center.
- Your head should remain visible within the webcam frame at all times.
- Feel free to move your head subtly while focusing on the target.
- Use familiar monitor-related positions, ensuring you stay within the webcam's view.
- To pause data collection, press the `spacebar`. Press `P` to resume and continue.
- Avoid extended sessions to prevent fatigue. (10-15 minutes per session is recommended)
- Use various data collection modes. Try to balance them.

Remember, the goal is to create a diverse dataset that mimics real-world scenarios. Make sure to focus on the center of the sphere while following the guidelines above. Your contributions will greatly enhance the AI's learning capabilities.

## Lighting Conditions Modes

To enhance the AI's robustness, consider activating the following lighting modes:

1. **Random Lighting:** Experiment with this mode during evenings or under low light conditions. Introducing random lighting conditions can aid the AI in adapting better. Activate using the `L` button.
2. **Random Background:** Enable this mode to introduce diverse backgrounds behind the sphere. While it might be slightly distracting, it contributes to the AI's learning process. Activate using the `B` button.

## Data Collection Modes

### Focus Mode
- **Description**: The Focus Mode is designed to gather data while users concentrate on a specific point on their screen. While slight head movements are acceptable, try to maintain your gaze primarily on the center of the sphere during data collection.
- **Utility**: This mode facilitates the generation of gaze samples directed precisely at designated screen points. This aids in training the AI to recognize the gaze pattern when users are concentrating on a particular object.
- **Instructions**:
  1. Enable data collection by pressing the `P` key.
  2. Locate the target on your screen and ensure your focus is on its center.
  3. Initiate data collection by pressing the `Right Arrow` key. The sphere will change its color to red, indicating that data collection has begun.
  4. Keep your gaze fixated on the target's center for the ensuing 5 seconds. Data collection will exclusively occur during this duration.
  5. After 5 seconds, data collection will pause automatically, and the sphere will relocate to a new position. Now you can repeat steps 2-5.

### Corner Mode

- **Description**: This mode is designed to collect samples when the user's gaze is directed at the corners of the screen.
- **Utility**: This mode helps the system understand when the user is looking at the corners of the screen, which are challenging areas for the system to track accurately.
- **Instructions**:
  1. Locate the target on your screen and ensure your focus is on its center.
  2. Initiate data collection by pressing the `P` key. **Keep your focus on the center of the sphere. If the target moves beyond the screen boundaries, try to anticipate its position outside the screen.**
  3. Press `Space`, `Left Arrow`, or `Right Arrow` to pause data collection. (Better to use `Space` to minimize occasional gaze direction changes)
  4. Use the left and right arrow keys to switch between corners where the target will relocate. Now you can repeat steps 1-4.
  
### Spline Mode
- **Description**: The Spline Mode is designed to gather gaze movement data following a spline curve on the screen. This trajectory enables the efficient collection of a wide spectrum of gaze positions and their corresponding movements.
- **Utility**: This mode generates gaze data along non-linear paths, which aids the AI in comprehending intricate gaze trajectories commonly encountered during real-world interactions.
- **Instructions**:
  1. Locate the target on your screen and ensure your focus is on its center.
  2. Initiate data collection by pressing the `P` key. **Keep your focus on the center of the target. If the target moves beyond the screen boundaries, try to anticipate its position outside the screen.**
  3. Press `Space` or `P` to pause data collection. (Better to use `Space` to minimize occasional gaze direction changes)
  
  You can use this mode as much as you want. However, it's advisable to take breaks when it becomes difficult to focus your gaze on the target.

### Circular Movement Mode
- **Description**: In this mode, the target will move along a rectangular trajectory around the center of the screen. Both clockwise and counterclockwise movements are available.
- **Utility**: This mode aims to collect gaze data from points along the screen's edges and ensure even coverage of the entire display.
- **Instructions**:
  1. Enable data collection by pressing the `P` key.
  2. Adjust the size of the rectangular trajectory by pressing the `Up Arrow` or `Down Arrow` key.
  3. Locate the target on your screen and ensure your focus is on its center.
  4. Initiate clockwise movement by pressing the `Right Arrow` key and counterclockwise movement by pressing the `Left Arrow` key. Keep your focus on the center of the target while it moves.
  5. Data collection will pause automatically after the target completes one full rotation. Now you can repeat steps 2-5.
Certainly, here's the information you provided formatted as examples in English:

### Game Mode
- **Description**: This mode is quite similar to the Focus Mode, but it utilizes a pre-trained model. In this mode, it's recommended to disable the display of predictions on the screen by pressing the `S` key. This helps you focus on the target and not get distracted by predictions. Your main focus should be on the target. Meanwhile, the cursor detection area will keep expanding until the predicted point is inside that area. After the first successful prediction, data collection will start, and the process will repeat several times. Once completed, the target will move to a new position. This way, more data will be gathered for problematic screen areas where the model struggles to predict gaze points. Until the first successful prediction, the area will be red, and data collection will not occur. If the model is well-trained, this process can turn into an engaging game. As always, try to maintain your focus on the target during data collection and avoid distractions. You can use the pause functionality by pressing the `Space` key when needed. Additionally, you can adjust the probability of the target appearing closer to the screen edges using the `Up Arrow` and `Down Arrow` keys.
- **Utility**: This mode serves to collect data while engaging users in an interactive process to improve gaze point prediction accuracy, especially in challenging areas of the screen.
- **Instructions**: 
  1. Press `S` to turn off prediction display.
  2. Press `P` to start data collection.
  3. Locate the target on your screen and concentrate your focus on its center.
  4. As the detection area expands, keep your focus on the target until the predicted point falls within the area.
  5. After the first successful prediction, data collection will start for a few repetitions. Try to maintain focus on the target and avoid distractions. Use `Space` to pause if needed.
  6. The target will then move to a new position, repeating the process.