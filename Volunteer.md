# Volunteer Instructions for the Alternative-input Project

Thank you for your interest in contributing to the Alternative-input project! Your participation will help us gather a diverse dataset to train AI, enhancing accessibility technology for individuals with disabilities. Below, you will find detailed instructions on how to participate.

## Overview

The Alternative-input project uses an ordinary webcam to track user gaze, aiming to create a more adaptive and personalized user experience than commercial trackers. Your contribution involves collecting data that will enable the AI to learn and adapt to various user environments, monitor settings, and individual needs.

## Getting Started

1. **Visit the Web Application:** [Alternative-input Web Client](https://alternative-input-web-client.vercel.app/)
2. **Check the Code:** You can view and contribute to the main application/AI code on [GitHub](https://github.com/GreenWizard2015/Alternative-input).

## Privacy Notice

- The application does not collect any personal data.
- Only unique identifiers are stored for convenience.

## Data Collection Instructions

1. **Camera Setup:**
   - Use a static camera (tablets or mobile devices are not suitable).
   - Ensure that only one person is in the camera's field of view.
   - Your eyes should be visible in the top left corner of the screen for correct setup.

2. **Session Guidelines:**
   - Perform sessions in 10-minute intervals to prevent eye fatigue and maintain focus.
   - Follow the red ball on the screen with your gaze. Move your head naturally but avoid fixing it.
   - Ignore other objects on the screen meant for augmentation.

3. **Creating New Places and Users:**
   - Create a new Place each time you change the position of the webcam, the window's location, or the distance to the screen.
   - Create a new User for each person, including variations like with and without glasses.

4. **Avoid Distractions:**
   - Avoid resizing the window during the session.
   - If tired or distracted, take a break. Press pause to prevent bad data from entering the system.

5. **Data Submission:**
   - Do not turn off the app immediately after finishing. Wait for the app to notify you that the data has been sent.

## Recommended Practices

- Conduct several short sessions throughout the day, each lasting 5-15 minutes.
- Perform sessions under different lighting conditions.
- Use primarily the spline mode, but also try other modes.
- Gather at least 20,000 frames for each user/location/etc. to ensure sufficient data.

## Known Issues

- The application may be slow due to AI-based face detection.
- Face detection does not work in Safari. Please use Chrome, Edge, or Firefox.
- The application is demanding and may not work well on some devices.
- The data collection speed should be at least 10 samples per second.

## Additional Notes

- The application is for data collection only and does not include the AI.
- The system has a 3-second window for pauses; ensure someone can quickly press the space bar if needed.
- If you have any feedback or encounter issues, please visit the [GitHub repository](https://github.com/GreenWizard2015/Alternative-input) to report them or contribute improvements.

Thank you for your valuable contribution to making technology more inclusive and accessible for everyone. Your participation can make a significant difference in the lives of many!