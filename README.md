# Alternative input (gaze tracking via ordinary webcam)

## Version 0 ([Commit 4d00c1](https://github.com/GreenWizard2015/Alternative-input/blob/4d00c1f09f1a6aa96ed46c70c132986a9b728d41/))

Video: [YouTube](https://youtu.be/oKqpPAhnBPQ).

The project was initiated on June 1, 2021, although its very first version was written in 2020. I was aware of eye trackers but always considered them too expensive and, most importantly, inconvenient. Although this might sound presumptuous, but recent practical experiences confirmed those doubts.

From the outset, the decision was made to use an extremely affordable webcam. Understanding the complexity of facial detection, I decided to build the solution based on [MediaPipe](https://developers.google.com/mediapipe).

The first version was quite simple and used a hardcoded algorithm. Unfortunately, this version proved unsuitable for mass adoption. Even if an interface for system configuration was implemented, it would be a time-consuming and challenging process with mediocre results.

In this version, I implemented a basic mapping of facial position/movement to screen coordinates. This approach is simple, relatively intuitive, and potentially accurate. However, in practice, it turned out to be highly uncomfortable, causing neck and eye strain. Also, it's clear that this method is not universal, and there will be many cases requiring significant additional improvements.

## Neural network based solution

Video: [YouTube](https://youtu.be/FE1eGgjzFq8).

This repository contains a solution based on neural networks for eye movement detection and gaze tracking. In the initial stages of development, I even provided [screencasts](https://www.youtube.com/playlist?list=PLRjmXqZTJnJUWnLDQGlNGBRW9_ICNuBRd) to demonstrate the development process. However, as the complexity grew, recording screencasts became problematic due to hardware limitations on my computer.

Utilizing MediaPipe for face detection remains a crucial part of the project. However, now, the identified face mesh and, most importantly, images of both eyes are fed into the neural network. This approach significantly improves the user experience by allowing the neural network to detect eye movements, reducing the need for moving the entire head.

The usage of the neural networks brought several advantages to the project:

1. Improved Convenience: Unlike previous versions that relied solely on head movements, the neural network can now track the movement of pupils, making interactions much more convenient and natural.

1. Lightweight Model: Despite aiming for improved accuracy, I made sure to keep the neural network lightweight to ensure it can run efficiently on low-cost computers.

1. Reduced Head Movement: The ability of the neural network to detect gaze direction means users can control the interface with subtle eye movements, minimizing the need for extensive head motion.

Initially, the prediction accuracy of the gaze point was disappointingly low. I made numerous attempts to enhance it while keeping the neural network lightweight for broader accessibility. However, my efforts were met with limited success. 

Later, I was fortunate to obtain a gaze tracker from [Kirsty McNaught](https://github.com/kmcnaught). To my surprise and disappointment, the performance of this commercial gaze tracker was only marginally better in terms of accuracy and convenience compared to my development. Currently, my observations remain subjective but I hope to conduct a comprehensive comparison between the two solutions someday. It is worth noting that the commercial gaze tracker has several advantages, for example, it does not impose a significant GPU load.

The next step I plan to take is to launch a large-scale data collection process (there is already a [prototype](https://github.com/GreenWizard2015/Alternative-input-web-client/tree/main) available) and train a universal model, which will then be fine-tuned for individual users.

If you carefully examine the repository history, you will notice that I have tried numerous different approaches. Unfortunately, there are even more ideas that I haven't had the chance to implement yet.

**As of now (July 2023), the project is partially on hold due to a lack of time and resources.**

If you liked my project, you might be interested in supporting me on [Patreon](https://www.patreon.com/GreenWizard).