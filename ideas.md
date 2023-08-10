- use (V)AE to compress face/eyes representations
- use siamese network to compress face/eyes representations
.....................................................
- learnable codes of environment
  - create a custom layer for embeddings. It should be able to save and load embeddings from the folder. 
  - for training, generate folder "train-context" and put there files with ContextID_global, ContextID_local, ContextID_sublocal as names "{id}-{hash of trajectory}.bin"
  - before testing, drop contexts/embeddings and perform fine-tuning of embeddings (1k small batches, 1 epoch, linear annealing lr from 1e-1 to 1e-5, trainable only embeddings). embeddings should be saved to "trained-context" folder.

- create test dataset. Its a folder with npz files. Each file is a single batch of clean samples.
- create pretrain dataset. Its a single npz file. It contains sampled batches of augmented samples. I.e. it is a just like a batch of infinite size.

This idea partially implemented in [this version](https://github.com/GreenWizard2015/Alternative-input/tree/6651b11ba46d950b15988e86b8f087260a26f92e).
.....................................................
- reduce the number of face keypoints
  - only eyes
  - eyes + face "corners" (4 points?)
- add extra data ("center" between eyes, angles, etc)
- normalize face keypoints
- auxiliary loss for face keypoints, because they have currently minor impact on the final result (we can set them to -1 and get almost the same result)

- normalize eyes images (by transforming them according to keypoints)
- add keypoints as an additional image channel (maybe, it will help to learn better representation)