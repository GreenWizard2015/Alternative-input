# Setup and training

## Setup

In theory, everything is very simple:

- Install Python 3.7+ (optionally, you can use Anaconda).
- Install TensorFlow by following the instructions on the [official website](https://www.tensorflow.org/install/pip), and don't forget about GPU drivers, etc.
- Install the necessary packages by executing `pip install -r requirements.txt` at the root of the project.

Unfortunately, in reality, things can be a bit more complicated. For example, I had to use TensorFlow version 2.7.0, as newer versions didn't recognize my GTX 1070 Ti graphics card.

## Training (Local)

To train the model on your computer, you would need a sufficiently powerful GPU. However, I personally prefer using Google Colab since my GPU is too weak for model training. The sequence of steps for training the model on your computer is quite straightforward:

1. Run `preprocess-dataset.py`
2. Run `create-test-dataset.py`
3. Run `train.py`

This should be sufficient for training a model, which will be saved as `Data/simple-model-best.h5`. This model will be automatically used by all other scripts.

## Training (Google Colab)

For training the model on Google Colab, you need to follow these steps:

1. Create a folder named `alternative-input` in your Google Drive.
2. Archive the contents of the project's root folder into `alternative-input.zip`.
3. Upload `alternative-input.zip` to the `alternative-input` folder on Google Drive.
4. Open [this notebook](https://colab.research.google.com/drive/15RBCmpVPsFfPESjBx2-XpqgXFiYL5pQi), make a copy, and run all code cells (Menu: `Runtime -> Run all`).
5. At the beginning of the notebook, you will be prompted for permission to access your Google Drive. Once you grant permission, the model training process should begin.

I use Google Colab Pro, which costs around $10 per month at minimum. This subscription tier should be sufficient for training 5-15 models, allowing you to iteratively train models for a single person. After training is complete, the `simple-model-best.h5` file will appear in Google Drive, which you'll need to download and place in your project folder.