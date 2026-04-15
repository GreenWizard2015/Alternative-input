# Setup and training

## Setup

In theory, everything is very simple:

- Install Python 3.7+ (optionally, you can use Anaconda).
- Install TensorFlow by following the instructions on the [official website](https://www.tensorflow.org/install/pip), and don't forget about GPU drivers, etc.
- Install the necessary packages by executing `pip install -r requirements.txt` at the root of the project.

Unfortunately, in reality, things can be a bit more complicated. For example, I had to use TensorFlow version 2.7.0, as newer versions didn't recognize my GTX 1070 Ti graphics card.

## Training

To train the model on your computer, you would need a sufficiently powerful GPU. The sequence of steps for training the model is quite straightforward:

1. Run `python3 scripts/preprocess-dataset.py`
2. Run `python3 scripts/create-test-dataset.py`
3. Run `python3 scripts/train.py`

This should be sufficient for training a model, which will be saved as `Data/simple-model-best.weights.h5`. This model will be automatically used by all other scripts.

## Testing

To run the test suite:

```bash
make test
```

This will run all tests using pytest. The test command activates the conda environment automatically.

To format code and verify it meets style guidelines:

```bash
make format
```

This runs black, flake8, and mypy checks.