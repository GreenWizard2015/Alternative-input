# Main application (app.py)

This is a minimalistic application designed to collect a dataset for training an AI model. The AI's purpose is to predict the likely point on the screen where your gaze is directed. The application can also be used to test a trained model's performance.

## Keyboard Shortcuts

- **ESC**: Quit the application.
- **S**: Toggle showing/hiding predictions on the screen.
- **1-9**: Switch between different application modes. Read more about these modes [here](#application-modes).
- **L**: Toggle random illumination effect.
- **B**: Toggle dynamic background color effect.
- **F1**: Toggle masking the face in the model's input.
- **F2**: Toggle masking the left eye in the model's input.
- **F3**: Toggle masking the right eye in the model's input.

## Dataset Collection Workflow And Application Modes

Read more about the dataset collection workflow [here](data-collection.md).

## Command-Line Arguments

Execute the script from the command line. If you provide the necessary command-line arguments, the application will start with the specified settings. It's recommended to customize the arguments according to your requirements.

```
python app.py --folder /path/to/dataset --steps 5 --model best --fps 30
```

The following command-line arguments can be used to customize the application's behavior:

- `--folder`: Specify the folder to store collected dataset and other data.
- `--steps`: Set the number of steps for prediction (default: 5).
- `--model`: Specify the model type (default: 'best', 'none' for no model). To reduce computational load, you can use the 'none' option to disable the AI model. This is very useful for collecting more accurate data.
- `--fps`: Set the frames per second (fps) for the application (default: 30).
