"""Fixtures for EvaluationTracker tests."""

import pytest
import numpy as np


class MockModel:
    """Mock model with eval() and save() methods."""

    def __init__(self):
        self.eval_calls = []
        self.save_calls = []
        self.eval_results = {}

    def eval(self, batch):
        """Mock eval method that returns predefined metrics."""
        self.eval_calls.append(batch)
        # Make a deep copy to avoid issues with the same reference
        # Convert scalar values to arrays for concatenation
        result = {}
        for k, v in self.eval_results.items():
            if isinstance(v, (int, float)):
                # Convert scalar to array with shape (1,) for concatenation
                result[k] = np.array([v])
            else:
                result[k] = v.copy() if hasattr(v, "copy") else v
        return result

    def set_eval_results(self, results):
        """Helper method to set evaluation results."""
        self.eval_results = results.copy()

    def save(self, folder, postfix="best"):
        """Mock save method that tracks save calls."""
        self.save_calls.append({"folder": folder, "postfix": postfix})


class MockDataset:
    """Mock dataset with required interface."""

    def __init__(self, size=5, eval_results=None):
        self.size = size
        self.eval_results = eval_results or {
            "total": 0.5,
            "loss": 0.3,
            "distance": 0.2,
            "result": np.ones((2, 4, 2)),  # Add required 'result' key
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """Return batch as (X_dict, Y_dict)."""
        # Create Y_dict with 'result' key containing tensor-like object
        Y_dict = {"result": np.ones((2, 4, 2))}  # Shape (batch_size, timesteps, 2)
        X_dict = {"points": np.ones((2, 4, 468, 2))}
        return (X_dict, Y_dict)

    def sample(self, batchId, no_face=False, no_eyes=False):
        """Mock sample method that returns generator yielding batch data."""
        # Create Y_dict with 'result' key containing tensor-like object
        Y_dict = {"result": np.ones((2, 4, 2))}  # Shape (batch_size, timesteps, 2)
        X_dict = {"points": np.ones((2, 4, 468, 2))}

        # Apply no_face/no_eyes modifications if requested
        if no_face:
            X_dict = {"points": np.zeros((2, 4, 468, 2))}
        if no_eyes:
            X_dict["left eye"] = np.zeros((2, 4, 32, 32, 1))
            X_dict["right eye"] = np.zeros((2, 4, 32, 32, 1))

        yield (X_dict, Y_dict)


@pytest.fixture
def mock_model():
    """Fixture providing mock model."""
    model = MockModel()
    model.eval_results = {
        "total": 0.5,
        "loss": 0.3,
        "distance": 0.2,
        "result": np.ones((2, 4, 2)),  # Add required 'result' key
    }
    return model


@pytest.fixture
def mock_datasets():
    """Fixture providing list of mock datasets."""
    return [
        MockDataset(
            size=2,
            eval_results={
                "total": 0.55,
                "loss": 0.35,
                "distance": 0.2,
                "result": np.ones((2, 4, 2)),
            },
        ),
        MockDataset(
            size=2,
            eval_results={
                "total": 0.55,
                "loss": 0.35,
                "distance": 0.2,
                "result": np.ones((2, 4, 2)),
            },
        ),
    ]
