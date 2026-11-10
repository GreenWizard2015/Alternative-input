"""Tests for train/test CSV metrics logging."""

import csv

import numpy as np

from scripts.training_utils import MetricsCsvLogger


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_metrics_csv_logger_writes_separate_unique_train_and_test_files(tmp_path):
    first = MetricsCsvLogger(tmp_path)
    second = MetricsCsvLogger(tmp_path)

    assert first.paths["train"] != first.paths["test"]
    assert first.paths["train"] != second.paths["train"]

    first.log("train", epoch=1, step=0, metrics={"loss": np.float32(0.5)})
    first.log("train", epoch=1, step=1, metrics={"accuracy": 0.9})
    first.log("test", epoch=0, metrics={"loss": np.float64(0.7)})
    first.close()
    second.close()

    train_rows = _read_csv(first.paths["train"])
    test_rows = _read_csv(first.paths["test"])

    with first.paths["train"].open(newline="", encoding="utf-8") as file:
        train_header = next(csv.reader(file))
    with first.paths["test"].open(newline="", encoding="utf-8") as file:
        test_header = next(csv.reader(file))

    assert train_header[:2] == ["epoch", "step"]
    assert set(train_header[2:]) == {"loss", "accuracy"}
    assert test_header[:1] == ["epoch"]
    assert test_header[1:] == ["loss"]
    assert train_rows[0]["epoch"] == "1"
    assert train_rows[0]["step"] == "0"
    assert float(train_rows[0]["loss"]) == 0.5
    assert train_rows[1]["loss"] == ""
    assert float(train_rows[1]["accuracy"]) == 0.9
    assert test_rows == [{"epoch": "0", "loss": "0.7"}]
