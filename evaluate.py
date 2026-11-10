
import argparse

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--ground-truth",
    required=True
)

parser.add_argument(
    "--predictions",
    required=True
)

arguments = parser.parse_args()


ground_truth = pd.read_csv(
    arguments.ground_truth
)

predictions = pd.read_csv(
    arguments.predictions
)


if "image_path" not in ground_truth.columns:
    raise ValueError(
        "В ground truth отсутствует столбец image_path"
    )

if "label" not in ground_truth.columns:
    raise ValueError(
        "В ground truth отсутствует столбец label"
    )

if "image_path" not in predictions.columns:
    raise ValueError(
        "В predictions отсутствует столбец image_path"
    )

if "label" not in predictions.columns:
    raise ValueError(
        "В predictions отсутствует столбец label"
    )


comparison = ground_truth.merge(
    predictions,
    on="image_path",
    suffixes=("_true", "_pred")
)


accuracy = accuracy_score(
    comparison["label_true"],
    comparison["label_pred"]
)

matrix = confusion_matrix(
    comparison["label_true"],
    comparison["label_pred"]
)


print(
    "Количество сравнённых изображений:",
    len(comparison)
)

print(
    "Accuracy:",
    round(accuracy, 4)
)

print()
print("Confusion Matrix:")
print(matrix)
