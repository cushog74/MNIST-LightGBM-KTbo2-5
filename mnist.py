
import argparse
import joblib

import numpy as np
import pandas as pd
import lightgbm as lgb

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve
)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


def load_images(csv_path):
    data = pd.read_csv(csv_path)

    images = []

    for image_path in data["image_path"]:
        image = Image.open(image_path).convert("L")
        image = image.resize((28, 28))

        image_array = np.array(
            image,
            dtype=np.uint8
        )

        images.append(
            image_array.reshape(784)
        )

    images = np.array(images)

    return data, images


def select_thresholds(labels, probabilities):
    binary_labels = label_binarize(
        labels,
        classes=range(10)
    )

    thresholds_result = np.zeros(10)

    for digit in range(10):
        precision, recall, thresholds = (
            precision_recall_curve(
                binary_labels[:, digit],
                probabilities[:, digit]
            )
        )

        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (
                precision[:-1]
                + recall[:-1]
                + 0.000001
            )
        )

        best_index = np.argmax(f1_values)

        thresholds_result[digit] = (
            thresholds[best_index]
        )

    return thresholds_result


def predict_with_thresholds(
    probabilities,
    thresholds
):
    predictions = []

    for image_probabilities in probabilities:
        passed_classes = np.where(
            image_probabilities >= thresholds
        )[0]

        if len(passed_classes) == 0:
            predicted_class = np.argmax(
                image_probabilities
            )
        else:
            predicted_class = passed_classes[
                np.argmax(
                    image_probabilities[
                        passed_classes
                    ]
                )
            ]

        predictions.append(predicted_class)

    return np.array(predictions)


def train_model(dataset_path, model_path):
    data, images = load_images(dataset_path)

    if "label" not in data.columns:
        raise ValueError(
            "В обучающем CSV отсутствует столбец label"
        )

    labels = data["label"].to_numpy()

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    models = []
    fold_scores = []

    oof_probabilities = np.zeros(
        (len(images), 10)
    )

    for fold_number, (
        train_index,
        valid_index
    ) in enumerate(
        cross_validation.split(
            images,
            labels
        ),
        start=1
    ):
        print(
            "Обучение части",
            fold_number
        )

        X_train = images[train_index]
        y_train = labels[train_index]

        X_valid = images[valid_index]
        y_valid = labels[valid_index]

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=10,
            n_estimators=500,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_valid, y_valid)
            ],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=20
                ),
                lgb.log_evaluation(
                    period=0
                )
            ]
        )

        valid_probabilities = (
            model.predict_proba(X_valid)
        )

        valid_predictions = np.argmax(
            valid_probabilities,
            axis=1
        )

        oof_probabilities[
            valid_index
        ] = valid_probabilities

        fold_accuracy = accuracy_score(
            y_valid,
            valid_predictions
        )

        models.append(model)
        fold_scores.append(fold_accuracy)

        print(
            "Accuracy части:",
            round(fold_accuracy, 4)
        )
        print(
            "Лучшая итерация:",
            model.best_iteration_
        )
        print()

    thresholds = select_thresholds(
        labels,
        oof_probabilities
    )

    model_data = {
        "models": models,
        "thresholds": thresholds
    }

    joblib.dump(
        model_data,
        model_path
    )

    print(
        "Средняя accuracy:",
        round(np.mean(fold_scores), 4)
    )
    print(
        "Модель сохранена:",
        model_path
    )


def run_inference(
    model_path,
    input_path,
    output_path
):
    model_data = joblib.load(model_path)

    models = model_data["models"]
    thresholds = model_data["thresholds"]

    input_data, images = load_images(
        input_path
    )

    probabilities = np.zeros(
        (len(images), 10)
    )

    for model in models:
        probabilities += model.predict_proba(
            images
        )

    probabilities = (
        probabilities / len(models)
    )

    predictions = predict_with_thresholds(
        probabilities,
        thresholds
    )

    result = pd.DataFrame({
        "image_path": input_data[
            "image_path"
        ],
        "label": predictions
    })

    result.to_csv(
        output_path,
        index=False
    )

    print(
        "Обработано изображений:",
        len(result)
    )
    print(
        "Предсказания сохранены:",
        output_path
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=["train", "inference"]
)

parser.add_argument("--dataset")
parser.add_argument("--model", required=True)
parser.add_argument("--input")
parser.add_argument("--output")

arguments = parser.parse_args()


if arguments.mode == "train":
    if arguments.dataset is None:
        raise ValueError(
            "Не указан параметр --dataset"
        )

    train_model(
        arguments.dataset,
        arguments.model
    )


if arguments.mode == "inference":
    if arguments.input is None:
        raise ValueError(
            "Не указан параметр --input"
        )

    if arguments.output is None:
        raise ValueError(
            "Не указан параметр --output"
        )

    run_inference(
        arguments.model,
        arguments.input,
        arguments.output
    )
