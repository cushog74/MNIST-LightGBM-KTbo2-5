# Классификация MNIST с помощью LightGBM

**Выполнил:** Крюков Антон  
**Группа:** КТбо2-5

## Описание

В работе построен пайплайн классификации рукописных цифр MNIST с помощью LightGBM.

Выполнены:

- кросс-валидация и early stopping;
- расчёт AUC, F1, precision и recall;
- построение confusion matrix;
- построение ROC- и Precision-Recall-кривых;
- подбор порогов классификации;
- SHAP summary plot;
- два локальных SHAP-примера;
- CLI-скрипты обучения, инференса и оценки.

## Результаты

- Accuracy: 0.9788
- AUC: 0.9997
- F1: 0.9787
- Precision: 0.9787
- Recall: 0.9786

## Файлы

- `MNIST_KryukovAA_KTbo2_5.ipynb` — основной блокнот;
- `mnist.py` — обучение и инференс;
- `evaluate.py` — оценка качества;
- `mnist_lightgbm_model.pkl` — сохранённая модель;
- `predictions.csv` — предсказания модели.
