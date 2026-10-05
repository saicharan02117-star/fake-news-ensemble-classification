# Fake News Detection Using Ensemble Text Classification

This is a Python machine-learning laboratory project that classifies article text as `FAKE` or `REAL`. It uses TF-IDF feature extraction and combines Logistic Regression, K-Nearest Neighbors and a Decision Tree through hard majority voting.

## Main workflow

1. Combine labelled `Fake.csv` and `True.csv` records.
2. Clean and normalize article text.
3. Fit TF-IDF on the training data.
4. Train Logistic Regression, KNN and Decision Tree models.
5. Select the majority class from the three predictions.
6. Report accuracy, precision, recall, F1-score and a confusion matrix.

## Run locally

```bash
python dev_server.py
```

Open `http://127.0.0.1:8000`.

## Train and evaluate

Place `Fake.csv` and `True.csv` in the `data` folder, then run:

```bash
python train_model.py --fake data/Fake.csv --true data/True.csv
```

If the two CSV files are not present, the script uses the bundled balanced classroom demonstration corpus. The deployed website uses this bundled corpus so that the demonstration works without an external database.

## Important limitation

The classifier identifies language patterns learned from labelled examples. Its result is not independent fact verification. Important claims should still be checked against reliable primary sources.

