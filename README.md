# Bank Marketing: Term Deposit Prediction

Predicts whether a bank client will subscribe to a term deposit, and compares five classifiers on the same data: logistic regression, decision tree, random forest, SVM, and XGBoost.

The focus of this project is an evaluation that holds up: no information from the test set reaches the model before the final evaluation.

## Dataset

Bank marketing data in the UCI Bank Marketing format (client details, previous campaign outcomes, and a yes/no target `y`). Download `bank.csv` from the UCI Machine Learning Repository (search "Bank Marketing") and place it in this folder. The data file is not included in this repository.

## How it works

1. **Load and clean.** The CSV separator is detected automatically. The target is mapped to 0 and 1, and categorical columns are one-hot encoded.
2. **Split first.** A stratified 80/20 train/test split happens before any resampling or scaling.
3. **Pipeline per model.** Each model sits in a pipeline of SMOTE, then standard scaling, then the classifier. Because SMOTE is inside the pipeline, it only creates synthetic samples from the training portion of each cross-validation fold.
4. **Tune and compare.** Every model is tuned with 5-fold stratified `GridSearchCV`, scored by ROC-AUC.
5. **Choose without peeking.** The best model is picked by cross-validated ROC-AUC. The held-out test set is used once to report accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
6. **Tune the decision threshold.** The default 0.5 cutoff is a poor choice for a rare outcome. The threshold is chosen from out-of-fold predictions on the training data, by maximising F1, or expected net value if a cost per call and a value per subscriber are given. The test set then reports what that threshold achieves next to 0.5.
7. **Rank clients with a lift table.** Clients in the test set are ranked by predicted probability, and the table shows what share of all subscribers is captured by contacting the top 10%, 20%, and so on.
8. **Save.** The fitted best pipeline, the chosen threshold, comparison tables, a lift table, and charts are written to `results/`.

## Design decisions

- **No leakage from resampling.** Applying SMOTE before the train/test split lets synthetic points built from neighbours leak across the split and inflates the scores. Here it is fitted on training folds only.
- **`duration` is dropped by default.** Call duration is only known after the call has ended, so a model using it cannot decide whom to call. Use `--keep-duration` to include it and see how much it changes the scores.
- **`pdays == 999` is kept as information.** In this dataset 999 means "not previously contacted". It becomes a `previously_contacted` flag, and `pdays` is set to -1 for those rows, instead of being treated as 999 days.
- **Precision, recall, and F1 are reported alongside accuracy**, because the classes are heavily imbalanced and accuracy alone is misleading.

## Run it

```bash
pip install -r requirements.txt
python bank_marketing_model.py --data bank.csv
```

Options:

| Flag | Purpose |
|---|---|
| `--data` | Path to the CSV file (default `bank.csv`) |
| `--output-dir` | Folder for outputs (default `results`) |
| `--keep-duration` | Keep the `duration` column |
| `--skip-svm` | Skip the SVM, which is slow on the full 45,000-row dataset |
| `--call-cost` | Cost of contacting one client. Use with `--subscriber-value` |
| `--subscriber-value` | Value of one client who subscribes. With both flags, the threshold maximises expected net value instead of F1 |

## Outputs

- `results/model_comparison.csv`: cross-validated ROC-AUC and test metrics for every model
- `results/model_comparison.png`: test ROC-AUC and F1 by model
- `results/confusion_matrix.png`: confusion matrix for the best model at the default 0.5 threshold
- `results/threshold_comparison.csv`: test-set metrics at the default and the tuned threshold
- `results/lift_table.csv`: share of subscribers captured at each contact depth, with lift
- `results/threshold_and_gains.png`: precision-recall curve with the chosen threshold, and the cumulative gains chart
- `results/decision_threshold.json`: the chosen threshold, to apply as `predict_proba(X)[:, 1] >= threshold`
- `results/best_model_*.joblib`: the fitted best pipeline, ready to load with `joblib.load` and call `predict_proba` on new rows

## Results

Held-out test set: 905 clients (stratified 20% split), of whom 11.5% subscribed. Models are ranked by cross-validated ROC-AUC on the training data, and the test set was used once, at the end.

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Precision (yes) | Recall (yes) | F1 (yes) |
|---|---|---|---|---|---|---|
| XGBoost | 0.753 | 0.757 | 0.890 | 0.559 | 0.183 | 0.275 |
| Random Forest | 0.745 | 0.725 | 0.884 | 0.480 | 0.115 | 0.186 |
| Logistic Regression | 0.722 | 0.727 | 0.713 | 0.219 | 0.587 | 0.319 |
| SVM | 0.689 | 0.694 | 0.865 | 0.378 | 0.269 | 0.315 |
| Decision Tree | 0.656 | 0.672 | 0.886 | 0.512 | 0.202 | 0.290 |

![Model comparison](results/model_comparison.png)

How to read this:

- **XGBoost is the best model by ROC-AUC.** Its cross-validated score (0.753) and test score (0.757) agree closely, which is what you expect when no information leaks from the test set into training.
- **Accuracy overstates performance.** Predicting "no" for every client scores 0.885 on this test set, so 0.890 is barely above that baseline. ROC-AUC and the precision and recall of the "yes" class are more informative.
- **The default 0.5 threshold misses most subscribers.** XGBoost finds 18% of subscribers at 56% precision. Logistic regression finds 59% of them at 22% precision. The right trade-off depends on the cost of a missed subscriber against the cost of a wasted call.
- **`duration` is excluded**, which matches the real setting where the call has not happened yet. Including it usually inflates scores on this dataset, but the resulting model could not be used to decide whom to call.

### Threshold tuning and lift

The threshold was chosen on out-of-fold predictions from the training data by maximising F1, giving 0.219. The figures below are on the test set, which played no part in choosing it.

| Setting | Threshold | Clients contacted | Subscribers found | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Default | 0.500 | 34 | 19 | 0.559 | 0.183 | 0.275 |
| Tuned by F1 | 0.219 | 141 | 49 | 0.348 | 0.471 | 0.400 |

Contacting clients in order of predicted probability:

| Share of clients contacted | Subscribers captured | Share of all subscribers | Lift over random |
|---|---|---|---|
| 10% | 35 | 33.7% | 3.4 |
| 20% | 59 | 56.7% | 2.8 |
| 30% | 67 | 64.4% | 2.1 |
| 50% | 83 | 79.8% | 1.6 |

- **Recall rises from 18% to 47%** with the tuned threshold, at the cost of precision, which falls from 56% to 35%. F1 improves from 0.275 to 0.400. Accuracy falls from 0.890 to 0.838, which is expected and not the goal for a rare outcome.
- **Lift is the most useful business view.** Contacting the top 20% of clients by score reaches 57% of subscribers, 2.8 times what random calling would reach. Half the list reaches about 80% of subscribers.
- **The ranking is meaningfully better than chance.** PR-AUC for XGBoost is 0.347 against a baseline of 0.115, the share of subscribers in the test set.
![Precision-recall trade-off and cumulative gains](results/threshold_and_gains.png)

- **Treat small differences with care.** The test set contains only 104 subscribers, so these figures carry noticeable sampling noise.

## Limitations and next steps

- SMOTE interpolates between rows, which is crude for one-hot encoded columns. `SMOTENC`, or class weights instead of resampling, would be worth comparing.
- The cost per call and value per subscriber are inputs you supply. The data does not contain them, so any net-value figure is only as good as those assumptions.
- Add model explanations (feature importance or SHAP) and a calibration check on the predicted probabilities.
