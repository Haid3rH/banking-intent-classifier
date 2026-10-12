# Banking Intent Classifier

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://banking-intent-classifier-haid3rh.streamlit.app/)

A traditional NLP machine-learning project that classifies a customer banking message into one of **77 intent categories**. The application combines word-level and character-level TF-IDF features with a Linear Support Vector Machine (Linear SVM), served through a dark-themed Streamlit interface.

**Live app:** https://banking-intent-classifier-haid3rh.streamlit.app/  
**Source code:** https://github.com/Haid3rH/banking-intent-classifier  
**Developer:** [Haid3rH](https://github.com/Haid3rH)

---

## Results

The final pipeline was selected using the validation set, then retrained on all 10,003 labeled training examples. The held-out test set was used for the final evaluation.

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | **91.71%** | **90.81%** |
| Macro F1 | **91.81%** | **90.83%** |
| Weighted F1 | **91.71%** | **90.83%** |

The test split contains 3,080 examples across 77 intents. The test results are reported as the final evaluation and were not used to tune the model.

### Feature experiments

| Feature representation | Validation accuracy | Validation Macro F1 |
|---|---:|---:|
| Word TF-IDF + Linear SVM | 89.01% | 89.12% |
| Character TF-IDF + Linear SVM (tuned) | 91.51% | 91.43% |
| **Word + Character TF-IDF + Linear SVM** | **91.71%** | **91.81%** |

The combined representation was the best candidate by validation Macro F1. A small cross-validated parameter search selected the same combined feature settings already used by the baseline combined pipeline, so tuning did not change the final validation score.

## What the app does

- Accepts a free-text banking customer message and predicts its intent.
- Shows the predicted intent in a readable format, alongside the raw class label.
- Lists the top related intents using Linear SVM decision scores.
- Provides example messages for quick demonstrations.
- Keeps a temporary prediction history for the current app session, with CSV export and clear-history controls.
- Includes model-performance and error-analysis sections to make the experimentation and limitations visible.
- Uses a custom dark navy and blue theme.

> **Score note:** Linear SVM decision scores are used to rank related intents. They are not calibrated probabilities, so the app does not present them as confidence percentages.

## Model and methodology

The final pipeline uses:

- **Word TF-IDF:** word n-grams `(1, 2)`
- **Character TF-IDF:** character n-grams `(3, 5)`
- **Vectorizer settings:** `min_df=2`, `sublinear_tf=True`
- **Feature combination:** scikit-learn `FeatureUnion`
- **Classifier:** `LinearSVC(C=1)`

### Prediction flow

```text
Customer message
      │
      ├── Word TF-IDF (1–2 grams)
      │
      └── Character TF-IDF (3–5 grams)
                    │
           Combined feature matrix
                    │
              Linear SVM
                    │
             Predicted intent
```

### Development process

1. Explored the labeled data and inspected text, labels, and data quality.
2. Compared seven traditional machine-learning baselines using training and validation metrics.
3. Tracked accuracy, Macro F1, weighted F1, training time, prediction time, and train/validation gaps.
4. Compared word-level, character-level, and combined TF-IDF representations.
5. Used validation Macro F1 as the primary selection metric and performed a small cross-validated hyperparameter search.
6. Inspected per-intent metrics and common validation confusion pairs.
7. Retrained the selected pipeline on all 10,003 labeled training examples.
8. Evaluated the final model once on the held-out 3,080-example test set and saved the pipeline as a Joblib file.

Macro F1 was especially useful because it gives each intent equal weight, rather than allowing more frequent or easier examples to dominate the score.

## Error analysis and limitations

Some of the hardest intents are semantically close. On the validation set, lower-F1 examples included:

| Intent | Validation F1 | Support |
|---|---:|---:|
| `topping_up_by_card` | 0.737 | 10 |
| `card_acceptance` | 0.769 | 6 |
| `top_up_reverted` | 0.774 | 15 |
| `top_up_failed` | 0.786 | 15 |

Frequent confusion pairs included:

- `balance_not_updated_after_bank_transfer` → `pending_transfer`
- `direct_debit_payment_not_recognised` → `card_payment_not_recognised`
- `top_up_failed` → `pending_top_up`
- `country_support` → `card_acceptance`
- `why_verify_identity` → `unable_to_verify_identity`

These errors show that the model can understand the general banking topic while occasionally missing the more specific intent. Per-class results should be interpreted alongside their support counts, since a small validation sample can make class metrics more variable.

Other limitations:

- The model predicts one of the 77 known labels; it does not generate answers or resolve banking issues.
- SVM decision scores are ranking signals, not probability estimates.
- Prediction history is stored only in Streamlit session state and is temporary; there is no persistent database.
- Performance may differ on customer messages that are unlike the dataset examples, contain multiple intents, or fall outside the 77 supported categories.

## Repository structure

```text
banking-intent-classifier/
├── .streamlit/
│   └── config.toml
├── app/
│   └── app.py
├── data/
│   ├── train.csv
│   └── test.csv
├── model/
│   └── banking_intent_classifier.joblib
├── src/
│   └── banking_intent.ipynb
├── .gitignore
└── requirements.txt
```

The `.venv/` local virtual environment is intentionally excluded from version control. The deployed app loads the saved model from `model/banking_intent_classifier.joblib`.

## Run locally

Tested in a Python 3.13.2 environment.

### 1. Clone the repository

```bash
git clone https://github.com/Haid3rH/banking-intent-classifier.git
cd banking-intent-classifier
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the app

Run this command from the repository root:

```bash
streamlit run app/app.py
```

## Dataset and attribution

This project uses the **BANKING77** dataset, a banking-domain intent-classification dataset containing customer-service queries labelled with 77 fine-grained intents.

- Original dataset repository: [PolyAI task-specific datasets — banking data](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)
- Research paper: [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807) by Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić (2020).
- Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See the source dataset for attribution and license terms. The dataset license is separate from the license that applies to this project's code.

## License

See the repository's [`LICENSE`](LICENSE) file for the code license. The BANKING77 dataset retains its own license and attribution requirements as described above.

---

Built by [Haid3rH](https://github.com/Haid3rH) as an applied NLP and machine-learning portfolio project.
