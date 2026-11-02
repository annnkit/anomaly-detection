# Credit Card Fraud Detection

A machine learning project that explores credit card transaction fraud detection using **Logistic Regression** and **Isolation Forest**. The workflow includes exploratory data analysis, random undersampling to address class imbalance, feature scaling, model training, and evaluation.

## Project Overview

The dataset contains **284,807 transactions** and 31 columns:

- `Time`: elapsed time information for a transaction
- `V1`–`V28`: anonymized numerical transaction features
- `Amount`: transaction amount
- `Class`: target label (`0` = legitimate, `1` = fraudulent)

The original dataset is highly imbalanced:

| Transaction type | Count |
|---|---:|
| Legitimate (`Class = 0`) | 284,315 |
| Fraudulent (`Class = 1`) | 492 |
| **Total** | **284,807** |

No missing values were found in the supplied dataset.

## Approach

### 1. Exploratory Data Analysis

The notebook/script examines the dataset structure, missing values, class distribution, transaction amounts by class, and class-wise feature means.

### 2. Handling Class Imbalance

Random undersampling is applied to the legitimate transactions. The code selects 492 legitimate transactions and combines them with all 492 fraudulent transactions, producing a balanced dataset of 984 rows.

### 3. Train-Test Split

The balanced dataset is split into training and testing sets:

- Test size: 20%
- Stratification: enabled (`stratify=y`)
- Random state: `2`

### 4. Preprocessing

A `StandardScaler` is applied to the input features through a `ColumnTransformer`. Preprocessing and model training are combined in scikit-learn pipelines.

### 5. Models

**Logistic Regression**

- Trained as a supervised binary classification model.
- Evaluated on both training and testing data.
- Metrics printed by the script: accuracy, classification report, and confusion matrix.

**Isolation Forest**

- Evaluated as an unsupervised anomaly-detection model.
- Configuration: `n_estimators=100`, `contamination=0.01`, `random_state=42`.
- Its predictions (`-1` for anomaly and `1` for inlier) are converted to fraud labels (`1` for anomaly and `0` for inlier).
- A classification report is printed for training and testing data.

The script concludes by selecting Logistic Regression as the final model. No numeric evaluation results are included in the source script, so this README does not claim a specific accuracy or other performance score.

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Project Structure

```text
.
├── anomaly_detection.py
├── creditcard.csv
└── README.md
```

Place the dataset in the same directory as the Python script and name it `creditcard.csv`. The script currently reads the dataset using:

```python
pd.read_csv("creditcard.csv")
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Add the dataset

Download or obtain the credit card fraud dataset and place it in the project directory as `creditcard.csv`.

### 5. Run the project

```bash
python anomaly_detection.py
```

The script prints the classification metrics and confusion matrices in the terminal.

## Notes and Limitations

- Random undersampling discards many legitimate transactions. The balanced sample is useful for experimentation, but it does not preserve the original class distribution.
- Evaluation on the undersampled test set does not directly represent performance on real-world transactions, where fraud is rare.
- The Isolation Forest contamination value is set to `0.01`, while the undersampled dataset contains a 50% fraud share. This setting may affect its predictions and should be revisited when evaluating the model.
- The source code prints evaluation results but does not save a trained model or provide an API, dashboard, or deployment workflow.
- The final model choice reflects the conclusion written in the source code; model performance should be compared using the printed test metrics and, ideally, precision, recall, F1-score, and PR-AUC for the fraud class.

## Future Improvements

- Evaluate models on a test set that retains the original class distribution.
- Compare random undersampling with other imbalance-handling techniques.
- Tune model hyperparameters using validation data or cross-validation.
- Evaluate fraud detection with precision, recall, F1-score, and precision-recall AUC.
- Save the fitted preprocessing and model pipeline for reuse.
- Add a prediction interface or API for new transactions.

## Disclaimer

This project is for educational and experimentation purposes. It is not a production-ready fraud detection system and should not be used to make real financial decisions.
