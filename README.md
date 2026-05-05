# Disease Prediction System (ML + GUI)

A complete machine learning project that predicts a disease from patient symptoms and exposes the model through a Streamlit web app.

## Problem Statement

Build an end-to-end disease prediction system using a real dataset, train a machine learning model, evaluate it properly, handle class imbalance, and deploy the final model in a GUI.

## Dataset



## Project Structure

```text
project/
├── train.py
├── app.py
├── model.pkl
├── preprocessor.pkl
├── label_encoder.pkl
├── metrics.json
├── confusion_matrix.png
├── requirements.txt
├── README.md
└── dataset.csv
```

## Pipeline Covered

1. Load data with pandas  
2. Clean column names and string values  
3. Remove duplicates  
4. Handle missing values  
5. Split features and target  
6. Label encode the target  
7. Stratified train-test split  
8. Apply scaling / encoding with preprocessing  
9. Handle imbalance with SMOTE  
10. Train `RandomForestClassifier`  
11. Evaluate with accuracy, precision, recall, F1-score, and confusion matrix  
12. Run Stratified K-Fold cross validation with mean F1-score  
13. Save model artifacts with joblib  
14. Load the trained model in Streamlit for prediction  

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train the model
```bash
python train.py --data dataset.csv
```

This creates in the project root:
- `model.pkl`
- `preprocessor.pkl`
- `label_encoder.pkl`
- `metrics.json`
- `confusion_matrix.png`

### 3) Start the Streamlit app
```bash
streamlit run app.py
```

## Model Used

- **RandomForestClassifier** for final training
- **LabelEncoder** for the target
- **StandardScaler** for numeric feature scaling
- **OrdinalEncoder** for categorical feature handling
- **SMOTE** for class balancing

## Results Summary

After training, the script prints:
- Test accuracy
- Precision / recall / F1-score per class
- Confusion matrix
- Stratified K-Fold mean F1-score

The exact numbers depend on the dataset version and preprocessing results.

## Notes

- The repository must include the real dataset file or a valid dataset link for submission.
- The Streamlit app loads saved artifacts from the project root.
- If your dataset uses a different target column name, the training script tries to detect it automatically.
