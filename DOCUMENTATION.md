# Documentation

## Files

### `train.py`
Trains the machine learning model end-to-end.

It performs:
- data loading
- cleaning
- duplicate removal
- missing-value handling
- target detection
- label encoding
- stratified train-test split
- SMOTE balancing
- feature preprocessing
- RandomForest training
- evaluation
- cross-validation
- artifact saving

### `app.py`
A Streamlit interface that:
- loads the saved model and encoder
- builds input fields for the features
- predicts the disease name
- shows a clean prediction result

### `requirements.txt`
Lists the Python packages required to run the project.

### `README.md`
Main project instructions:
- problem statement
- dataset information
- how to run the code
- model details
- output artifacts
- expected project structure

### `metrics.json`
Saved training summary containing:
- target column
- feature columns
- accuracy
- per-class metrics
- mean cross-validation F1

### `confusion_matrix.png`
A saved visualization of the classification confusion matrix.

## Important Submission Note

The assignment requires the Kaggle CSV dataset. If it is not already in the repository, place it as `dataset.csv` in the project root before running `train.py`.

## Suggested GitHub Commit History

1. `initial project structure`
2. `added data cleaning and preprocessing`
3. `added model training and evaluation`
4. `added Streamlit app`
5. `added README and documentation`
