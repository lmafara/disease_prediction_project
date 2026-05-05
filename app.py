"""
app.py
Streamlit app for disease prediction.

Loads:
- model.pkl
- label_encoder.pkl
- preprocessor.pkl
- metrics.json

Then lets the user enter symptom values and predicts the disease label.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path(".")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    preprocessor = joblib.load(ARTIFACT_DIR / "preprocessor.pkl")
    import json
    metadata = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8")) if (ARTIFACT_DIR / "metrics.json").exists() else None
    return model, encoder, preprocessor, metadata


def get_default_value(dtype_kind: str):
    if dtype_kind == "numeric":
        return 0.0
    return ""


def main():
    st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")
    st.title("Disease Prediction System")
    st.write("Enter the symptoms or patient details below and click **Predict Disease**.")

    if not (ARTIFACT_DIR / "model.pkl").exists():
        st.error(
            f"Missing artifacts in the project root. Run `python train.py --data dataset.csv` first."
        )
        st.stop()

    model, encoder, preprocessor, metadata = load_artifacts()

    feature_columns = metadata["feature_columns"] if metadata and "feature_columns" in metadata else []
    numeric_columns = set(metadata["numeric_columns"]) if metadata and "numeric_columns" in metadata else set()
    categorical_columns = set(metadata["categorical_columns"]) if metadata and "categorical_columns" in metadata else set()

    st.sidebar.header("Model Information")
    if metadata:
        st.sidebar.metric("Accuracy", f"{metadata.get('accuracy', 0):.3f}")
        st.sidebar.metric("Mean CV F1", f"{metadata.get('macro_f1_cv_mean', 0):.3f}")

    user_input = {}

    if not feature_columns:
        st.warning("Feature metadata not found. The app cannot build inputs automatically.")
        st.stop()

    st.subheader("Patient Input")

    cols = st.columns(2)
    for idx, feature in enumerate(feature_columns):
        with cols[idx % 2]:
            label = feature.replace("_", " ").title()

            if feature in numeric_columns:
                user_input[feature] = st.number_input(label, value=0.0, step=0.1, format="%.4f")
            elif feature in categorical_columns:
                # Generic categorical entry
                user_input[feature] = st.text_input(label, value="")
            else:
                # Fallback: let the user type any value
                user_input[feature] = st.text_input(label, value="")

    if st.button("Predict Disease"):
        input_df = pd.DataFrame([user_input], columns=feature_columns)

        # Try to match numeric columns by converting them
        for col in feature_columns:
            if col in numeric_columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        try:
            transformed = preprocessor.transform(input_df)
            prediction = model.predict(transformed)[0]
            disease = encoder.inverse_transform([prediction])[0]

            st.success(f"Predicted Disease: **{str(disease).title()}**")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(transformed)[0]
                top_idx = proba.argsort()[::-1][:5]
                top_classes = [encoder.inverse_transform([i])[0] for i in top_idx]
                top_probs = [proba[i] for i in top_idx]

                st.subheader("Top Prediction Probabilities")
                prob_df = pd.DataFrame(
                    {
                        "Disease": [str(x).title() for x in top_classes],
                        "Probability": top_probs,
                    }
                )
                st.dataframe(prob_df, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
