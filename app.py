
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data import FEATURES


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "xgb_final.joblib"


# ==================================================
# 2. LOAD THE SAVED MODEL
# ==================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ==================================================
# 3. APP HEADER
# ==================================================
st.title("💳 Credit Card Fraud Detection")
st.write(
    "An XGBoost machine learning application that ranks "
    "transactions by their predicted fraud risk."
)

st.info(
    "Educational demonstration only. A high fraud score is "
    "a model prediction, not proof that a transaction is fraudulent."
)

st.caption(
    "Upload transaction data in CSV format to generate fraud-risk scores."
)


# ==================================================
# 4. MODEL INFORMATION
# ==================================================
with st.expander("How does this model work?"):
    st.write(
        """
        - **Model:** XGBoost classifier.
        - **Input:** Transaction features V1–V28, Amount and Time.
        - **Feature engineering:** Amount is log-transformed; Time is
          converted into cyclical hour features.
        - **Output:** A predicted fraud score for each transaction.
        - **Ranking:** Transactions with higher scores appear first.
        """
    )

    st.warning(
        "The Time-derived features are based on elapsed seconds in "
        "the dataset, not verified real-world transaction clock times."
    )


# ==================================================
# 5. CSV UPLOAD
# ==================================================
st.header("1. Upload transactions")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "Use the original credit-card dataset format with Time, "
        "Amount and V1–V28. The Class column is optional."
    ),
)


# ==================================================
# 6. PROCESS DATA AND PREDICT
# ==================================================
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV contains no transactions.")

        else:
            # Original columns needed to reproduce training features
            required_raw = ["Time", "Amount"]

            missing_raw = [
                col for col in required_raw
                if col not in data.columns
            ]

            if missing_raw:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_raw)
                )

            else:
                # Check raw numeric inputs
                for col in ["Time", "Amount"]:
                    data[col] = pd.to_numeric(
                        data[col], errors="coerce"
                    )

                if data[["Time", "Amount"]].isnull().any().any():
                    st.error(
                        "Time and Amount must contain valid numeric values."
                    )

                elif not np.isfinite(
                    data[["Time", "Amount"]].to_numpy(dtype=float)
                ).all():
                    st.error(
                        "Time and Amount cannot contain infinite values."
                    )

                elif (data["Amount"] < 0).any():
                    st.error("Amount cannot be negative.")

                elif (data["Time"] < 0).any():
                    st.error("Time cannot be negative.")

                else:
                    # Reproduce the exact feature engineering used in training
                    data["log_amount"] = np.log1p(data["Amount"])

                    hour = (data["Time"] // 3600) % 24

                    data["hour_sin"] = np.sin(
                        2 * np.pi * hour / 24
                    )
                    data["hour_cos"] = np.cos(
                        2 * np.pi * hour / 24
                    )

                    # Ensure all model features exist
                    missing_features = [
                        col for col in FEATURES
                        if col not in data.columns
                    ]

                    if missing_features:
                        st.error(
                            "Missing model features: "
                            + ", ".join(missing_features)
                        )

                    else:
                        # Select features in the exact training order
                        X = data[FEATURES].copy()

                        # Validate feature values
                        for col in X.columns:
                            X[col] = pd.to_numeric(
                                X[col], errors="coerce"
                            )

                        if X.isnull().any().any():
                            st.error(
                                "Some model features contain missing "
                                "or non-numeric values. Please correct "
                                "the CSV before prediction."
                            )

                        elif not np.isfinite(
                            X.to_numpy(dtype=float)
                        ).all():
                            st.error(
                                "Model features cannot contain infinite "
                                "values."
                            )

                        else:
                            # Load model and predict
                            model = load_model()
                            scores = model.predict_proba(X)[:, 1]

                            # Create result table
                            results = data.copy()
                            results["fraud_score"] = scores

                            st.success(
                                f"Successfully scored {len(results):,} "
                                "transactions."
                            )

                            # --------------------------------------
                            # 7. THRESHOLD
                            # --------------------------------------
                            st.header("2. Choose a review threshold")

                            threshold = st.slider(
                                "Flag transactions with a score at or above:",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.01,
                            )

                            results["flagged_for_review"] = (
                                results["fraud_score"] >= threshold
                            )

                            # --------------------------------------
                            # 8. SUMMARY
                            # --------------------------------------
                            st.header("3. Prediction summary")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Total transactions",
                                f"{len(results):,}",
                            )

                            col2.metric(
                                "Flagged for review",
                                f"{int(results['flagged_for_review'].sum()):,}",
                            )

                            col3.metric(
                                "Highest fraud score",
                                f"{results['fraud_score'].max():.4f}",
                            )

                            st.caption(
                                "The threshold is configurable. It has "
                                "not been established as an optimal "
                                "operational threshold."
                            )

                            # --------------------------------------
                            # 9. RESULTS TABLE
                            # --------------------------------------
                            st.header("4. Scored transactions")

                            sorted_results = results.sort_values(
                                "fraud_score",
                                ascending=False,
                            )

                            st.dataframe(
                                sorted_results,
                                use_container_width=True,
                            )

                            # --------------------------------------
                            # 10. DOWNLOAD RESULTS
                            # --------------------------------------
                            csv_output = results.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                label="⬇️ Download predictions as CSV",
                                data=csv_output,
                                file_name="fraud_predictions.csv",
                                mime="text/csv",
                            )

                            # --------------------------------------
                            # 11. OPTIONAL EVALUATION
                            # --------------------------------------
                            if "Class" in data.columns:

                                st.header(
                                    "5. Evaluation (if true labels are available)"
                                )

                                labels = pd.to_numeric(
                                    data["Class"], errors="coerce"
                                )

                                valid_labels = (
                                    labels.notna()
                                    & labels.isin([0, 1])
                                )

                                if not valid_labels.all():
                                    st.warning(
                                        "Class must contain only 0 or 1 "
                                        "for every row to calculate metrics."
                                    )

                                elif labels.nunique() < 2:
                                    st.warning(
                                        "Evaluation needs both legitimate "
                                        "and fraudulent examples."
                                    )

                                else:
                                    from sklearn.metrics import (
                                        average_precision_score,
                                        precision_score,
                                        recall_score,
                                    )

                                    y_true = labels.astype(int)
                                    y_pred = (
                                        scores >= threshold
                                    ).astype(int)

                                    m1, m2, m3 = st.columns(3)

                                    m1.metric(
                                        "PR-AUC",
                                        f"{average_precision_score(y_true, scores):.4f}",
                                    )

                                    m2.metric(
                                        "Precision",
                                        f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
                                    )

                                    m3.metric(
                                        "Recall",
                                        f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
                                    )

                                    st.caption(
                                        "These metrics are meaningful only "
                                        "when Class contains trustworthy "
                                        "ground-truth labels. For an unbiased "
                                        "final result, use a held-out test set."
                                    )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.caption(
            "Check the uploaded file, model file, and required dependencies."
        )


# ==================================================
# 12. FOOTER
# ==================================================
st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-learn, XGBoost and Streamlit. "
    "Predictions are for educational purposes and should not be used "
    "as the sole basis for financial decisions."
)

