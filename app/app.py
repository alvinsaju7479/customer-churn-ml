import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_RF = ROOT / "artifacts" / "models" / "churn_model_rf.joblib"
MODEL_LR = ROOT / "artifacts" / "models" / "churn_model_logreg.joblib"

MODEL_PATH = MODEL_RF if MODEL_RF.exists() else MODEL_LR

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📉 Customer Churn Predictor")
st.caption("Predict churn probability using a trained ML pipeline (preprocess + model).")

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run notebooks 01–03 to generate artifacts/models/*.joblib.\n\n"
        f"Expected one of:\n- {MODEL_RF}\n- {MODEL_LR}"
    )
    st.stop()

model = load_model(MODEL_PATH)
st.success(f"Loaded model: `{MODEL_PATH.name}`")


tab1, tab2 = st.tabs(["⬆️ Upload CSV", " Manual Input"])

with tab1:
    st.subheader("Upload CSV")
    st.write("Upload a CSV with the same columns used during training (excluding the target).")
    file = st.file_uploader("Choose a CSV file", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Predict for uploaded data"):
            try:
                
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df)[:, 1]
                    preds = (proba >= 0.5).astype(int)
                    out = df.copy()
                    out["churn_probability"] = proba
                    out["churn_prediction"] = preds
                else:
                    preds = model.predict(df)
                    out = df.copy()
                    out["churn_prediction"] = preds

                st.success("Predictions generated ✅")
                st.dataframe(out.head(50), use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Manual input (1 customer)")
    st.write("Fill only the fields you have. Missing columns will be left blank (may reduce accuracy).")
    cols = None
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
    except Exception:
        pass

    if not cols:
        st.warning("Could not infer feature columns. Use CSV upload tab for best results.")
    else:
        input_data = {}
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(cols):
            if i % 3 == 0:
                input_data[col] = c1.text_input(col, "")
            elif i % 3 == 1:
                input_data[col] = c2.text_input(col, "")
            else:
                input_data[col] = c3.text_input(col, "")

        if st.button("Predict churn"):
            df_one = pd.DataFrame([input_data])
            for c in df_one.columns:
                df_one[c] = pd.to_numeric(df_one[c], errors="ignore")

            try:
                if hasattr(model, "predict_proba"):
                    proba = float(model.predict_proba(df_one)[:, 1][0])
                    pred = int(proba >= 0.5)
                    st.metric("Churn probability", f"{proba:.3f}")
                    st.metric("Prediction (1=Churn)", pred)
                else:
                    pred = int(model.predict(df_one)[0])
                    st.metric("Prediction (1=Churn)", pred)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Tip: For best results, use the same columns as training. See the README for example input format.")
