

import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Iris Species Prediction App")

# Load model and features
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

def load_features():
    with open("features.txt", "r") as f:
        features = f.read().strip().split(",")
    return features

model = load_model()
features = load_features()

uploaded_file = st.file_uploader(
    f"Upload your CSV file with columns: {', '.join(features)}",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    if all(col in df.columns for col in features):
        if st.button("Predict"):
            preds = model.predict(df[features])
            results = df.copy()
            results["Predicted_Species"] = preds
            st.write("### Predictions", results.head())

            csv = results.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="iris_predictions.csv",
                mime='text/csv',
            )
    else:
        st.error(f"Your file must have these columns: {features}")
else:
    st.info("Please upload a CSV file.")

st.markdown("---")
st.caption("Built with Streamlit and scikit-learn")
