# app.py
import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor 🚢")

@st.cache_resource
def load_model():
    return load("titanic_logreg_pipeline.pkl")

clf = load_model()

st.subheader("Passenger inputs")
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"], index=0)
    age = st.number_input("Age", 0.0, 100.0, 29.0, 1.0)
    embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)
with col2:
    sibsp = st.number_input("Siblings/Spouses (SibSp)", 0, 10, 0, 1)
    parch = st.number_input("Parents/Children (Parch)", 0, 10, 0, 1)
    fare = st.number_input("Fare", 0.0, 600.0, 32.2, 0.1)
    title = st.selectbox("Title", ["Mr","Mrs","Miss","Master","Officer","Royal"], index=0)

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
st.caption(f"Computed features — FamilySize: {family_size}, IsAlone: {is_alone}")

X_input = pd.DataFrame([{
    "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
    "Fare": fare, "Embarked": embarked, "Title": title,
    "FamilySize": family_size, "IsAlone": is_alone
}])

if st.button("Predict"):
    proba = float(clf.predict_proba(X_input)[:, 1][0])
    pred = int(proba >= 0.5)
    st.metric("Survival probability", f"{proba:.3f}")
    st.metric("Predicted Survived (0/1)", f"{pred}")
    st.caption("Model: Logistic Regression + imputation, scaling, one-hot encoding.")
