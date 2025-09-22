# app.py
# Streamlit app that loads a saved model if present; otherwise trains a fresh pipeline on Titanic data.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Import ML libs after Streamlit to ensure clearer error logs if packages are missing
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor 🚢")

# Show versions to confirm environment on Cloud
st.caption({
    "python": sys.version,
    # versions show only after scikit-learn is installed by Cloud from requirements.txt
})

MODEL_NAME = "titanic_logreg_pipeline.pkl"
MODEL_PATH = Path(__file__).with_name(MODEL_NAME)
LOCAL_TRAIN = Path(__file__).with_name("Titanic_train.csv")
REMOTE_TRAIN = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Feature config
FEATURES = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","FamilySize","IsAlone"]
NUMERIC = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone"]
CATEGORICAL = ["Pclass","Sex","Embarked","Title"]
CAT_CATEGORIES = [
    [1, 2, 3],                               # Pclass
    ["male", "female"],                      # Sex
    ["S", "C", "Q"],                         # Embarked
    ["Mr","Mrs","Miss","Master","Officer","Royal","Unknown"]  # Title
]

def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    t = name.split(",")[1].split(".")[0].strip()
    mapping = {
        "Mr":"Mr","Mrs":"Mrs","Miss":"Miss","Master":"Master",
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
        "Lady":"Royal","Countess":"Royal","Sir":"Royal","Don":"Royal","Dona":"Royal","Jonkheer":"Royal",
        "Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer"
    }
    return mapping.get(t, "Unknown")  # collapse all others

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

def build_pipeline() -> Pipeline:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=CAT_CATEGORIES, drop="first", handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", cat, CATEGORICAL)
    ])
    clf = Pipeline([
        ("preprocess", pre),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])
    return clf

@st.cache_resource
def get_or_train_model() -> Pipeline:
    # 1) Load if model file exists
    if MODEL_PATH.exists():
        return load(MODEL_PATH)

    # 2) Else train from local CSV if present
    if LOCAL_TRAIN.exists():
        train = pd.read_csv(LOCAL_TRAIN)
    else:
        # 3) Else fallback to remote public Titanic CSV (same schema)
        train = pd.read_csv(REMOTE_TRAIN)

    train = add_engineered(train)

    # Ensure all required columns exist (Name must be present for Title)
    missing = set(FEATURES + ["Survived"]) - set(train.columns)
    if missing:
        st.error(f"Training data missing columns: {sorted(missing)}")
        st.stop()

    X = train[FEATURES]
    y = train["Survived"].astype(int)

    clf = build_pipeline()
    clf.fit(X, y)

    # Save for next runs
    try:
        dump(clf, MODEL_PATH)
    except Exception:
        # In read-only environments, skip saving
        pass

    return clf

clf = get_or_train_model()

st.subheader("Passenger inputs")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1,2,3], index=2)
    sex = st.selectbox("Sex", ["male","female"], index=0)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
    embarked = st.selectbox("Embarked", ["S","C","Q"], index=0)
with col2:
    sibsp = st.number_input("Siblings/Spouses (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children (Parch)", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.2, step=0.1)
    title = st.selectbox("Title", ["Mr","Mrs","Miss","Master","Officer","Royal","Unknown"], index=0)

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
st.caption(f"Computed — FamilySize: {family_size}, IsAlone: {is_alone}")

row = {
    "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
    "Fare": fare, "Embarked": embarked, "Title": title,
    "FamilySize": family_size, "IsAlone": is_alone
}
X_input = pd.DataFrame([row])

if st.button("Predict"):
    proba = float(clf.predict_proba(X_input)[:, 1][0])
    pred = int(proba >= 0.5)
    st.metric("Survival probability", f"{proba:.3f}")
    st.metric("Predicted Survived (0/1)", f"{pred}")
    st.caption("Pipeline: imputation, scaling, one-hot encoding (drop='first'), logistic regression.")
