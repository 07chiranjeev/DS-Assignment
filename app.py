# app.py
# Robust Streamlit app: loads saved model if present; otherwise trains on local Titanic_train.csv (or remote fallback).

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor 🚢")

# Versions confirm dependencies installed on Cloud
try:
    import sklearn, joblib
    st.caption({"python": sys.version, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__})
except Exception:
    st.caption({"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__})

MODEL_NAME = "titanic_logreg_pipeline.pkl"
MODEL_PATH = Path(__file__).with_name(MODEL_NAME)
LOCAL_TRAIN = Path(__file__).with_name("Titanic_train.csv")
REMOTE_TRAIN = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

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
    return mapping.get(t, "Unknown")

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Title" not in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
    if "FamilySize" not in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    if "IsAlone" not in df.columns:
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

def build_pipeline() -> Pipeline:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(categories=CAT_CATEGORIES, drop="first", handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
    clf = Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=1000, solver="liblinear"))])
    return clf

@st.cache_resource(show_spinner=False)
def get_or_train_model() -> Pipeline:
    # Load saved model if present
    if MODEL_PATH.exists():
        return load(MODEL_PATH)
    # Otherwise pick local CSV if present, else remote fallback
    if LOCAL_TRAIN.exists():
        df = pd.read_csv(LOCAL_TRAIN)
    else:
        df = pd.read_csv(REMOTE_TRAIN)
    df = add_engineered(df)
    missing = set(FEATURES + ["Survived"]) - set(df.columns)
    if missing:
        st.error(f"Training data missing columns: {sorted(missing)}")
        st.stop()
    X = df[FEATURES]
    y = df["Survived"].astype(int)
    clf = build_pipeline()
    clf.fit(X, y)
    # Save when writable
    try:
        dump(clf, MODEL_PATH)
    except Exception:
        pass
    return clf

with st.status("Preparing model...", expanded=False):
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

# Show an immediate demo prediction so the page always shows output
proba_demo = float(clf.predict_proba(X_input)[:, 1][0])
pred_demo = int(proba_demo >= 0.5)
st.metric("Survival probability", f"{proba_demo:.3f}")
st.metric("Predicted Survived (0/1)", f"{pred_demo}")

# Optional explicit button for re-run clarity
if st.button("Predict"):
    proba = float(clf.predict_proba(X_input)[:, 1][0])
    pred = int(proba >= 0.5)
    st.metric("Survival probability", f"{proba:.3f}")
    st.metric("Predicted Survived (0/1)", f"{pred}")
    st.caption("Pipeline: imputation, scaling, one-hot encoding (drop='first'), logistic regression.")
