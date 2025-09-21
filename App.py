import streamlit as st
import pandas as pd
import numpy as np
import sklearn, joblib, sys
from pathlib import Path
from joblib import load

st.write({
    "python": sys.version,
    "sklearn": sklearn.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "joblib": joblib.__version__
})

@st.cache_resource
def load_model():
    model_path = Path(__file__).with_name("titanic_logreg_pipeline.pkl")
    try:
        return load(model_path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

clf = load_model()
