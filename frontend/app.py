import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap


model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
columns_path = os.path.join(os.path.dirname(__file__), "../model/columns.pkl")

model = pickle.load(open(model_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))


explainer = shap.TreeExplainer(model)


label_map = {
    0: "Flop ",
    1: "Average ",
    2: "Hit "
}

st.set_page_config(page_title="Greenlight Simulator", layout="wide")

st.title(" Greenlight Simulator")
st.write("AI-powered decision system for movie success prediction")


st.sidebar.header(" Movie Inputs")

budget = st.sidebar.slider("Budget ($)", 1_000_000, 300_000_000, 50_000_000)
runtime = st.sidebar.slider("Runtime (minutes)", 60, 180, 120)
release_month = st.sidebar.slider("Release Month", 1, 12, 6)
title = st.sidebar.text_input("Movie Title", "Avengers 2")


keywords = ["2", "3", "II", "III", "Part", "Volume"]
is_franchise = int(any(k.lower() in title.lower() for k in keywords))

def create_input_df(budget_val):
    data = {
        "Budget": budget_val,
        "Runtime": runtime,
        "Release_Month": release_month,
        "Is_Franchise": is_franchise
    }
    df_input = pd.DataFrame([data])

    for col in columns:
        if col not in df_input:
            df_input[col] = 0

    return df_input[columns]

input_df = create_input_df(budget)


if st.button(" Predict"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max()

    st.subheader(" Prediction")
    st.success(label_map[pred])
    st.progress(float(prob))
    st.write(f"Confidence: {prob:.2f}")

    
    st.subheader(" Feature Importance (Top 10)")

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=columns)
    top_feats = feat_imp.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()
    top_feats.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Top Features")

    st.pyplot(fig)

    
    st.subheader(" Why this prediction?")

    shap_values = explainer.shap_values(input_df)

    
    if isinstance(shap_values, list):
        shap_vals = shap_values[pred][0]
    else:
        shap_vals = shap_values[0]

    shap_df = pd.DataFrame({
        "Feature": columns,
        "Impact": shap_vals
    })

    shap_df["abs_impact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["Impact"])
    ax.set_title("Top Feature Impacts")
    st.pyplot(fig)

    
    st.subheader(" ROI vs Budget Simulation")

    budgets = np.linspace(1_000_000, 300_000_000, 20)
    scores = []

    for b in budgets:
        temp_df = create_input_df(b)
        probs = model.predict_proba(temp_df)[0]
        scores.append(probs[2])  # probability of "Hit"

    fig, ax = plt.subplots()
    ax.plot(budgets, scores)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Success Probability (Hit)")
    ax.set_title("Budget vs Success Probability")

    st.pyplot(fig)

    
    st.subheader(" Budget Recommendation")

    best_budget = budgets[np.argmax(scores)]
    best_score = max(scores)

    st.success(f"Recommended Budget: ${int(best_budget):,}")
    st.write(f"Expected Success Probability: {best_score:.2f}")

    if best_budget < budget:
        st.info(" Increasing budget may improve success chances")
    elif best_budget > budget:
        st.warning(" Current budget might be too high")

    
    st.subheader(" Insights")

    if budget > 100_000_000:
        st.warning("High budget increases financial risk")

    if is_franchise:
        st.success("Franchise advantage detected")

    if release_month in [5, 6, 11, 12]:
        st.info("Peak release season")