import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Agent Shopping Simulator", layout="wide")

CSV_PATH = "concat_df.csv"  # change if needed

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    # normalize types
    for col in ['price', 'rating', 'rating_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'selected' in df.columns:
        df['selected'] = df['selected'].astype(int)
    return df

df = load_data()

if df.empty:
    st.error("Data not loaded. Check CSV_PATH.")
    st.stop()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.title("Simulator Controls")

# Filters for query, model, prompt
queries = sorted(df['query'].dropna().unique().tolist())
query_selected = st.sidebar.selectbox("Product Category", queries)

models = sorted(df['model_name_dir'].dropna().unique().tolist())
model_selected = st.sidebar.selectbox("VLM Model", models)

prompts = sorted(df['prompt_type'].dropna().unique().tolist())
prompt_selected = st.sidebar.selectbox("Prompt Type", prompts)

# -------------------------------
# FILTER THE DATA
# -------------------------------
subset = df[
    (df['query'] == query_selected)
    & (df['model_name_dir'] == model_selected)
    & (df['prompt_type'] == prompt_selected)
].copy()

if subset.empty:
    st.warning("No rows found for this combination.")
    st.stop()

# find experiment (there should be one experiment per condition, but show dropdown if multiple)
exps = sorted(subset['experiment_number'].unique())
exp_selected = st.sidebar.selectbox("Experiment Number", exps)
exp_df = subset[subset['experiment_number'] == exp_selected].sort_values(by='position_in_experiment')

# -------------------------------
# DISPLAY
# -------------------------------
st.title("🎯 Agent Shopping Simulator")
st.markdown(f"""
**Model:** {model_selected}  
**Prompt:** {prompt_selected}  
**Category:** {query_selected}  
**Experiment #:** {exp_selected}
""")

cols = st.columns(4)
for i, (_, row) in enumerate(exp_df.iterrows()):
    with cols[i % 4]:
        # image
        if isinstance(row.get("image_url"), str) and row["image_url"].startswith("http"):
            try:
                img = Image.open(BytesIO(requests.get(row["image_url"], timeout=5).content))
                st.image(img, width="stretch")
            except:
                st.image("https://via.placeholder.com/150", width="stretch")
        else:
            st.image("https://via.placeholder.com/150", width="stretch")

        st.markdown(f"**{row['title']}**")
        st.write(f"💲${row['price']:.2f} | ⭐ {row['rating']:.1f} | 🧾 {int(row['rating_count'])} reviews")

        # Highlight if selected
        if row.get("selected", 0) == 1:
            st.success("✅ Selected by the Agent")

st.divider()

# -------------------------------
# SUMMARY
# -------------------------------
chosen = exp_df[exp_df['selected'] == 1]
if not chosen.empty:
    chosen = chosen.iloc[0]
    st.subheader("Agent's Choice")
    st.write(f"**{chosen['title']}**")
    if isinstance(chosen.get("image_url"), str) and chosen["image_url"].startswith("http"):
        try:
            img = Image.open(BytesIO(requests.get(chosen["image_url"], timeout=5).content))
            st.image(img, width=250)
        except:
            st.image("https://via.placeholder.com/150", width=250)
    st.write(f"💲${chosen['price']:.2f} | ⭐ {chosen['rating']:.1f} | 🧾 {int(chosen['rating_count'])} reviews")
else:
    st.info("No selected product recorded for this experiment.")
