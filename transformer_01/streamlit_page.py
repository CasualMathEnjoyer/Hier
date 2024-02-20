import streamlit as st
import pandas as pd
import pickle
# streamlit run your_script.py

model_file_name = "model_to_delete"

with open(model_file_name + '_HistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)


keys = list(history)
ar = []

st.header("The English Model")

genre = st.radio(
    "Select metrics",
    ["loss", "accuracy", "precision&recall", "F1 score"])
dict = {
    "loss" : 0,
    "accuracy" : 1,
    "precision&recall" : 2,
    "F1 score" : 3
}

for i in range(len(history[keys[dict[genre]]])):
    item = [history[keys[dict[genre]]][i], history[keys[dict[genre]+5]][i]]
    ar.append(item)
chart_data = pd.DataFrame(ar, columns=[genre, "val_" + genre])


#st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])
# st.line_chart(history[keys[1]])