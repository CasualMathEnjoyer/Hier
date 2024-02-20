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

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "Enter text"

# input box
text_input = st.text_input(
        "Enter text 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

# todo - load model

# future output box
text_output = st.text("Changed input:" + text_input)

# GRAPHS
genre = st.radio(
    "Select metrics",
    ["loss", "accuracy", "precision", "recall", "F1 score"])

dict = {
    "loss" : 0,
    "accuracy" : 1,
    "precision" : 2,
    "recall" : 3,
    "F1 score" : 4
}

# i do hate the mess bellow
for i in range(len(history[keys[dict[genre]]])):
    item = [history[keys[dict[genre]]][i], history[keys[dict[genre]+5]][i]]
    ar.append(item)
chart_data = pd.DataFrame(ar, columns=[genre, "val_" + genre])


#st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])
# st.line_chart(history[keys[1]])