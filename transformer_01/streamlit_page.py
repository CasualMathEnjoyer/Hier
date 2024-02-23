import streamlit as st
import pandas as pd
import pickle
from transform2bin import load_model_mine, Data
# cd C:\Users\katka\PycharmProjects\Hier\transformer_01
# streamlit run streamlit_page.py
def load_data():
    with open(class_data, 'rb') as inp:
        d = pickle.load(inp)

    d.sep = ' '
    d.space = "_"
    return d

def data(text_input):
    try:
        d = load_data()

        x_test, _ = d.non_slidng_data(text_input, False)
        # print(len(x_test), len(y_test))

        x_valid_tokenized = d.tokenize(x_test)
        prediction = d.model_use(x_valid_tokenized, model_file_name)
        # print(prediction)
        output = d.print_separation(x_test, prediction)
    except Exception as e:
        output = e
    return output

# -----------------------------------------------------------------------------------------------------------------------
model_file_name = "transform2bin"
class_data = model_file_name + "_data.plk"

with open(model_file_name + '_HistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)

keys = list(history)
ar = []
# history
st.header(f"Model: {model_file_name}")
st.caption("POSSIBLE INPUTS:")
st.caption("Aa1 D21 Aa13 Aa1 X1 D54 O4 D21 G43 N5 Z2 Z9 D54 D2 Z1 M22 M22")
st.caption("D37 D37 P5 G43 Z2 D21 D20 Z1 Z2 N41 X1 B1")
st.caption("G17 X1 Z7 B1 D37 X1 N35 I9 T21 D36 Z1 V19 N35 M34 X1 Z4 U10 Z2 G17 X8 Z1 U10 Z2")

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

if st.button('Separate'):
    output = data(text_input)
# future output box
# out_pred = ''
# for item in prediction[0]:
#     if item > 0.5:
#         out_pred += "1"
#     else:
#         out_pred += "0"
# text_output = st.text("Prediction:" + out_pred)
try:
    text_output = st.text("OUTPUT:" + str(output))
except Exception:
    text_output = st.text("OUTPUT:")

st.caption("THE CORRECT OUTPUT")
st.caption("Aa1 D21 Aa13 Aa1 X1 D54 _ O4 D21 G43 N5 Z2 _ Z9 D54 _ D2 Z1 _ M22 M22 _ ")
st.caption("D37 D37 _ P5 G43 Z2 _ D21 _ D20 Z1 Z2 _ N41 X1 B1 _ ")
st.caption("G17 X1 Z7 _ B1 _ D37 X1 _ N35 _ I9 _ T21 D36 Z1 _ V19 _ N35 _ M34 X1 Z4 U10 Z2 _ G17 _ X8 Z1 U10 Z2 _")

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

st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])

@st.cache_data
def get_testing_data():
    # testing stats
    test_file_name = "../data/src-sep-test.txt"
    with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_file = f.read()
        f.close()

    d = load_data()

    x_test, y_test = d.non_slidng_data(test_file[:10000], False)
    x_valid_tokenized = d.tokenize(x_test)
    prediction, metrics = d.model_test(x_valid_tokenized, y_test, model_file_name)
    return metrics

metrics = get_testing_data()
dict = {
    "accuracy  " : metrics[0],
    "precission" : metrics[1],
    "recall    " : metrics[2],
    "F1 score  " : metrics[3]
}

"DATA FROM TESTING"
dict


# ---------------------------------------------------------------------------------------------------------------------
# model_file_name = "transform2bin_focal"
# class_data = "hiero_data_focal.plk"
#
# with open(model_file_name + '_HistoryDict', "rb") as file_pi:
#     history = pickle.load(file_pi)
#
# keys = list(history)
# ar = []
# # history
# st.header("Model 2 - focal")
#
# # Store the initial value of widgets in session state
# if "visibility" not in st.session_state:
#     st.session_state.visibility = "visible"
#     st.session_state.disabled = False
#     st.session_state.placeholder = "Enter text"
#
#
# # input box
# text_input2 = st.text_input(
#     "Enter text 👇",
#     label_visibility=st.session_state.visibility,
#     disabled=st.session_state.disabled,
#     placeholder=st.session_state.placeholder,
#     key = "2.2"
# )
#
# if st.button('Separate', key = "2.3"):
#     output = data(text_input2)
#
# # future output box
# # out_pred = ''
# # for item in prediction[0]:
# #     if item > 0.5:
# #         out_pred += "1"
# #     else:
# #         out_pred += "0"
# # text_output = st.text("Prediction:" + out_pred)
# try:
#     text_output = st.text("OUTPUT:" + str(output))
# except Exception:
#     text_output = st.text("OUTPUT:")
#
# # GRAPHS
# genre = st.radio(
#     "Select metrics",
#     ["loss", "accuracy", "precision", "recall", "F1 score"],
#     key = "2.4")
#
# dict = {
#     "loss": 0,
#     "accuracy": 1,
#     "precision": 2,
#     "recall": 3,
#     "F1 score": 4
# }
#
# # i do hate the mess bellow
# for i in range(len(history[keys[dict[genre]]])):
#     item = [history[keys[dict[genre]]][i], history[keys[dict[genre] + 5]][i]]
#     ar.append(item)
# chart_data = pd.DataFrame(ar, columns=[genre, "val_" + genre])
#
# # st.line_chart(data=None, *, x=None, y=None, color=None, width=0, height=0, use_container_width=True)
# st.line_chart(chart_data, color=["#f5ad42", "#42b9f5"])