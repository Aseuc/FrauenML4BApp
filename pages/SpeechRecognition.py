
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from io import StringIO
import io
import pickle
from sklearn import datasets
from sklearn import svm
import altair as alt


d = None
max_files = 1
dfs = []
merged_df2 = None
uploaded_files = st.file_uploader(
    "Lade eine Microphone.csv aus deiner Sensor Logger App hoch!",
    type={"csv"},
    accept_multiple_files=True)
i = 1
counter = 0
if uploaded_files is not None:
    for up in uploaded_files:
        st.write(f'Datei Nr. {i}:', up.name)
        i = i + 1
    if len(uploaded_files) > max_files:
        st.error(f'Bitte lade nicht mehr als 1 Dateien hoch!')
    elif len(uploaded_files) < max_files:
        st.error(f'Bitte lade mindestens 1 Dateien hoch!')
    else:

        for uploaded_file in uploaded_files:
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(string_data)
            df1 = pd.DataFrame(df)
            # st.write("df1")
            # st.write(df1)
            dfs.append(df1)

            # merged_df = pd.merge(dfs[0],dfs[1],left_index=True, right_index=True)
            # st.write(merged_df)
        pass

   