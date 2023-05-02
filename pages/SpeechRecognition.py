
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

if dfs: 
    st.write(dfs[0])

    st.write("Stimmenerkennung:")
    data = pd.read_excel('MicroDataLabled.xlsx')
    df = pd.DataFrame(data)
    shuffled_df = df.sample(frac=1)
    # st.write(df)
    np.random.seed(42)
    X = shuffled_df.drop('target', axis=1)
    y = shuffled_df["target"]
    # st.write(y)
    clf = None
    file = "savedmodel.sav"
    load_model = pickle.load(open(file, 'rb'))
    pred1 = None
    if load_model is None:
        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        filename = 'newFile.sav'
        pickle.dump(clf, open(filename, 'wb'))
        pred1 = clf.predict(merged_df2)
    else:
        pred1 = load_model.predict(merged_df2)

    result = pd.DataFrame({'Vorhersage': pred1})
    merged_df3 = pd.merge(merged_df2, result, left_index=True, right_index=True)
    merged_df3Download = merged_df3



   