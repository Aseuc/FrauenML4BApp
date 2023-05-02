import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from io import StringIO
import pickle
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

d = None
max_files = 1
dfs = []
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
            dfs.append(df1)
        pass



if dfs: 
    st.write(dfs[0])

    st.write("Stimmenerkennung:")
    data = pd.read_excel('TrainingDataSpeechRecognition.xlsx')
    df = pd.DataFrame(data)
    shuffled_df = df.sample(frac=1)
    st.write(shuffled_df)
    np.random.seed(42)
    X = shuffled_df.drop('target', axis=1)
    y = shuffled_df["target"]

    clf = None
    file = "savedmodel.sav"
    load_model = pickle.load(open(file, 'rb'))
    pred1 = None
    knn = KNeighborsClassifier(n_neighbors=100)
    knn.fit(X, y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    pred1 = clf.predict(dfs[0])
    pred2 = knn.predict(dfs[0])
    st.write("Random Forest Classifier:")
    st.write(pred1)
    st.write("K nearest neighbors:")
    st.write(pred2)

    # model = Sequential()
    # model.add(SimpleRNN(10,input_shape=(None,5)))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binars_crossentropy', optimizer='adam', metrics=['accuracy'])
    # X_train = shuffled_df.drop('target',axis=1)
    # y_train = shuffled_df["target"]
    # model.fit(X_train, y_train, epochs=10)
    # newData = dfs[0]
    # pred3 = model.predict(newData)
    # st.write(pred3)














    # if load_model is None:
    #      X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)
    #      clf = RandomForestClassifier(n_estimators=100)
    #      clf.fit(X_train, y_train)
    #      filename = 'newFile.sav'
    #      pickle.dump(clf, open(filename, 'wb'))
    #      pred1 = clf.predict(merged_df2)
    # else:
    #      pred1 = load_model.predict(merged_df2)

    # result = pd.DataFrame({'Vorhersage': pred1})
    # merged_df3 = pd.merge(merged_df2, result, left_index=True, right_index=True)
    # merged_df3Download = merged_df3



   