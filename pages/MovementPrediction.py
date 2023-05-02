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
max_files = 3
dfs = []
merged_df2 = None
uploaded_files = st.file_uploader(
    "3 Dateien des Sensor Loggers bitte hochladen in folgender Reihenfolge: Acceleratoren-, "
    "Gravitydaten und Gyrosscopedaten!:",
    type={"csv"},
    accept_multiple_files=True)
i = 1
counter = 0
if uploaded_files is not None:
    for up in uploaded_files:
        st.write(f'Datei Nr. {i}:', up.name)
        i = i + 1
    if len(uploaded_files) > max_files:
        st.error(f'Bitte lade nicht mehr als 3 Dateien hoch!')
    elif len(uploaded_files) < max_files:
        st.error(f'Bitte lade mindestens 3 Dateien hoch!')
    else:

        for uploaded_file in uploaded_files:
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(string_data)
            df1 = pd.DataFrame(df)
            df1 = df1.drop(columns=['time', 'seconds_elapsed'])
            # st.write("df1")
            # st.write(df1)
            if counter == 0:
                df1 = df1.rename(columns={'z': 'acc_z', 'y': 'acc_y', 'x': 'acc_x'})
            elif counter == 1:
                df1 = df1.rename(columns={'z': 'gravity_z', 'y': 'gravity_y', 'x': 'gravity_x'})
            elif counter == 2:
                df1 = df1.rename(columns={'z': 'gyro_z', 'y': 'gyro_y', 'x': 'gyro_x'})

            counter = counter + 1
            dfs.append(df1)

            # merged_df = pd.merge(dfs[0],dfs[1],left_index=True, right_index=True)
            # st.write(merged_df)
        pass

    if dfs:
        # st.write(dfs[0])
        # st.write(dfs[1])
        # st.write(dfs[2])
        merged_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
        # st.write(merged_df)
        merged_df2 = pd.merge(merged_df, dfs[2], left_index=True, right_index=True)
        html = merged_df2.to_html()
        html = f"""<style> table{{font-size:15px; margin-left:auto; margin-right:auto;}}</style>{html}"""
        # st.write(html,unsafe_allow_html=True)
        st.write("Acceleratoren Daten:")
        colacc_x, colacc_y, colacc_z = st.columns(3)
        acc_x = colacc_x.checkbox("acc_x verbergen")
        acc_y = colacc_y.checkbox('acc_y verbergen')
        acc_z = colacc_z.checkbox('acc_z verbergen')
        if acc_z and acc_y and acc_z:
            st.write("Nix zu zeigen!")
        elif acc_x and acc_y:
            st.line_chart(dfs[0].drop(columns=['acc_x', 'acc_y']))
        elif acc_x and acc_z:
            st.line_chart(dfs[0].drop(columns=['acc_x', 'acc_z']))
        elif acc_z and acc_y:
            st.line_chart(dfs[0].drop(columns=['acc_y', 'acc_z']))
        elif acc_y:
            st.line_chart(dfs[0].drop('acc_y', axis=1))
        elif acc_x:
            st.line_chart(dfs[0].drop('acc_x', axis=1))
        elif acc_z:
            st.line_chart(dfs[0].drop('acc_z', axis=1))
        else:
            st.line_chart(dfs[0])
        # elif acc_y:
        #     st.line_chart(dfs[0].drop('acc_y',axis=))
        dfs1 = 1
        gy = "gravity_y"
        gz = "gravity_z"
        gx = "gravity_x"
        st.write("Gravity Daten:")
        colgrav_x, colgrav_y, colgrav_z = st.columns(3)
        grav_x = colgrav_x.checkbox('gravity_x verbergen')
        grav_y = colgrav_y.checkbox('gravity_y verbergen')
        grav_z = colgrav_z.checkbox('gravity_z verbergen')

        if grav_x and grav_y and grav_z:
            st.write("Nix zu zeigen!")
            # st.line_chart(dfs[dfs1].drop(columns=[gy, gz, gx]))
        elif grav_x and grav_y:
            st.line_chart(dfs[dfs1].drop(columns=[gy, gx]))
        elif grav_x and grav_z:
            st.line_chart(dfs[dfs1].drop(columns=[gx, gz]))
        elif grav_z and grav_y:
            st.line_chart(dfs[dfs1].drop(columns=[gz, gy]))
        elif grav_y:
            st.line_chart(dfs[dfs1].drop(gy, axis=1))
        elif grav_x:
            st.line_chart(dfs[dfs1].drop(gx, axis=1))
        elif grav_z:
            st.line_chart(dfs[dfs1].drop(gz, axis=1))
        else:
            st.line_chart(dfs[dfs1])

        grx = "gyro_x"
        gry = "gyro_y"
        grz = "gyro_z"
        dfs2 = 2
        st.write("Gyrosscope Daten:")
        colgyro_x, colgyro_y, colgyro_z = st.columns(3)
        gyro_x = colgyro_x.checkbox('gyro_x verbergen')
        gyro_y = colgyro_y.checkbox('gyro_y verbergen')
        gyro_z = colgyro_z.checkbox('gyro_z verbergen')
        if gyro_x and gyro_y and gyro_z:
            st.write("Nix zu zeigen!")
            # st.line_chart(dfs[dfs1].drop(columns=[gy, gz, gx]))
        elif gyro_x and gyro_y:
            st.line_chart(dfs[dfs2].drop(columns=[gry, grx]))
        elif gyro_x and gyro_z:
            st.line_chart(dfs[dfs2].drop(columns=[grx, grz]))
        elif gyro_z and gyro_y:
            st.line_chart(dfs[dfs2].drop(columns=[grz, gry]))
        elif gyro_y:
            st.line_chart(dfs[dfs2].drop(gry, axis=1))
        elif gyro_x:
            st.line_chart(dfs[dfs2].drop(grx, axis=1))
        elif gyro_z:
            st.line_chart(dfs[dfs2].drop(grz, axis=1))
        else:
            st.line_chart(dfs[dfs2])

    else:
        pass

merged_df3Download = None
barDF = None


def trainModelOnAccData():
    acc_data = pd.read_excel('/pages/AcceleratorTrainingsData.xlsx')
    acc_df = pd.DataFrame(data)
    acc_df_sample = df.sample(frac=1)
    # st.write(df)
    np.random.seed(42)
    X = shuffled_df.drop('target', axis=1)
    y = shuffled_df["target"]


if dfs:
    "Dataframe mit Vorhersagen:"
    data = pd.read_excel('pages/TestDataFinalCorrected.xlsx')
    df = pd.DataFrame(data)
    shuffled_df = df.sample(frac=1)
    # st.write(df)
    np.random.seed(42)
    X = shuffled_df.drop('target', axis=1)
    y = shuffled_df["target"]
    # st.write(y)
    clf = None
    file = "pages/savedmodel.sav"
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
    barDF = merged_df3
    st.write(merged_df3)
    drop_col_for_select = merged_df3.drop(
        columns=["acc_y", "acc_x", "acc_z", "gyro_z", "gyro_x", "gyro_y", "gravity_z", "gravity_y", "gravity_x"])
    select_x1 = drop_col_for_select.loc[merged_df3['Vorhersage'] == "rennend"]
    select_x2 = drop_col_for_select.loc[merged_df3['Vorhersage'] == "stehend"]
    select_x3 = drop_col_for_select.loc[merged_df3['Vorhersage'] == "fallend"]

    x1 = select_x1.count()
    x2 = select_x2.count()
    x3 = select_x3.count()

    x1_rename = x1.rename(index={'Vorhersage': 'rennend'})
    x2_rename = x2.rename(index={'Vorhersage': "stehend"})
    x3_rename = x3.rename(index={'Vorhersage': 'fallend'})
    colx1, colx2, colx3 = st.columns(3)
    colx1.write(x1_rename)
    colx2.write(x2_rename)
    colx3.write(x3_rename)

    # st.write(x1_rename)
    # st.write(x2_rename)
    # st.write(x3_rename)

    # seriesx1 = pd.list(select_x1).value_counts()
    # seriesx2 = pd.List(select_x2).value_counts()
    # seriesx3 = pd.List(select_x3).value_counts()


    # hist_data = [seriesx1,seriesx2,seriesx3]
    # group_labels = ['Stehend', 'Falled', 'Rennend']
    # fig = ff.create_distplot(hist_data, group_labels)
    #
    # st.plotly_chart(fig, use_container_width=True)

buffer = io.BytesIO()


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


# st.markdown("""<href>.st-bb{margin-right:0px}</style>""", unsafe_allow_html=True)
if merged_df3Download is not None:
    options = ["CSV", "xlsx"]
    filetype = st.selectbox("Wähle Dateiart für den Download:", options)

    if filetype == "CSV":
        csv = convert_df(merged_df3Download)

        b1 = st.download_button(
            label="Download CSV",
            data=csv,
            file_name='MovementPrediction.csv',
            mime='text/csv',
        )
    elif filetype == "xlsx":
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            merged_df3Download.to_excel(writer, sheet_name="Sheet1", index=False)
            writer.save()
            download2 = st.download_button(label="Download xlsx", data=buffer, file_name="MovementPrediction.xlsx",
                                           mime='application/vnd.ms-excel')
