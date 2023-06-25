
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
import os
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import pandas as pd
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split as ts 
from scipy.io import wavfile
import librosa
from sklearn.svm import SVC as svc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from openpyxl import Workbook
import wave
import random
from pydub import AudioSegment
import pandas as pd
from openpyxl import load_workbook
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import accuracy_score
import os
from pydub import AudioSegment
import os
import tensorflow as tf
import tqdm
from keras.models import Sequential
from keras.layers  import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier




def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"], key="file_uploader")
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
        st.write(file_details)
        with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
            f.write(uploaded_file.getbuffer())
        if uploaded_file.type == "audio/mp4":
            mp4_audio = AudioSegment.from_file(os.path.join("tempDir",uploaded_file.name), format="mp4")
            wav_filename = os.path.splitext(uploaded_file.name)[0] + ".wav"
            mp4_audio.export(os.path.join("tempDir",wav_filename), format="wav")
            st.success(f"Konvertierte Datei: {wav_filename}")
        else:
            st.success("Hochgeladene Datei ist bereits im WAV-Format")
    else:
        for file in os.listdir("tempDir"):
            os.remove(os.path.join("tempDir", file))
            

upload_and_convert()














