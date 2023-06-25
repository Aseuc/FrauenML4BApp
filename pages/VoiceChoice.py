
import streamlit as st
from pydub import AudioSegment
import os


def upload_and_convert():
    uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hochladen aus", type=["mp4", "wav"])
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

















