import streamlit as st
from audiorecorder import audiorecorder
from datetime import datetime
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image
import pytz
import unicodedata
import re
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import pandas as pd
import parselmouth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
import joblib

service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive.file'])
drive_folder_id = st.secrets["DRIVE_FOLDER_ID"]  # Get from Streamlit secrets
service = build('drive', 'v3', credentials=creds)
vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')

def extract_info(input_string):
  parts = input_string.split("_")

  name = None
  status = None
  gender = None
  age = None
  yod = None

  if parts[0] != '0':
      status = 1
      name = parts[0]
      gender = parts[1]
      age = datetime.now().year - int(parts[2])
      yod = int(parts[3])
  else:
      status = 0
      name = parts[1]
      gender = parts[2]
      age = datetime.now().year - int(parts[3])

  return {
      "name": name,
      "gender": 1 if gender == 'Nam' else 0,
      "age": age,
      "yod": yod,
      "status": status
  }

def sort_dataframe_by_columns(df, columns=None, ascending=True):
    if columns is None:
        columns = df.columns.tolist()  # Sort by all columns

    sorted_df = df.sort_values(by=columns, ascending=ascending)
    return sorted_df
    
def extract_features(audio_file):
    try:
        file_name = os.path.basename(audio_file)
        info = extract_info(file_name)

        # Load audio files into a waveform (y) and the sample rate (sr)
        # y_parkinson and y_normal are 1D NumPy arrays representing the waveform amplitudes
        # sr_parkinson and sr_normal are the respective sampling rates (e.g., 22050 Hz by default)
        y_au, sr_au = librosa.load(audio_file,sr=48000,mono=True)

        # harmonization
        # Trim silence from beginning and end
        # y_au, _ = librosa.effects.trim(y_au, top_db=20)
        # Normalize volume (peak normalization)
        # y_au = y_au / np.max(np.abs(y_au))

        # Feature Extraction using librosa (Example: MFCCs)
        mfccs_au = librosa.feature.mfcc(y=y_au, sr=sr_au, n_mfcc=13)

        # Basic Comparison of Average MFCCs
        avg_mfccs_au = np.mean(mfccs_au, axis=1)

        # Feature Extraction using parselmouth (Example: Pitch, Jitter, Shimmer)
        # Convert audio to Parselmouth Sound objects
        sound_au = parselmouth.Sound(audio_file)

        # Extract pitch (f0)
        pitch_au = sound_au.to_pitch()

        # Extract jitter and shimmer (requires pulse detection)
        point_process_au = parselmouth.praat.call([sound_au, pitch_au], "To PointProcess (cc)")

        jitter_au = parselmouth.praat.call(point_process_au, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.32) * 100
        # Corrected: Extract shimmer from the Sound object directly
        shimmer_au = parselmouth.praat.call([sound_au, point_process_au], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.32, 1.6) * 100
        # print(f"Jitter={jitter_au:.2f}%, Shimmer={shimmer_au:.2f}%")

        # Harmonicity (HNR - Harmonics-to-Noise Ratio)
        hnr_au = parselmouth.praat.call(sound_au, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_db_au = parselmouth.praat.call(hnr_au, "Get mean", 0, 0)
        # print(f"\nHarmonicity (HNR): {hnr_db_au:.2f} dB")

        # Zero Crossing Rate (tần số âm thanh thay đổi cực tính)
        zcr_au = np.mean(librosa.feature.zero_crossing_rate(y_au))

        # Spectral Centroid (trung tâm phổ)
        centroid_au = np.mean(librosa.feature.spectral_centroid(y=y_au, sr=sr_au))

        # Spectral Bandwidth
        bw_au = np.mean(librosa.feature.spectral_bandwidth(y=y_au, sr=sr_au))

        # print(f"ZCR: {zcr_au:.4f}")
        # print(f"Spectral Centroid: {centroid_au:.2f}")
        # print(f"Bandwidth: {bw_au:.2f}")

        features = {
            "file": file_name,
            "name": info["name"],
            "gender": info["gender"],
            "age": info["age"],
            "yod": info["yod"],
            "jitter": jitter_au,
            "shimmer": shimmer_au,
            "hnr": hnr_db_au,
            "zcr": zcr_au,
            "centroid": centroid_au,
            "bandwidth": bw_au,
            **{f"mfcc_{i}": avg_mfccs_au[i] for i in range(len(avg_mfccs_au))},
            "status": info["status"]
        }
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None
    
def predict_pd(audio, _name, _gender, _year_of_birth, _phone):
    st.audio(audio.export().read())
    
    utc_now = datetime.now().replace(tzinfo=pytz.utc)
    vietnam_now = utc_now.astimezone(vietnam_timezone)
    timestamp = vietnam_now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    __gender = _gender
    if unicodedata.normalize("NFC", _gender) == "Nữ":
        __gender = "Nu"
    filename = f"{_name}_{__gender}_{_year_of_birth}_{_phone}_{timestamp}_a.wav"

    audio.export(filename, format="wav")
    print(filename)
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    all_features = []
    features = extract_features(filename)
    if features:
        all_features.append(features)
        print(f"Extracted features for {filename}")
        print(features)
    else:
        print(f"Skipping {filename} due to errors.")
    df = pd.DataFrame(all_features)
    print(df)
    
    # clean data
    df.drop(['file','name','status'], axis=1, inplace=True)
    df["yod"] = df["yod"].fillna(0)
    df = df.fillna(df.mean(numeric_only=True))

    # Load the model and scaler
    model_filename = 'logistic_regression_model.pkl'
    scaler_filename = 'scaler.pkl'
    loaded_model = joblib.load(model_filename)
    loaded_scaler = joblib.load(scaler_filename)

    # Predict
    npy_arr = df.to_numpy()
    print('npy_arr:')
    print(npy_arr)
    # index = pd.Index(['gender','age','yod','jitter','shimmer','hnr','zcr','centroid','bandwidth','mfcc_0','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11','mfcc_12'])
    index = ['gender','age','yod','jitter','shimmer','hnr','zcr','centroid','bandwidth','mfcc_0','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11','mfcc_12']
    new_data = pd.DataFrame(npy_arr, columns=index)
    # new_data = pd.DataFrame(npy_arr, columns=loaded_scaler.feature_names_in_)
    new_data_scaled = loaded_scaler.transform(new_data)
    predictions = loaded_model.predict(new_data_scaled)
    print("\nPredictions using loaded model:\n", predictions)

    file_metadata = {
        'name': filename,
        'parents': [drive_folder_id]
    }

    media = MediaFileUpload(filename, mimetype='audio/wav')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Ghi âm '{filename}' đã được lưu vào Google Drive")
    print(f"File ID: {file.get('id')}")

    # Clean up the local file after upload
    os.remove(filename)
    return predictions

st.markdown(
    """
    <link rel="icon" href="favicon.ico" type="image/x-icon">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .st-emotion-cache-1v0mbdj.e115fcil1 {display: none;}  /* Streamlit Cloud profile footer */
        h1 {
            font-size: 24px;  /* Adjust the size here */
        }
        [data-testid="stColumn"] {
            padding: 0px !important;
        }
        label {
            font-size: 14px; /* Reduced label font size */
            margin-bottom: 2px; /* Reduced margin */
        }
        input, [data-baseweb="input"], [data-baseweb="input-container"] {
            font-size: 14px; /* Reduced input font size */
            padding: 4px; /* Reduced padding */
            margin-bottom: 4px; /* Reduced margin */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

logo = Image.open("logo_new.png")
col1a, col2a = st.columns([2, 4])  # Điều chỉnh tỷ lệ cột tùy ý
with col1a:
    st.image(logo, width=300)
with col2a:
    st.subheader("CHẨN ĐOÁN BỆNH PARKINSON QUA GIỌNG NÓI")
st.write("""
         Giới thiệu: đây là 1 đồ án nghiên cứu, giọng nói ông/bà cô/chú anh/chị sẽ được lưu lại nhằm mục đích nâng cao kết quả nghiên cứu.
         """)
st.markdown("THÔNG TIN CÁ NHÂN:")

col1, col2 = st.columns([1, 2])
with col1:
    st.write("Họ tên:")
with col2:
    name = st.text_input("name_input", key="name_input", label_visibility="collapsed")

col7, col8 = st.columns([1, 2])
with col7:
    st.write("Giới tính:")
with col8:
    gender = st.radio("gender_input", ['Nam', 'Nữ'], key="gender_input", label_visibility="collapsed")

col3, col4 = st.columns([1, 2])
with col3:
    st.write("Năm sinh:")
with col4:
    year_of_birth = st.number_input("yob_input", value=1960, min_value=1900, max_value=2025, step=1, key="yob_input", label_visibility="collapsed")

phone = '0908123456'

# Khởi tạo trạng thái
if "recording" not in st.session_state:
    st.session_state.recording = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st.markdown("---")
st.markdown("NỘI DUNG CHẨN ĐOÁN:")

st.write("Hít nhẹ và phát âm nguyên âm “A” thật to, đều, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá...")
audio1 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam1")

if len(audio1) > 0:
    with st.spinner("Đang phân tích..."):
        predict = predict_pd(audio1, name, gender, year_of_birth, phone)
        print(f"Predict: {predict}")
        if predict[0] == 0:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh thấp")
        else:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh cao")

st.markdown("---")
st.write("Lời cảm ơn: Xin cảm ơn ông/bà cô/chú anh/chị Cộng Đồng PARKINTON VIỆT NAM, đặc biệt là anh admin Tung Mix vì đã hỗ trợ em thực hiện đồ án này!")
logo2 = Image.open("logo2.png")
st.image(logo2)
